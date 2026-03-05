import argparse
import html
import json
import mimetypes
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote, urlparse

import cv2
from ultralytics import YOLO


DEFAULT_ELECTRONIC_CLASSES = [
    "cell phone",
    "laptop",
    "tv",
    "remote",
    "keyboard",
    "mouse",
    "microwave",
    "oven",
    "toaster",
    "refrigerator",
    "clock",
    "hair drier",
]

DEFAULT_WEAPON_CLASSES = ["knife"]


@dataclass
class StreamConfig:
    name: str
    url: str
    room: str = "default-room"
    enabled: bool = True

    @property
    def key(self) -> str:
        return f"{self.room}/{self.name}"

    @property
    def display_name(self) -> str:
        return self.key


@dataclass
class AppConfig:
    model_path: str
    output_dir: Path
    confidence: float
    iou: float
    device: Optional[str]
    reconnect_delay_seconds: int
    save_cooldown_seconds: int
    inference_workers: int
    per_camera_min_detection_interval_ms: int
    detection_max_width: Optional[int]
    detection_max_height: Optional[int]
    control_host: str
    control_port: int
    pull_interval_seconds: int
    live_timeout_seconds: int
    pull_warmup_frames: int
    electronic_classes: List[str]
    weapon_classes: List[str]
    streams: List[StreamConfig]


@dataclass
class DetectionRecord:
    label: str
    confidence: float
    xyxy: Tuple[float, float, float, float]


@dataclass
class FrameSnapshot:
    frame_id: int
    captured_at: datetime
    detection_frame: Any
    evidence_frame: Any


@dataclass
class CameraState:
    stream: StreamConfig
    last_saved_at: float = 0.0
    latest_frame_id: int = 0
    queued_or_running: bool = False
    last_scheduled_at: float = 0.0
    latest_captured_at: Optional[datetime] = None
    latest_detection_frame: Any = field(default=None, repr=False)
    latest_evidence_frame: Any = field(default=None, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def store_frame(self, detection_frame, evidence_frame, min_interval_seconds: float) -> bool:
        with self.lock:
            captured_at = datetime.now()
            self.latest_frame_id += 1
            self.latest_captured_at = captured_at
            self.latest_detection_frame = detection_frame
            self.latest_evidence_frame = evidence_frame

            now = time.time()
            if self.queued_or_running:
                return False

            if now - self.last_scheduled_at < min_interval_seconds:
                return False

            self.queued_or_running = True
            self.last_scheduled_at = now
            return True

    def snapshot(self) -> Optional[FrameSnapshot]:
        with self.lock:
            if (
                self.latest_detection_frame is None
                or self.latest_evidence_frame is None
                or self.latest_captured_at is None
            ):
                return None

            return FrameSnapshot(
                frame_id=self.latest_frame_id,
                captured_at=self.latest_captured_at,
                detection_frame=self.latest_detection_frame.copy(),
                evidence_frame=self.latest_evidence_frame.copy(),
            )

    def finish_inference(self, processed_frame_id: int, min_interval_seconds: float) -> bool:
        with self.lock:
            self.queued_or_running = False

            now = time.time()
            if self.latest_frame_id == processed_frame_id:
                return False

            if now - self.last_scheduled_at < min_interval_seconds:
                return False

            self.queued_or_running = True
            self.last_scheduled_at = now
            return True

    def can_save_now(self, cooldown_seconds: int) -> bool:
        with self.lock:
            return time.time() - self.last_saved_at >= cooldown_seconds

    def mark_saved_now(self) -> None:
        with self.lock:
            self.last_saved_at = time.time()


class CaptureModeController:
    def __init__(self, pull_interval_seconds: int, live_timeout_seconds: int) -> None:
        self._pull_interval_seconds = pull_interval_seconds
        self._live_timeout_seconds = live_timeout_seconds
        self._live_until = 0.0
        self._version = 0
        self._condition = threading.Condition()

    def get_status(self) -> Dict[str, Any]:
        with self._condition:
            return self._build_status_locked()

    def wait_for_change(self, last_version: int, timeout: float) -> bool:
        with self._condition:
            if self._version != last_version:
                return True

            self._condition.wait(timeout)
            return self._version != last_version

    def update(
        self,
        mode: Optional[str] = None,
        pull_interval_seconds: Optional[int] = None,
        live_timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        normalized_mode = None if mode is None else mode.strip().lower()
        if normalized_mode not in {None, "live", "pull"}:
            raise ValueError("mode must be 'live' or 'pull'")

        with self._condition:
            updated = False

            if pull_interval_seconds is not None:
                if pull_interval_seconds <= 0:
                    raise ValueError("pull_interval_seconds must be greater than 0")
                self._pull_interval_seconds = int(pull_interval_seconds)
                updated = True

            if live_timeout_seconds is not None:
                if live_timeout_seconds <= 0:
                    raise ValueError("live_timeout_seconds must be greater than 0")
                self._live_timeout_seconds = int(live_timeout_seconds)
                updated = True

            if normalized_mode == "live":
                self._live_until = time.time() + self._live_timeout_seconds
                updated = True
            elif normalized_mode == "pull":
                self._live_until = 0.0
                updated = True

            if not updated:
                raise ValueError(
                    "Provide at least one of mode, pull_interval_seconds, or live_timeout_seconds"
                )

            self._version += 1
            status = self._build_status_locked()
            self._condition.notify_all()

        print(
            f"[CONTROL] effective_mode={status['effective_mode']} "
            f"pull_interval_seconds={status['pull_interval_seconds']} "
            f"live_timeout_seconds={status['live_timeout_seconds']} "
            f"live_until={status['live_until']}"
        )
        return status

    def _build_status_locked(self) -> Dict[str, Any]:
        now = time.time()
        live_seconds_remaining = max(0, int(self._live_until - now))
        effective_mode = "live" if live_seconds_remaining > 0 else "pull"
        live_until = None
        if effective_mode == "live":
            live_until = datetime.fromtimestamp(self._live_until).isoformat(timespec="seconds")

        return {
            "effective_mode": effective_mode,
            "pull_interval_seconds": self._pull_interval_seconds,
            "live_timeout_seconds": self._live_timeout_seconds,
            "live_seconds_remaining": live_seconds_remaining,
            "live_until": live_until,
            "controller_version": self._version,
        }


def build_control_handler(mode_controller: CaptureModeController, served_dir: Path):
    served_root = served_dir.resolve()

    class ControlHandler(BaseHTTPRequestHandler):
        server_version = "RtspControl/1.0"

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._send_common_headers()
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/mode", "/status"}:
                self._send_json(200, mode_controller.get_status())
                return

            if parsed.path == "/files" or parsed.path.startswith("/files/"):
                self._handle_files_request(parsed.path)
                return

            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path not in {"/mode", "/mode/live", "/mode/pull"}:
                self._send_json(404, {"error": "not found"})
                return

            try:
                payload = self._read_payload()
                if parsed.path == "/mode/live":
                    payload["mode"] = "live"
                elif parsed.path == "/mode/pull":
                    payload["mode"] = "pull"

                status = mode_controller.update(
                    mode=payload.get("mode"),
                    pull_interval_seconds=parse_optional_positive_int(
                        payload.get("pull_interval_seconds"),
                        "pull_interval_seconds",
                    ),
                    live_timeout_seconds=parse_optional_positive_int(
                        payload.get("live_timeout_seconds"),
                        "live_timeout_seconds",
                    ),
                )
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return

            self._send_json(200, status)

        def log_message(self, fmt: str, *args) -> None:
            print(f"[HTTP] {self.address_string()} {fmt % args}")

        def _read_payload(self) -> Dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0") or 0)
            if content_length <= 0:
                return {}

            raw = self.rfile.read(content_length).decode("utf-8").strip()
            if not raw:
                return {}

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {"mode": raw}

            if not isinstance(parsed, dict):
                raise ValueError("Request body must be a JSON object or plain text mode")

            return parsed

        def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self._send_common_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_common_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _handle_files_request(self, request_path: str) -> None:
            target_path = self._resolve_requested_path(request_path)
            if target_path is None:
                self.send_error(403, "Access denied")
                return

            if not target_path.exists():
                self.send_error(404, "Path not found")
                return

            if target_path.is_dir():
                self._send_directory_listing(target_path)
                return

            self._send_file(target_path)

        def _resolve_requested_path(self, request_path: str) -> Optional[Path]:
            relative_url_path = ""
            if request_path == "/files":
                relative_url_path = ""
            elif request_path.startswith("/files/"):
                relative_url_path = request_path[len("/files/") :]
            else:
                return None

            resolved = (served_root / unquote(relative_url_path)).resolve()
            try:
                resolved.relative_to(served_root)
            except ValueError:
                return None
            return resolved

        def _send_directory_listing(self, dir_path: Path) -> None:
            try:
                children = sorted(dir_path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
            except OSError as exc:
                self.send_error(500, f"Failed to list directory: {exc}")
                return

            relative_dir = ""
            if dir_path != served_root:
                relative_dir = dir_path.relative_to(served_root).as_posix()

            title_path = "/" if not relative_dir else f"/{relative_dir}"
            links: List[str] = []

            if dir_path != served_root:
                parent = dir_path.parent
                parent_rel = ""
                if parent != served_root:
                    parent_rel = parent.relative_to(served_root).as_posix()
                parent_href = "/files/" + quote(parent_rel, safe="/")
                links.append(f'<li><a href="{parent_href}">../</a></li>')

            for child in children:
                child_rel = child.relative_to(served_root).as_posix()
                href = "/files/" + quote(child_rel, safe="/")
                display_name = child.name
                if child.is_dir():
                    href += "/"
                    display_name += "/"
                links.append(f'<li><a href="{href}">{html.escape(display_name)}</a></li>')

            body = (
                "<!DOCTYPE html>\n"
                "<html>\n"
                "<head><meta charset=\"utf-8\"><title>Index of "
                f"{html.escape(title_path)}"
                "</title></head>\n"
                "<body>\n"
                f"<h1>Index of {html.escape(title_path)}</h1>\n"
                "<ul>\n"
                + "\n".join(links)
                + "\n</ul>\n"
                "</body>\n"
                "</html>\n"
            )
            encoded = body.encode("utf-8")

            self.send_response(200)
            self._send_common_headers()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_file(self, file_path: Path) -> None:
            try:
                file_size = file_path.stat().st_size
                content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
                self.send_response(200)
                self._send_common_headers()
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(file_size))
                self.send_header("Content-Disposition", f'inline; filename="{file_path.name}"')
                self.end_headers()

                with file_path.open("rb") as handle:
                    while True:
                        chunk = handle.read(64 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except OSError as exc:
                self.send_error(500, f"Failed to read file: {exc}")

    return ControlHandler


class ControlServer:
    def __init__(
        self,
        host: str,
        port: int,
        mode_controller: CaptureModeController,
        served_dir: Path,
    ) -> None:
        handler = build_control_handler(mode_controller, served_dir)
        self._server = ThreadingHTTPServer((host, port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="control-server", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


class RtspCaptureWorker:
    def __init__(
        self,
        state: CameraState,
        config: AppConfig,
        ready_queue: "queue.Queue[str]",
        stop_event: threading.Event,
        mode_controller: CaptureModeController,
        initial_pull_delay_seconds: float,
    ) -> None:
        self.state = state
        self.config = config
        self.ready_queue = ready_queue
        self.stop_event = stop_event
        self.mode_controller = mode_controller
        self.initial_pull_delay_seconds = initial_pull_delay_seconds
        self._thread = threading.Thread(
            target=self._run,
            name=(
                f"capture-{sanitize_path_component(self.state.stream.room)}-"
                f"{sanitize_path_component(self.state.stream.name)}"
            ),
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        min_interval_seconds = self.config.per_camera_min_detection_interval_ms / 1000.0
        next_pull_at = time.time() + self.initial_pull_delay_seconds
        previous_mode: Optional[str] = None

        while not self.stop_event.is_set():
            status = self.mode_controller.get_status()
            effective_mode = status["effective_mode"]

            if effective_mode == "live":
                previous_mode = "live"
                next_pull_at = time.time()
                self._run_live_mode(min_interval_seconds)
                continue

            if previous_mode == "live":
                next_pull_at = time.time()
            previous_mode = "pull"

            now = time.time()
            if now >= next_pull_at:
                success = self._pull_once(min_interval_seconds)
                delay_seconds = (
                    status["pull_interval_seconds"] if success else min(
                        self.config.reconnect_delay_seconds,
                        status["pull_interval_seconds"],
                    )
                )
                next_pull_at = time.time() + delay_seconds
                continue

            self.mode_controller.wait_for_change(
                status["controller_version"],
                min(1.0, next_pull_at - now),
            )

    def _run_live_mode(self, min_interval_seconds: float) -> None:
        while not self.stop_event.is_set():
            if self.mode_controller.get_status()["effective_mode"] != "live":
                return

            capture = open_video_capture(self.state.stream.url)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not capture.isOpened():
                print(
                    f"[WARN] Could not open stream '{self.state.stream.display_name}' in live mode. "
                    f"Retrying in {self.config.reconnect_delay_seconds}s."
                )
                capture.release()
                if self._wait_with_mode_checks(self.config.reconnect_delay_seconds):
                    return
                continue

            print(f"[INFO] Live mode connected to stream '{self.state.stream.display_name}'.")

            while not self.stop_event.is_set():
                if self.mode_controller.get_status()["effective_mode"] != "live":
                    capture.release()
                    return

                ok, frame = capture.read()
                if not ok or frame is None:
                    print(f"[WARN] Lost live stream '{self.state.stream.display_name}'. Reconnecting.")
                    break

                detection_frame = prepare_detection_frame(
                    frame,
                    self.config.detection_max_width,
                    self.config.detection_max_height,
                )

                if self.state.store_frame(detection_frame, frame, min_interval_seconds):
                    self.ready_queue.put(self.state.stream.key)

            capture.release()
            if self._wait_with_mode_checks(self.config.reconnect_delay_seconds):
                return

    def _pull_once(self, min_interval_seconds: float) -> bool:
        capture = open_video_capture(self.state.stream.url)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not capture.isOpened():
            print(
                f"[WARN] Could not open stream '{self.state.stream.display_name}' in pull mode. "
                f"Retrying later."
            )
            capture.release()
            return False

        frame = None
        for _ in range(self.config.pull_warmup_frames):
            ok, candidate = capture.read()
            if ok and candidate is not None:
                frame = candidate

        capture.release()

        if frame is None:
            print(f"[WARN] Pull mode did not get a frame from '{self.state.stream.display_name}'.")
            return False

        detection_frame = prepare_detection_frame(
            frame,
            self.config.detection_max_width,
            self.config.detection_max_height,
        )

        if self.state.store_frame(detection_frame, frame, min_interval_seconds):
            self.ready_queue.put(self.state.stream.key)

        return True

    def _wait_with_mode_checks(self, timeout_seconds: int) -> bool:
        deadline = time.time() + timeout_seconds
        while not self.stop_event.is_set():
            if self.mode_controller.get_status()["effective_mode"] != "live":
                return True

            remaining = deadline - time.time()
            if remaining <= 0:
                return False

            status = self.mode_controller.get_status()
            self.mode_controller.wait_for_change(status["controller_version"], min(1.0, remaining))

        return True


class InferenceWorker:
    def __init__(
        self,
        worker_id: int,
        config: AppConfig,
        states: Dict[str, CameraState],
        ready_queue: "queue.Queue[str]",
        stop_event: threading.Event,
        target_class_ids: List[int],
    ) -> None:
        self.worker_id = worker_id
        self.config = config
        self.states = states
        self.ready_queue = ready_queue
        self.stop_event = stop_event
        self.target_class_ids = target_class_ids
        self._thread = threading.Thread(
            target=self._run,
            name=f"inference-{worker_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        try:
            model = YOLO(self.config.model_path)
        except Exception as exc:
            print(f"[ERROR] Worker {self.worker_id} could not load YOLO model: {exc}")
            self.stop_event.set()
            return

        print(f"[INFO] Inference worker {self.worker_id} ready.")

        min_interval_seconds = self.config.per_camera_min_detection_interval_ms / 1000.0

        while not self.stop_event.is_set():
            try:
                stream_key = self.ready_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                state = self.states.get(stream_key)
                if state is None:
                    continue

                snapshot = state.snapshot()
                if snapshot is None:
                    if state.finish_inference(-1, min_interval_seconds):
                        self.ready_queue.put(stream_key)
                    continue

                detections = detect_objects(model, snapshot.detection_frame, self.target_class_ids, self.config)
                if detections and state.can_save_now(self.config.save_cooldown_seconds):
                    annotated = annotate_evidence(
                        snapshot.evidence_frame,
                        snapshot.detection_frame,
                        detections,
                        snapshot.captured_at,
                    )
                    image_path, text_path = save_evidence(
                        self.config.output_dir,
                        state.stream.room,
                        state.stream.name,
                        snapshot.captured_at,
                        annotated,
                        detections,
                    )
                    labels = [detection.label for detection in detections]
                    state.mark_saved_now()
                    print(
                        f"[DETECTED] worker={self.worker_id} stream={state.stream.display_name} "
                        f"labels={sorted(set(labels))} image={image_path} text={text_path}"
                    )

                if state.finish_inference(snapshot.frame_id, min_interval_seconds):
                    self.ready_queue.put(stream_key)
            except Exception as exc:
                print(f"[ERROR] Worker {self.worker_id} failed on stream '{stream_key}': {exc}")
                state = self.states.get(stream_key)
                if state is not None and state.finish_inference(-1, min_interval_seconds):
                    self.ready_queue.put(stream_key)
            finally:
                self.ready_queue.task_done()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTSP evidence capture with OpenCV and YOLO.")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the JSON configuration file. Defaults to ./config.json",
    )
    return parser.parse_args()


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    config_dir = path.resolve().parent

    streams: List[StreamConfig] = []
    seen_stream_keys = set()
    for stream in raw.get("streams", []):
        if not stream.get("enabled", True):
            continue

        room = str(stream.get("room", "default-room")).strip() or "default-room"
        name = str(stream["name"]).strip()
        url = str(stream["url"]).strip()
        stream_config = StreamConfig(name=name, url=url, room=room, enabled=True)

        if stream_config.key in seen_stream_keys:
            raise ValueError(f"Duplicate enabled stream '{stream_config.key}' in configuration.")
        seen_stream_keys.add(stream_config.key)
        streams.append(stream_config)

    if not streams:
        raise ValueError("No enabled streams were found in the configuration.")

    return AppConfig(
        model_path=resolve_model_path(config_dir, raw.get("model_path", "yolov8n.pt")),
        output_dir=resolve_path(config_dir, raw.get("output_dir", "evidence")),
        confidence=float(raw.get("confidence", 0.35)),
        iou=float(raw.get("iou", 0.45)),
        device=raw.get("device"),
        reconnect_delay_seconds=int(raw.get("reconnect_delay_seconds", 5)),
        save_cooldown_seconds=int(raw.get("save_cooldown_seconds", 10)),
        inference_workers=max(1, int(raw.get("inference_workers", 2))),
        per_camera_min_detection_interval_ms=max(
            0,
            int(raw.get("per_camera_min_detection_interval_ms", raw.get("detection_interval_ms", 250))),
        ),
        detection_max_width=optional_int(raw.get("detection_max_width", 640)),
        detection_max_height=optional_int(raw.get("detection_max_height")),
        control_host=str(raw.get("control_host", "0.0.0.0")),
        control_port=int(raw.get("control_port", 8080)),
        pull_interval_seconds=max(1, int(raw.get("pull_interval_seconds", 300))),
        live_timeout_seconds=max(1, int(raw.get("live_timeout_seconds", 600))),
        pull_warmup_frames=max(1, int(raw.get("pull_warmup_frames", 1))),
        electronic_classes=list(raw.get("electronic_classes", DEFAULT_ELECTRONIC_CLASSES)),
        weapon_classes=list(raw.get("weapon_classes", DEFAULT_WEAPON_CLASSES)),
        streams=streams,
    )


def optional_int(value) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def parse_optional_positive_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None

    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be greater than 0")
    return parsed


def resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def resolve_model_path(base_dir: Path, value: str) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)

    resolved = (base_dir / candidate).resolve()
    if candidate.parent != Path(".") or resolved.exists():
        return str(resolved)

    return value


def get_model_names(model: YOLO) -> Dict[int, str]:
    raw_names = model.names
    if isinstance(raw_names, dict):
        return {int(class_id): str(name) for class_id, name in raw_names.items()}
    return {class_id: str(name) for class_id, name in enumerate(raw_names)}


def resolve_target_class_ids(model: YOLO, config: AppConfig) -> List[int]:
    model_names = get_model_names(model)
    normalized = {name.lower(): class_id for class_id, name in model_names.items()}
    requested = [name.lower() for name in config.electronic_classes + config.weapon_classes]

    missing = [name for name in requested if name not in normalized]
    if missing:
        print(f"[WARN] These target classes are not in the model and will be ignored: {', '.join(missing)}")

    class_ids = sorted({normalized[name] for name in requested if name in normalized})
    if not class_ids:
        raise ValueError("None of the requested target classes exist in the loaded YOLO model.")

    resolved_names = [str(model_names[class_id]) for class_id in class_ids]
    print(f"[INFO] Watching for classes: {', '.join(resolved_names)}")
    return class_ids


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_path_component(name: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
    return sanitized or "unnamed"


def save_evidence(
    output_dir: Path,
    room_name: str,
    stream_name: str,
    captured_at: datetime,
    annotated_frame,
    detections: List[DetectionRecord],
) -> Tuple[Path, Path]:
    room_dir = output_dir / sanitize_path_component(room_name)
    camera_dir = room_dir / sanitize_path_component(stream_name)
    camera_dir.mkdir(parents=True, exist_ok=True)

    basename = str(int(captured_at.timestamp() * 1000))
    image_path = camera_dir / f"{basename}.jpg"
    text_path = camera_dir / f"{basename}.txt"

    if not cv2.imwrite(str(image_path), annotated_frame):
        raise RuntimeError(f"Failed to save evidence image to {image_path}")

    text_lines = [detection.label for detection in detections]
    text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    return image_path, text_path


def open_video_capture(url: str) -> cv2.VideoCapture:
    ffmpeg_backend = getattr(cv2, "CAP_FFMPEG", None)
    if ffmpeg_backend is not None:
        capture = cv2.VideoCapture(url, ffmpeg_backend)
        if capture.isOpened():
            return capture
        capture.release()

    return cv2.VideoCapture(url)


def prepare_detection_frame(frame, max_width: Optional[int], max_height: Optional[int]):
    height, width = frame.shape[:2]
    scale = 1.0

    if max_width is not None and width > max_width:
        scale = min(scale, max_width / width)

    if max_height is not None and height > max_height:
        scale = min(scale, max_height / height)

    if scale >= 1.0:
        return frame

    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def detect_objects(
    model: YOLO,
    frame,
    target_class_ids: List[int],
    config: AppConfig,
) -> List[DetectionRecord]:
    prediction = model.predict(
        source=frame,
        conf=config.confidence,
        iou=config.iou,
        classes=target_class_ids,
        device=config.device,
        verbose=False,
    )[0]

    if prediction.boxes is None or len(prediction.boxes) == 0:
        return []

    detections: List[DetectionRecord] = []
    for box in prediction.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            DetectionRecord(
                label=str(prediction.names[class_id]),
                confidence=confidence,
                xyxy=(float(x1), float(y1), float(x2), float(y2)),
            )
        )

    return detections


def annotate_evidence(
    evidence_frame,
    detection_frame,
    detections: List[DetectionRecord],
    captured_at: datetime,
):
    annotated = evidence_frame.copy()

    detection_height, detection_width = detection_frame.shape[:2]
    evidence_height, evidence_width = evidence_frame.shape[:2]
    scale_x = evidence_width / detection_width
    scale_y = evidence_height / detection_height

    line_thickness = max(2, min(evidence_width, evidence_height) // 400)
    font_scale = max(0.5, min(evidence_width, evidence_height) / 1200.0)

    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        left = max(0, min(evidence_width - 1, int(x1 * scale_x)))
        top = max(0, min(evidence_height - 1, int(y1 * scale_y)))
        right = max(0, min(evidence_width - 1, int(x2 * scale_x)))
        bottom = max(0, min(evidence_height - 1, int(y2 * scale_y)))

        color = (0, 0, 255) if detection.label == "knife" else (0, 165, 255)
        cv2.rectangle(annotated, (left, top), (right, bottom), color, line_thickness)

        text = f"{detection.label} {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            line_thickness,
        )
        text_top = max(0, top - text_height - baseline - 6)
        text_bottom = min(evidence_height - 1, text_top + text_height + baseline + 6)
        text_right = min(evidence_width - 1, left + text_width + 10)

        cv2.rectangle(annotated, (left, text_top), (text_right, text_bottom), color, thickness=-1)
        cv2.putText(
            annotated,
            text,
            (left + 5, text_bottom - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, line_thickness - 1),
            cv2.LINE_AA,
        )

    timestamp_text = captured_at.strftime("%Y-%m-%d %H:%M:%S.%f")
    cv2.putText(
        annotated,
        timestamp_text,
        (12, evidence_height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        max(1, line_thickness - 1),
        cv2.LINE_AA,
    )

    return annotated


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    ensure_output_dir(config.output_dir)

    validation_model = YOLO(config.model_path)
    target_class_ids = resolve_target_class_ids(validation_model, config)
    del validation_model

    states = {stream.key: CameraState(stream=stream) for stream in config.streams}
    ready_queue: "queue.Queue[str]" = queue.Queue()
    stop_event = threading.Event()
    mode_controller = CaptureModeController(
        pull_interval_seconds=config.pull_interval_seconds,
        live_timeout_seconds=config.live_timeout_seconds,
    )
    control_server = ControlServer(
        config.control_host,
        config.control_port,
        mode_controller,
        served_dir=Path.cwd(),
    )

    worker_count = min(config.inference_workers, len(states))
    stream_count = max(1, len(states))
    pull_interval_seconds = config.pull_interval_seconds

    capture_workers = []
    for index, state in enumerate(states.values()):
        initial_delay_seconds = (pull_interval_seconds / stream_count) * index
        capture_workers.append(
            RtspCaptureWorker(
                state=state,
                config=config,
                ready_queue=ready_queue,
                stop_event=stop_event,
                mode_controller=mode_controller,
                initial_pull_delay_seconds=initial_delay_seconds,
            )
        )

    inference_workers = [
        InferenceWorker(index + 1, config, states, ready_queue, stop_event, target_class_ids)
        for index in range(worker_count)
    ]

    print(
        f"[INFO] Starting {len(capture_workers)} capture workers and {worker_count} inference workers. "
        f"Default mode is pull every {config.pull_interval_seconds}s. "
        f"Control endpoint: http://{config.control_host}:{config.control_port}/mode"
    )
    print(
        f"[INFO] File browser endpoint: http://{config.control_host}:{config.control_port}/files "
        f"(served root: '{Path.cwd()}')."
    )
    print(
        f"[INFO] Live mode times out after {config.live_timeout_seconds}s without another live command. "
        f"Detection resize: max_width={config.detection_max_width}, max_height={config.detection_max_height}. "
        f"Evidence dir: '{config.output_dir}'."
    )

    control_server.start()

    for worker in inference_workers:
        worker.start()

    for worker in capture_workers:
        worker.start()

    last_mode = None

    try:
        while not stop_event.wait(1.0):
            status = mode_controller.get_status()
            if status["effective_mode"] != last_mode:
                if status["effective_mode"] == "live":
                    print(
                        f"[MODE] Switched to live mode until {status['live_until']}."
                    )
                else:
                    print(
                        f"[MODE] Switched to pull mode. Pull interval is "
                        f"{status['pull_interval_seconds']}s."
                    )
                last_mode = status["effective_mode"]
    except KeyboardInterrupt:
        print("[INFO] Stopping capture.")
        stop_event.set()
    finally:
        stop_event.set()
        control_server.stop()
        for worker in capture_workers:
            worker.join(timeout=2)
        for worker in inference_workers:
            worker.join(timeout=2)


if __name__ == "__main__":
    main()
