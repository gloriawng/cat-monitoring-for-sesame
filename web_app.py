"""Local web dashboard for cat-presence inference and Blink clip monitoring."""

import asyncio
import base64
import binascii
import json
import logging
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
BLINK_CONFIG = ROOT / "blink_config.json"
CLIP_ROOT = ROOT / "data" / "blink-clips"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
SYNC_INTERVAL_SECONDS = 60
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cat-monitor")


class CatMonitor:
    def __init__(self):
        self._lock = threading.Lock()
        self._classifier_lock = threading.Lock()
        self._classifier = None
        self._worker = None
        self._stop_event = None
        self._status = {
            "connected": False,
            "running": False,
            "camera": None,
            "last_sync": None,
            "last_error": None,
            "events": [],
        }

    def status(self):
        with self._lock:
            result = dict(self._status)
            result["events"] = list(self._status["events"])
        result["blink_configured"] = BLINK_CONFIG.is_file()
        return result

    def predict_image(self, image_bytes):
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                if image.width * image.height > 20_000_000:
                    raise ValueError("Image dimensions exceed the 20-megapixel limit.")
                image.load()
                image = image.convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Upload a valid image file.") from exc

        return self._predict_pil_image(image)

    def _predict_pil_image(self, image):
        with self._classifier_lock:
            if self._classifier is None:
                from inference import CatPresenceClassifier

                self._classifier = CatPresenceClassifier()
            result = self._classifier.predict_image(image)
        return {
            "class": result["class"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
        }

    async def _connect_blink(self):
        if not BLINK_CONFIG.is_file():
            raise ValueError(
                "Blink is not configured. Run setup.py first to save Blink credentials."
            )

        from aiohttp import ClientSession
        from blinkpy.auth import Auth
        from blinkpy.blinkpy import Blink
        from blinkpy.helpers.util import json_load

        async with ClientSession() as session:
            blink = Blink(session=session)
            blink.auth = Auth(await json_load(str(BLINK_CONFIG)), no_prompt=True)
            if not await blink.start():
                raise RuntimeError("Blink sign-in failed. Re-run setup.py to refresh credentials.")
            return list(blink.cameras.keys())

    def connect_blink(self):
        return asyncio.run(self._connect_blink())

    def start_monitoring(self, camera):
        if not isinstance(camera, str) or not camera.strip():
            raise ValueError("Choose a Blink camera before starting monitoring.")
        if not BLINK_CONFIG.is_file():
            raise ValueError(
                "Blink is not configured. Run setup.py first to save Blink credentials."
            )

        with self._lock:
            if self._worker and self._worker.is_alive():
                raise ValueError("Blink monitoring is already running.")
            self._stop_event = threading.Event()
            self._status.update(
                connected=False,
                running=True,
                camera=camera,
                last_error=None,
            )
            self._worker = threading.Thread(
                target=self._run_monitor,
                args=(camera, self._stop_event),
                name="blink-cat-monitor",
                daemon=True,
            )
            self._worker.start()

    def stop_monitoring(self):
        with self._lock:
            if self._stop_event:
                self._stop_event.set()
            self._status["running"] = False

    def _update_status(self, **updates):
        with self._lock:
            self._status.update(updates)

    def _add_event(self, event):
        with self._lock:
            self._status["events"].insert(0, event)
            del self._status["events"][50:]

    def _run_monitor(self, camera, stop_event):
        try:
            asyncio.run(self._sync_loop(camera, stop_event))
        except Exception as exc:
            logger.exception("Blink monitoring stopped unexpectedly")
            self._update_status(
                connected=False,
                running=False,
                last_error=str(exc),
            )
        finally:
            self._update_status(running=False)

    async def _sync_loop(self, camera, stop_event):
        from aiohttp import ClientSession
        from blinkpy.auth import Auth
        from blinkpy.blinkpy import Blink
        from blinkpy.helpers.util import json_load

        CLIP_ROOT.mkdir(parents=True, exist_ok=True)
        known_clips = {clip.name for clip in CLIP_ROOT.glob("*.mp4")}
        cursor = datetime.now() - timedelta(minutes=10)

        async with ClientSession() as session:
            blink = Blink(session=session)
            blink.auth = Auth(await json_load(str(BLINK_CONFIG)), no_prompt=True)
            if not await blink.start():
                raise RuntimeError(
                    "Blink sign-in failed. Re-run setup.py to refresh credentials."
                )
            if camera not in blink.cameras:
                raise ValueError(f"Blink camera '{camera}' was not found in this account.")

            self._update_status(connected=True, last_error=None)
            while not stop_event.is_set():
                try:
                    await blink.download_videos(
                        path=str(CLIP_ROOT),
                        since=cursor.strftime("%Y/%m/%d %H:%M:%S"),
                        camera=camera,
                        stop=2,
                        delay=1,
                    )
                    for clip_path in sorted(CLIP_ROOT.glob("*.mp4")):
                        if clip_path.name in known_clips:
                            continue
                        event = self._classify_clip(clip_path, camera)
                        known_clips.add(clip_path.name)
                        if event:
                            self._add_event(event)
                    cursor = datetime.now() - timedelta(minutes=1)
                    self._update_status(
                        connected=True,
                        last_sync=datetime.now().astimezone().isoformat(timespec="seconds"),
                        last_error=None,
                    )
                except Exception as exc:
                    logger.exception("Blink clip synchronization failed")
                    self._update_status(connected=True, last_error=str(exc))

                for _ in range(SYNC_INTERVAL_SECONDS):
                    if stop_event.is_set():
                        break
                    await asyncio.sleep(1)

    def _classify_clip(self, clip_path, camera):
        import cv2
        import numpy as np

        capture = cv2.VideoCapture(str(clip_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open downloaded Blink clip '{clip_path.name}'.")

        fps = capture.get(cv2.CAP_PROP_FPS)
        interval = max(1, round(fps * 2)) if fps > 0 else 30
        cat_scores = []
        frame_index = 0
        try:
            while len(cat_scores) < 4:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % interval == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    prediction = self._predict_pil_image(Image.fromarray(rgb_frame))
                    cat_scores.append(prediction["probabilities"]["cat-present"])
                frame_index += 1
        finally:
            capture.release()

        if not cat_scores:
            raise RuntimeError(f"No readable frames were found in '{clip_path.name}'.")

        cat_probability = float(np.mean(cat_scores))
        label = "cat-present" if cat_probability >= 0.5 else "no-cat"
        confidence = cat_probability if label == "cat-present" else 1 - cat_probability
        return {
            "camera": camera,
            "clip": clip_path.name,
            "detected": label == "cat-present",
            "class": label,
            "confidence": confidence,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        }


monitor = CatMonitor()


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "CatMonitor/1.0"

    def do_GET(self):
        if self.path == "/":
            return self._serve_file(WEB_ROOT / "index.html")
        if self.path == "/api/status":
            return self._json_response(200, monitor.status())
        if self.path.startswith("/static/"):
            relative_path = unquote(self.path.removeprefix("/static/"))
            file_path = (WEB_ROOT / relative_path).resolve()
            if WEB_ROOT.resolve() not in file_path.parents:
                return self._json_response(404, {"error": "File not found."})
            return self._serve_file(file_path)
        return self._json_response(404, {"error": "Not found."})

    def do_POST(self):
        try:
            if self.path == "/api/blink/connect":
                cameras = monitor.connect_blink()
                return self._json_response(200, {"cameras": cameras})

            body = self._read_json_body()
            if self.path == "/api/predict":
                return self._predict(body)
            if self.path == "/api/blink/monitor":
                if body.get("enabled"):
                    monitor.start_monitoring(body.get("camera"))
                else:
                    monitor.stop_monitoring()
                return self._json_response(200, monitor.status())
            return self._json_response(404, {"error": "Not found."})
        except ValueError as exc:
            return self._json_response(400, {"error": str(exc)})
        except Exception as exc:
            logger.exception("Request failed")
            return self._json_response(500, {"error": str(exc)})

    def _predict(self, body):
        encoded_image = body.get("image")
        if not isinstance(encoded_image, str):
            raise ValueError("Choose an image to upload.")
        try:
            image_bytes = base64.b64decode(encoded_image, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("The uploaded image data is invalid.") from exc
        if not image_bytes or len(image_bytes) > MAX_UPLOAD_BYTES:
            raise ValueError("Images must be smaller than 12 MB.")
        prediction = monitor.predict_image(image_bytes)
        prediction["filename"] = str(body.get("filename", "upload"))
        return self._json_response(200, prediction)

    def _read_json_body(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid request size.") from exc
        if content_length <= 0 or content_length > MAX_UPLOAD_BYTES * 2:
            raise ValueError("Request is empty or exceeds the upload limit.")
        try:
            body = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object.")
        return body

    def _serve_file(self, file_path):
        if not file_path.is_file():
            return self._json_response(404, {"error": "File not found."})
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
        }.get(file_path.suffix, "application/octet-stream")
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _json_response(self, status, payload):
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format_string, *args):
        logger.info("%s - %s", self.address_string(), format_string % args)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8000), RequestHandler)
    print("Cat Monitor is available at http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Cat Monitor.")
    finally:
        monitor.stop_monitoring()
        server.server_close()


if __name__ == "__main__":
    main()
