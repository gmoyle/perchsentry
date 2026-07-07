import time
import logging
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from daynight import is_daytime

TIMELAPSE_DIR = Path(__file__).parent / "timelapse"
TIMELAPSE_DIR.mkdir(exist_ok=True)

log = logging.getLogger("perchsentry")


class TimelapseCapturer:
    def __init__(self, camera, get_settings):
        self.camera = camera
        self.get_settings = get_settings
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        last_capture = 0
        while not self._stop.is_set():
            s = self.get_settings()
            interval_mins = s.get("timelapse_interval", 0)
            lat, lon = s.get("latitude"), s.get("longitude")
            is_day = is_daytime(lat, lon) if (lat is not None and lon is not None) else True
            if interval_mins > 0 and is_day:
                now = time.time()
                if now - last_capture >= interval_mins * 60:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = TIMELAPSE_DIR / f"tl_{ts}.jpg"
                    self.camera.capture_file(path)
                    log.debug(f"Timelapse frame: {path.name}")
                    last_capture = now
            time.sleep(10)


def build_video(fps=10):
    frames = sorted(TIMELAPSE_DIR.glob("tl_*.jpg"))
    if not frames:
        return None, "No timelapse frames found"

    output = TIMELAPSE_DIR / "timelapse.mp4"
    # Write frame list for ffmpeg
    list_file = TIMELAPSE_DIR / "frames.txt"
    list_file.write_text("\n".join(f"file '{f}'" for f in frames))

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-vf", f"fps={fps},scale=1920:1080:force_original_aspect_ratio=decrease",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    list_file.unlink(missing_ok=True)

    if result.returncode != 0:
        return None, result.stderr[-500:]

    return output, f"Built from {len(frames)} frames"
