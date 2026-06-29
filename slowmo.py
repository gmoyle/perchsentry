import time
import logging
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

SLOWMO_DIR = Path(__file__).parent / "slowmo"
SLOWMO_DIR.mkdir(exist_ok=True)

HUMMINGBIRD_SPECIES = {
    "Archilochus colubris",   # Ruby-throated
    "Archilochus alexandri",  # Black-chinned
    "Calypte anna",           # Anna's
    "Calypte costae",         # Costa's
    "Selasphorus rufus",      # Rufous
    "Selasphorus calliope",   # Calliope
    "Selasphorus sasin",      # Allen's
    "Amazilia yucatanensis",  # Buff-bellied
    "Eugenes fulgens",        # Rivoli's
    "Lampornis clemenciae",   # Blue-throated
}

CAPTURE_FPS = 120       # target fps during burst
CAPTURE_SECS = 3        # burst duration
PLAYBACK_FPS = 25       # output fps → 4.8× slowdown

# IMX708 2x2 binned mode: 616×462 @ 120fps
SLOWMO_SIZE = (616, 462)
FRAME_DURATION_US = 1_000_000 // CAPTURE_FPS   # 8333µs per frame

log = logging.getLogger("birdbuddy")


def is_hummingbird(species):
    return species in HUMMINGBIRD_SPECIES


def _h264_to_mp4(h264_path, mp4_path):
    """Wrap raw H264 in MP4 and encode to playback speed."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(CAPTURE_FPS),
        "-i", str(h264_path),
        "-vf", f"fps={PLAYBACK_FPS},scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast",
        str(mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.warning(f"Slow-mo encode failed: {result.stderr[-400:]}")
        return False
    return True


class SlowMoCapture:
    def __init__(self, camera):
        self.camera = camera
        self._lock = threading.Lock()
        self._active = False

    def is_active(self):
        return self._active

    def capture(self, species):
        if self._active:
            return
        threading.Thread(target=self._run, args=(species,), daemon=True).start()

    def _run(self, species):
        with self._lock:
            self._active = True
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            h264_path = SLOWMO_DIR / f"raw_{ts}.h264"
            output_path = SLOWMO_DIR / f"slowmo_{ts}.mp4"

            log.info(f"Slow-mo burst started for {species} ({CAPTURE_SECS}s @ {CAPTURE_FPS}fps)")

            cam = self.camera.cam

            try:
                # 1. Pause the normal encode loop
                self.camera._stop.set()
                cam.stop()

                # 2. Reconfigure for high-framerate binned mode
                hfr_config = cam.create_video_configuration(
                    main={"size": SLOWMO_SIZE, "format": "YUV420"},
                    controls={"FrameDurationLimits": (FRAME_DURATION_US, FRAME_DURATION_US)},
                )
                cam.configure(hfr_config)
                cam.start()

                # 3. Record H264 natively for CAPTURE_SECS seconds
                encoder = H264Encoder()
                with open(h264_path, "wb") as f:
                    cam.start_recording(encoder, FileOutput(f))
                    time.sleep(CAPTURE_SECS)
                    cam.stop_recording()

                log.info(f"Burst recorded ({h264_path.stat().st_size // 1024} KB), encoding to MP4…")

                # 4. Convert to MP4 at playback speed
                if _h264_to_mp4(h264_path, output_path):
                    log.info(
                        f"Slow-mo saved: {output_path.name} "
                        f"({CAPTURE_FPS/PLAYBACK_FPS:.1f}x slowdown)"
                    )
                h264_path.unlink(missing_ok=True)

            except Exception as e:
                log.error(f"Slow-mo capture failed: {e}", exc_info=True)
                h264_path.unlink(missing_ok=True)

            finally:
                # 5. Restore normal config and restart encode loop
                try:
                    cam.stop()
                    self.camera._configure()
                    cam.start()
                    self.camera._stop.clear()
                    import threading as _t
                    self.camera._capture_thread = _t.Thread(
                        target=self.camera._encode_loop, daemon=True
                    )
                    self.camera._capture_thread.start()
                    log.info("Camera restored to normal mode after slow-mo")
                except Exception as e2:
                    log.error(f"Failed to restore camera after slow-mo: {e2}", exc_info=True)

                self._active = False
