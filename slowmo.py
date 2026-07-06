import time
import logging
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

from fancontrol import RecordingFan

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
    "Selasphorus platycercus",  # Broad-tailed
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
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(CAPTURE_FPS),
        "-i", str(h264_path),
        "-vf", (
            f"fps={PLAYBACK_FPS},"
            "scale=1280:720:force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2"
        ),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast",
        str(mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.warning(f"Slow-mo encode failed: {result.stderr[-400:]}")
        return False
    return True


# Shared flag so the MJPEG stream can overlay a banner
_capturing = False

def is_capturing():
    return _capturing


class SlowMoCapture:
    def __init__(self, camera, get_settings=None):
        self.camera = camera
        self.get_settings = get_settings
        self._lock = threading.Lock()
        self._active = False

    def is_active(self):
        return self._active

    def capture(self, species, confidence=None):
        if self._active:
            return
        threading.Thread(target=self._run, args=(species, confidence), daemon=True).start()

    def _run(self, species, confidence=None):
        global _capturing
        with self._lock:
            self._active = True
            _capturing = True
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            h264_path = SLOWMO_DIR / f"raw_{ts}.h264"
            output_path = SLOWMO_DIR / f"slowmo_{ts}.mp4"
            meta_path = SLOWMO_DIR / f"slowmo_{ts}.json"

            log.info(f"Slow-mo burst started for {species} ({CAPTURE_SECS}s @ {CAPTURE_FPS}fps)")

            cam = self.camera.cam

            # Hold the camera lock for the WHOLE reconfigure+record+restore
            # sequence. Without this, a settings POST landing mid-sequence
            # (e.g. apply_settings calling set_controls while we're between
            # cam.stop()/cam.configure()/cam.start()) can race with this
            # thread on the same picamera2/libcamera object hard enough to
            # hang the camera driver — and on this hardware that has taken
            # the whole Pi down, not just the Python process.
            with self.camera.cam_lock:
                try:
                    # Pause encode loop (last frame stays frozen in _frames)
                    self.camera.pause_for_slowmo()

                    # Reconfigure for high-framerate binned mode
                    cam.stop()
                    hfr_config = cam.create_video_configuration(
                        main={"size": SLOWMO_SIZE, "format": "YUV420"},
                        controls={"FrameDurationLimits": (FRAME_DURATION_US, FRAME_DURATION_US)},
                    )
                    cam.configure(hfr_config)
                    cam.start()

                    # Record H264 natively for CAPTURE_SECS seconds. Optionally
                    # quiet/silence the cooling fan for just this window so it
                    # doesn't whine into an attached microphone; the fan is
                    # always restored to automatic control afterwards.
                    fan_mode = "normal"
                    if self.get_settings:
                        fan_mode = self.get_settings().get("recording_fan_mode", "normal")
                    encoder = H264Encoder()
                    with open(h264_path, "wb") as f, RecordingFan(fan_mode):
                        cam.start_recording(encoder, FileOutput(f))
                        time.sleep(CAPTURE_SECS)
                        cam.stop_recording()

                    log.info(f"Burst recorded ({h264_path.stat().st_size // 1024} KB), encoding to MP4…")

                    if _h264_to_mp4(h264_path, output_path):
                        log.info(
                            f"Slow-mo saved: {output_path.name} "
                            f"({CAPTURE_FPS/PLAYBACK_FPS:.1f}x slowdown)"
                        )
                        try:
                            import json as _json
                            meta_path.write_text(_json.dumps({
                                "trigger_species": species,
                                "trigger_confidence": (round(float(confidence), 4)
                                                       if confidence is not None else None),
                                "trigger_is_hummingbird": is_hummingbird(species),
                                "created_at": datetime.now().isoformat(timespec="seconds"),
                            }))
                        except Exception as _e:
                            log.warning(f"Slow-mo sidecar write failed: {_e}")
                    h264_path.unlink(missing_ok=True)

                except Exception as e:
                    log.error(f"Slow-mo capture failed: {e}", exc_info=True)
                    h264_path.unlink(missing_ok=True)

                finally:
                    # Restore normal config and resume encode loop
                    try:
                        cam.stop()
                        self.camera._configure()
                        cam.start()
                        self.camera.resume_from_slowmo()
                        log.info("Camera restored to normal mode after slow-mo")
                    except Exception as e2:
                        log.error(f"Failed to restore camera after slow-mo: {e2}", exc_info=True)

                    _capturing = False
                    self._active = False
