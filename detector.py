import time
import base64
import logging
import threading
import subprocess
import urllib.request
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image

from classify import load_interpreter, load_labels, classify_image
import speciesclassify
from slowmo import SlowMoCapture
from daynight import is_daytime
from objdetect import analyze_frame

CAPTURES_DIR = Path(__file__).parent / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)

# Camera AE/AWB is still settling for a moment after (re)start, which can
# make the very first frames unreliable. Don't trigger captures during this
# window.
STARTUP_GRACE_SECS = 3.0

# Auto-recovery for a wedged Hailo NPU (drops off the PCIe bus). Installed via
# sudoers so the service user may run just this one script as root. Rate-limited
# so a permanently-dead card doesn't reset the bus in a tight loop.
HAILO_RECOVER_CMD = ["sudo", "-n", "/usr/local/sbin/hailo_recover.sh"]
HAILO_RECOVER_COOLDOWN_SECS = 300
# Auto-recovery is DISABLED: the recovery script does a PCIe remove +
# secondary-bus-reset + rescan, but running it while this process still holds
# /dev/hailo0 open drives the hailo_pci driver's release path
# (fops_release → nnc_driver_down → write_firmware_driver_shutdown) into a
# kernel oops against the just-reset device. After that every ioctl returns
# ENODEV and the backend re-init deadlocks the gunicorn worker, taking the whole
# site down. Until the card can be reset with the service stopped (no open fd),
# a dead NPU must simply fail closed and keep the web UI + camera serving.
HAILO_AUTO_RECOVER = False

log = logging.getLogger("perchsentry")


def notify(species, confidence, settings, message=None):
    url = settings.get("ntfy_url", "").strip()
    if not url:
        return
    try:
        # `message` overrides the default sighting text — used for system
        # alerts (e.g. NPU down) that aren't a species sighting.
        if message is not None:
            body = message
            tags = "warning"
        else:
            body = f"{species} spotted! ({confidence:.1%} confidence)"
            tags = "bird"
        headers = {
            "Title": "PerchSentry",
            "Tags": tags,
            "Actions": "view, View capture, http://perchsentry.local:8080/, clear=true",
        }
        user = settings.get("ntfy_user", "").strip()
        passwd = settings.get("ntfy_pass", "").strip()
        if user and passwd:
            token = base64.b64encode(f"{user}:{passwd}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        req = urllib.request.Request(
            url,
            data=body.encode(),
            headers=headers,
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        log.warning(f"ntfy notification failed: {e}")


def images_are_similar(path_a, path_b, threshold=0.98):
    """Return True if two images are nearly identical (dedup check)."""
    try:
        a = np.array(Image.open(path_a).resize((64, 36))).astype(np.float32)
        b = np.array(Image.open(path_b).resize((64, 36))).astype(np.float32)
        similarity = 1 - np.mean(np.abs(a - b)) / 255
        return similarity >= threshold
    except Exception:
        return False


def crop_to_box(path, box, margin=0.2, min_px=48):
    """Save a padded crop of a normalized (x1,y1,x2,y2) box next to `path`.

    Returns the crop path, or None if the box is too small or cropping fails —
    callers then classify the full frame as before.
    """
    try:
        with Image.open(path) as img:
            w, h = img.size
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            x1 = max(0.0, x1 - bw * margin)
            y1 = max(0.0, y1 - bh * margin)
            x2 = min(1.0, x2 + bw * margin)
            y2 = min(1.0, y2 + bh * margin)
            px = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
            if px[2] - px[0] < min_px or px[3] - px[1] < min_px:
                return None
            out = path.with_name(path.stem + "_crop.jpg")
            img.crop(px).save(out, "JPEG", quality=90)
            return out
    except Exception:
        return None


class MotionDetector:
    """NPU-driven presence detector.

    Runs the Hailo NPU continuously against live camera frames (not against a
    saved file — capture_rgb_array() hands back an in-memory frame straight
    from the sensor). "A bird/animal is in frame" IS the trigger: there is no
    pixel-diff motion stage, no per-location threshold tuning. Cars, wind,
    shadows, a swaying chain — none of it matters, because none of it is an
    animal.

    Real-time captures liberally: any frame the NPU calls a bird is saved,
    without gating on SpeciesNet's confidence (that number is unreliable — a
    false positive has been observed to score HIGHER than a real bird). The
    nightly capture verifier (verify_captures.py) culls noise afterwards using
    visit-burst membership + the NPU's own detection score, which are both far
    more trustworthy signals than the species classifier's confidence.
    """

    def __init__(self, camera, get_settings, clients_active=None):
        self.camera = camera
        self.get_settings = get_settings
        # Returns True when someone is viewing the site; slow-mo is suppressed
        # then to avoid colliding with live-stream/gallery load.
        self._clients_active = clients_active or (lambda: False)
        self._thread = None
        self._stop_event = threading.Event()
        self._interp = load_interpreter()
        self._labels = load_labels()
        # SpeciesNet names non-bird animals on the CPU. It's a 214 MB ONNX, so
        # degrade gracefully if it's missing — animals just log generically then.
        try:
            self._species_session = speciesclassify.load_session()
            self._species_labels = speciesclassify.load_labels()
            log.info("SpeciesNet wildlife classifier loaded")
        except Exception as e:
            self._species_session = None
            self._species_labels = None
            log.warning(f"SpeciesNet unavailable ({e}); animals logged generically")
        self._slowmo = SlowMoCapture(camera, get_settings)
        self._last_saved_path = None
        self._started_at = 0
        self._is_day = True
        self._next_day_check = 0
        self._last_npu_recover = 0
        self._status_lock = threading.Lock()
        self._status = {
            "last_event": "idle",
            "detect_score": 0.0,
            "detect_label": None,
            "last_species": None,
            "last_confidence": None,
            "last_filename": None,
            "last_event_at": 0,
            "updated_at": 0,
        }

    def _set_status(self, **kwargs):
        with self._status_lock:
            self._status.update(kwargs)
            self._status["updated_at"] = time.time()

    def get_status(self):
        with self._status_lock:
            st = dict(self._status)
        # Surface NPU health so the UI/HA can show when the Hailo card has
        # dropped out (in which case detection is paused, fail-closed).
        try:
            from objdetect import get_health
            st["npu"] = get_health()
        except Exception:
            st["npu"] = None
        return st

    def _name_species(self, path, box, min_conf):
        """Run SpeciesNet on the animal crop the NPU found and return
        (common_name, confidence), or None to keep the generic 'animal' label
        (classifier absent, error, low confidence, or a non-species output like
        'blank'/'human')."""
        if self._species_session is None:
            return None
        crop = crop_to_box(path, box) if box else None
        src = crop or path
        try:
            r = speciesclassify.classify_image(src, self._species_session, self._species_labels)
        except Exception as e:
            log.warning(f"Species classify failed: {e}")
            return None
        finally:
            if crop:
                crop.unlink(missing_ok=True)
        if r["confidence"] < min_conf or r["species"] in speciesclassify.NON_SPECIES:
            log.debug(f"SpeciesNet inconclusive: {r['species']} ({r['confidence']:.1%})")
            return None
        return r["species"], r["confidence"]

    def start(self):
        self._stop_event.clear()
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _npu_dead(self):
        try:
            from objdetect import get_health
            return not get_health()["healthy"]
        except Exception:
            return False

    def _try_npu_recover(self):
        """Attempt automatic PCIe recovery of a wedged Hailo card, rate-limited.

        Runs the installed recovery script (remove/rescan + secondary bus reset).
        On success, reset objdetect's health/backend so the next frame re-inits
        and inference resumes. Returns True if the device responded afterwards.
        """
        now = time.time()
        if now - self._last_npu_recover < HAILO_RECOVER_COOLDOWN_SECS:
            return False
        self._last_npu_recover = now
        log.warning("Attempting automatic Hailo PCIe recovery…")
        try:
            proc = subprocess.run(HAILO_RECOVER_CMD, capture_output=True,
                                  text=True, timeout=30)
        except Exception as e:
            log.error(f"Hailo recovery script failed to run: {e}")
            return False
        if proc.returncode == 0:
            log.info("Hailo PCIe recovery succeeded; re-initializing backend")
            try:
                import objdetect
                # Force a clean re-init on the next analyze_frame() and clear the
                # dead flag so the pipeline stops failing closed.
                objdetect._backend = None
                objdetect._npu_dead = False
                objdetect._consecutive_failures = 0
            except Exception:
                pass
            return True
        log.error(f"Hailo PCIe recovery failed (rc={proc.returncode}): "
                  f"{proc.stderr.strip() or proc.stdout.strip()}")
        return False

    def _check_npu_health(self, was_dead):
        """Detect NPU down/up transitions: alert on each, and on going down try
        automatic PCIe recovery. Returns the current dead state to carry over."""
        dead = self._npu_dead()
        if dead and not was_dead and HAILO_AUTO_RECOVER:
            recovered = self._try_npu_recover()
            if recovered:
                dead = self._npu_dead()  # may already be healthy again
        if dead != was_dead:
            if dead:
                msg = ("Hailo NPU is DOWN — detection paused (failing closed). "
                       "Card reset needs the service stopped; a reboot will "
                       "clear it.")
                log.warning(msg)
            else:
                msg = "Hailo NPU recovered — detection resumed."
                log.info(msg)
            threading.Thread(
                target=notify, args=("PerchSentry NPU", None, self.get_settings()),
                kwargs={"message": msg}, daemon=True,
            ).start()
        return dead

    def _loop(self):
        last_capture = 0
        npu_was_dead = False

        while not self._stop_event.is_set():
            s = self.get_settings()
            poll_interval = max(0.05, float(s.get("detect_poll_interval", 0.2)))
            time.sleep(poll_interval)
            try:
                # Watch NPU health: alert on the down/up transitions (detection
                # fails closed while it's down) and attempt automatic PCIe
                # recovery when it dies, so a dropped-out Hailo card self-heals
                # without a manual reset.
                npu_was_dead = self._check_npu_health(npu_was_dead)

                if time.time() - self._started_at < STARTUP_GRACE_SECS:
                    self._set_status(last_event="warming_up")
                    continue

                # No captures at night — this camera has no IR/low-light
                # capability, so night frames are useless. Re-check once a
                # minute; wait in short chunks so shutdown stays responsive.
                if s.get("latitude") is not None and s.get("longitude") is not None:
                    now = time.time()
                    if now >= self._next_day_check:
                        self._is_day = is_daytime(s["latitude"], s["longitude"])
                        self._next_day_check = now + 60
                    if not self._is_day:
                        self._set_status(last_event="night_pause")
                        self._stop_event.wait(30)
                        continue

                if self._slowmo.is_active():
                    # Camera is mid-reconfigure for a slow-mo burst (different
                    # resolution/mode entirely) — skip this tick rather than
                    # fight it for cam_lock or capture a meaningless frame.
                    self._set_status(last_event="slowmo_capturing")
                    continue

                last_capture = self._npu_tick(s, last_capture)
            except Exception as e:
                # Never let an unexpected error kill this thread permanently —
                # log it and keep going next tick. Unlike the old pixel-diff
                # design there's no cross-tick baseline to reset; each poll is
                # independent.
                log.error(f"Detector loop error: {e}", exc_info=True)

    def _npu_tick(self, s, last_capture):
        arr = self.camera.capture_rgb_array()
        if arr is None:
            self._set_status(last_event="no_frame")
            return last_capture

        min_conf = s.get("detect_confidence", 0.15)
        img = Image.fromarray(arr, mode="RGB")
        det = analyze_frame(img, min_conf)

        self._set_status(
            last_event=("presence" if det["has_animal"] else "watching"),
            detect_score=det.get("score", 0.0),
            detect_label=det.get("label"),
        )

        if not det["has_animal"]:
            return last_capture

        now = time.time()
        if now - last_capture < s.get("motion_cooldown", 3):
            return last_capture  # something's present, but too soon to re-capture
        last_capture = now

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = CAPTURES_DIR / f"motion_{ts}.jpg"
        self.camera.capture_file(path)  # fresh full-res still for the record

        self._process_capture(path, now)
        return last_capture

    def _process_capture(self, path, now):
        s = self.get_settings()

        # Deduplication — skip if nearly identical to last save
        if self._last_saved_path and images_are_similar(path, self._last_saved_path):
            path.unlink()
            log.debug("Duplicate frame skipped")
            self._set_status(last_event="duplicate", last_event_at=now)
            return
        self._last_saved_path = path

        # Re-check the NPU on the saved full-res file: the poll frame that
        # triggered this capture is a beat older than what capture_file() just
        # grabbed (different framing is possible), so get a fresh box from the
        # actual saved image rather than reusing the trigger's.
        det = analyze_frame(path, s.get("detect_confidence", 0.15))
        if not det["has_animal"]:
            log.debug(f"No animal on saved frame, deleting {path.name}")
            path.unlink(missing_ok=True)
            self._last_saved_path = None
            self._set_status(last_event="no_animal", last_event_at=now)
            return

        # Classify a bird-centered crop when the NPU gave us a box: the bird
        # fills the classifier input instead of ~5% of the frame, which
        # dramatically improves species confidence.
        classify_path = path
        crop_tmp = None
        if det.get("box"):
            crop_tmp = crop_to_box(path, det["box"])
            if crop_tmp:
                classify_path = crop_tmp
        try:
            result = classify_image(classify_path, self._interp, self._labels)
        finally:
            if crop_tmp:
                crop_tmp.unlink(missing_ok=True)

        if det.get("label") == "bird":
            # Capture liberally: trust the NPU presence signal and keep the
            # frame. SpeciesNet only gives a tentative label here — it can
            # hallucinate species/confidence on blurry frames — so real-time
            # does NOT gate on its confidence. The nightly capture verifier
            # decides real-vs-noise from visit-burst membership + the NPU
            # detection score (logged below), both far more trustworthy.
            species = result["species"] if result["is_bird"] else "unidentified bird"
            confidence = result["confidence"] if result["is_bird"] else 0.0
            log.info(f"BIRD DETECTED: {species} ({confidence:.1%}) → {path.name} "
                     f"[npu {det.get('score', 0.0):.2f}]")
            self._set_status(
                last_event="bird_detected", last_event_at=now,
                last_species=species, last_confidence=confidence,
                last_filename=path.name,
            )

            if s.get("slowmo_birds", False) and not self._slowmo.is_active():
                if self._clients_active():
                    log.info("Bird detected — skipping slow-mo (someone is viewing the site)")
                else:
                    log.info("Bird detected! Triggering slow-mo capture")
                    self._slowmo.capture(species, confidence)
            threading.Thread(
                target=notify,
                args=(species, confidence, s),
                daemon=True,
            ).start()
        else:
            # Not a bird — but the NPU may have labelled it a known non-bird
            # animal (cat/dog/bear/…). Keep those as a separate "Animals"
            # track instead of discarding, so daytime wildlife isn't missed.
            # Bird stats stay pure because this logs a distinct ANIMAL
            # DETECTED line.
            animal = det.get("label")
            if animal and s.get("capture_animals", True):
                score = det.get("score", 0.0)
                # Name the species with SpeciesNet on the same crop; fall back
                # to the generic "animal" label if it's unsure. This
                # confidence is the classifier's, a different (and more
                # meaningful) number than the detector's box score it replaces.
                named = self._name_species(
                    path, det.get("box"),
                    s.get("species_confidence", 50) / 100.0,
                )
                if named:
                    animal, score = named
                log.info(f"ANIMAL DETECTED: {animal} ({score:.1%}) → {path.name}")
                self._last_saved_path = path
                self._set_status(
                    last_event="animal_detected", last_event_at=now,
                    last_species=animal, last_confidence=score,
                    last_filename=path.name,
                )
                if not self._slowmo.is_active() and s.get("slowmo_animals", True):
                    if self._clients_active():
                        log.info(f"{animal} detected — skipping slow-mo (someone is viewing the site)")
                    else:
                        log.info(f"{animal} detected! Triggering slow-mo capture")
                        self._slowmo.capture(animal, score)
                if s.get("notify_animals", False):
                    threading.Thread(
                        target=notify, args=(animal, score, s), daemon=True,
                    ).start()
            else:
                log.debug(f"Capture has no known bird/animal, deleting {path.name}")
                path.unlink(missing_ok=True)
                self._last_saved_path = None
                self._set_status(
                    last_event="no_bird", last_event_at=now,
                    last_species=None, last_confidence=result.get("confidence"),
                )
