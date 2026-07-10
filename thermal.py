import time
import logging
import threading

from sysmon import read_temp

log = logging.getLogger("perchsentry")

# How often to sample the SoC temperature.
POLL_SECS = 15
# Require this many consecutive over-threshold reads before entering siesta, so
# a single transient spike (a passing inference burst) doesn't nap the unit.
ENTER_STREAK = 2


class ThermalGovernor:
    """Watches the SoC temperature and drives a "thermal siesta": when the
    enclosure overheats in midday sun the CPU throttles and the Hailo NPU wedges
    under load, so we pause capture/detection and just serve a banner until it
    cools. Enter at settings['thermal_siesta_c'], resume below
    settings['thermal_resume_c'] — the gap gives hysteresis so it doesn't flap.

    The governor owns no workers directly; it flips a shared state dict (read via
    is_siesta()) that the detector/timelapse loops gate on and the web UI polls.
    """

    def __init__(self, get_settings, on_change=None):
        self.get_settings = get_settings
        self._on_change = on_change or (lambda active, temp_c: None)
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state = {
            "active": False,
            "since": 0.0,
            "temp_c": None,
            "entered_c": None,   # temp that tripped the nap, for the banner
        }
        self._hot_streak = 0

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def is_siesta(self):
        with self._lock:
            return self._state["active"]

    def status(self):
        s = self.get_settings()
        with self._lock:
            st = dict(self._state)
        st["enabled"] = bool(s.get("thermal_siesta", True))
        st["siesta_c"] = s.get("thermal_siesta_c", 80)
        st["resume_c"] = s.get("thermal_resume_c", 68)
        return st

    def _loop(self):
        while not self._stop.is_set():
            self._stop.wait(POLL_SECS)
            if self._stop.is_set():
                break
            try:
                self._tick()
            except Exception as e:
                log.error(f"Thermal governor error: {e}", exc_info=True)

    def _tick(self):
        s = self.get_settings()
        enabled = bool(s.get("thermal_siesta", True))
        siesta_c = float(s.get("thermal_siesta_c", 80))
        resume_c = float(s.get("thermal_resume_c", 68))
        temp_c = read_temp().get("temp_c")

        with self._lock:
            active = self._state["active"]
            self._state["temp_c"] = temp_c

        # Disabled at runtime — clear any nap so work resumes.
        if not enabled:
            self._hot_streak = 0
            if active:
                self._set_active(False, temp_c)
            return

        if temp_c is None:
            return  # can't read temp — don't change state on a blind read

        if not active:
            if temp_c >= siesta_c:
                self._hot_streak += 1
                if self._hot_streak >= ENTER_STREAK:
                    self._set_active(True, temp_c)
            else:
                self._hot_streak = 0
        else:
            # Hysteresis: stay napping until it drops back below resume_c.
            if temp_c <= resume_c:
                self._set_active(False, temp_c)

    def _set_active(self, active, temp_c):
        with self._lock:
            self._state["active"] = active
            self._state["since"] = time.time() if active else 0.0
            self._state["entered_c"] = temp_c if active else None
        self._hot_streak = 0
        if active:
            log.warning(
                f"Thermal siesta ENGAGED at {temp_c}°C — capture/detection "
                f"paused until the unit cools."
            )
        else:
            log.info(f"Thermal siesta lifted at {temp_c}°C — resuming.")
        try:
            self._on_change(active, temp_c)
        except Exception as e:
            log.warning(f"Thermal siesta on_change hook failed: {e}")
