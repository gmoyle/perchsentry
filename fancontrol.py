import logging
import subprocess
import threading

from sysmon import read_temp

log = logging.getLogger("birdbuddy")

# Root helper (see fanctl.py) reachable via a narrow NOPASSWD sudoers rule.
FANCTL = "/usr/local/sbin/birdbuddy-fanctl"

# If the SoC reaches this while the fan is suppressed for a recording, restore
# cooling immediately — mid-recording noise beats thermal throttling. Sits
# below the firmware's ~80°C soft-throttle point to leave headroom.
CEILING_C = 78
_WATCH_INTERVAL = 3


def _set(cmd):
    """Ask the helper to set a fan mode. Returns True on success; on any
    failure we log and report False so the caller leaves cooling automatic."""
    try:
        subprocess.run(["sudo", "-n", FANCTL, cmd],
                       capture_output=True, text=True, timeout=5, check=True)
        return True
    except Exception as e:
        log.warning(f"Fan '{cmd}' failed (leaving fan in automatic mode): {e}")
        return False


class RecordingFan:
    """Context manager: quiet or silence the fan for a recording, then always
    hand control back to the thermal governor.

    mode "quiet"  → low steady speed: a soft hum instead of the jarring
                    spin-up-to-max the governor would otherwise do, which is
                    far easier to keep out of the audio.
    mode "silent" → fan off. The governor turns it back on at the next trip
                    point on its own; a watchdog also restores it early if the
                    temperature climbs to CEILING_C.
    any other value → no-op (normal governed cooling).

    Firmware CPU throttling is independent of the fan, so the SoC cannot
    overheat regardless of the mode chosen here."""

    def __init__(self, mode):
        self.mode = mode
        self._stop = threading.Event()
        self._suppressed = False

    def __enter__(self):
        if self.mode == "quiet":
            self._suppressed = _set("quiet")
        elif self.mode == "silent":
            self._suppressed = _set("off")
        if self._suppressed:
            threading.Thread(target=self._watchdog, daemon=True).start()
        return self

    def _watchdog(self):
        while not self._stop.wait(_WATCH_INTERVAL):
            t = read_temp().get("temp_c")
            if t is not None and t >= CEILING_C:
                log.warning(f"Recording fan: {t}°C reached ceiling {CEILING_C}°C"
                            f" — restoring cooling mid-recording")
                _set("auto")
                self._suppressed = False
                return

    def __exit__(self, *exc):
        self._stop.set()
        if self._suppressed:
            _set("auto")
        return False
