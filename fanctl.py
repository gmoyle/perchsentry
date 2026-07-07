#!/usr/bin/python3
"""Privileged fan-level helper for PerchSentry (invoked via sudo).

Sets the Pi 5 active-cooler speed through the pwm-fan cooling device's
`cur_state` (0=off … 4=max). We drive the cooling device rather than the raw
PWM because the kernel thermal governor still reasserts control on the next
trip-point crossing, so this only holds *between* crossings — exactly enough to
silence the fan for a short recording without defeating thermal protection.

`auto` restores the level the current temperature calls for (matching the
firmware fan curve). This is necessary because the thermal zone is trip-driven
with no polling: after a manual change it will not self-correct within a
temperature band, so we must hand it back a sensible level explicitly.

Firmware CPU throttling (~80°C soft, ~85°C hard, ~110°C shutdown) is entirely
independent of this, so the SoC is protected no matter what level we set.

Deployed copy lives at /usr/local/sbin/perchsentry-fanctl (root-owned); this
repo copy is the source of truth — reinstall it there after editing.
"""
import sys
import glob

# temp °C ≥ threshold → fan level, checked high-to-low. Mirrors the kernel
# trip points (50 / 60 / 67.5 / 75 °C) seen on this Pi's active cooler.
TRIPS = [(75.0, 4), (67.5, 3), (60.0, 2), (50.0, 1)]


def find_cooling():
    for p in glob.glob("/sys/class/thermal/cooling_device*"):
        try:
            with open(p + "/type") as f:
                if f.read().strip() == "pwm-fan":
                    return p
        except OSError:
            continue
    return None


def zone_temp():
    with open("/sys/class/thermal/thermal_zone0/temp") as f:
        return int(f.read().strip()) / 1000.0


def auto_level():
    t = zone_temp()
    for temp, lvl in TRIPS:
        if t >= temp:
            return lvl
    return 0


def main(argv):
    if len(argv) != 2 or argv[1] not in ("off", "quiet", "auto"):
        sys.stderr.write("usage: perchsentry-fanctl off|quiet|auto\n")
        return 2
    cd = find_cooling()
    if not cd:
        sys.stderr.write("pwm-fan cooling device not found\n")
        return 1
    level = {"off": 0, "quiet": 1}.get(argv[1])
    if level is None:                 # "auto"
        level = auto_level()
    try:
        with open(cd + "/cur_state", "w") as f:
            f.write(str(level))
    except OSError as e:
        sys.stderr.write(f"failed to set fan level: {e}\n")
        return 1
    print(level)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
