import json
from pathlib import Path

SETTINGS_FILE = Path(__file__).parent / "settings.json"

DEFAULTS = {
    "brightness": 0.0,
    "contrast": 1.0,
    "saturation": 1.0,
    "sharpness": 1.0,
    "motion_threshold": 25,
    "motion_min_pixels": 500,
    "motion_cooldown": 3,
    "ntfy_url": "http://jetson.local:8088/perchsentry",
    "ntfy_user": "",
    "ntfy_pass": "",
    "timelapse_interval": 0,       # minutes between frames, 0 = disabled
    "focus_mode": "auto",          # "auto" or "manual"
    "focus_position": 1.0,         # diopters (1/m): 0=inf, 1=1m, 2=0.5m, 4=0.25m
    "confidence_threshold": 30,    # percent (0-100)
    "retention_days": 30,          # 0 = keep forever
    "latitude": None,              # for day/night mode
    "longitude": None,
    "backup_path": "",             # rsync destination, e.g. user@nas:/backups/birds
    "backup_interval": 0,          # hours between backups, 0 = disabled
    "recording_fan_mode": "normal",  # fan during slow-mo: normal | quiet | silent
    # Non-bird animals (cat/dog/bear/… — whatever the NPU object detector knows)
    # are kept as a separate "Animals" track so they don't dilute bird stats.
    "capture_animals": True,       # keep captures of detected non-bird animals
    "slowmo_animals": True,        # also record a slow-mo clip for them
    "notify_animals": False,       # send an ntfy alert for them (off = quieter)
}


def load():
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text())
            return {**DEFAULTS, **data}
        except Exception:
            pass
    return dict(DEFAULTS)


def save(settings):
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2))
