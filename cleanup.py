import time
import shutil
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta

CAPTURES_DIR = Path(__file__).parent / "captures"
TIMELAPSE_DIR = Path(__file__).parent / "timelapse"
SLOWMO_DIR = Path(__file__).parent / "slowmo"

log = logging.getLogger("perchsentry")


def disk_usage():
    total, used, free = shutil.disk_usage("/")
    captures_size = sum(f.stat().st_size for f in CAPTURES_DIR.glob("*.jpg")) if CAPTURES_DIR.exists() else 0
    return {
        "total_gb": round(total / 2**30, 1),
        "used_gb": round(used / 2**30, 1),
        "free_gb": round(free / 2**30, 1),
        "used_pct": round(used / total * 100, 1),
        "captures_mb": round(captures_size / 2**20, 1),
    }


def run_cleanup(retention_days):
    if retention_days <= 0:
        return 0
    cutoff = datetime.now() - timedelta(days=retention_days)
    deleted = 0
    for path in CAPTURES_DIR.glob("*.jpg"):
        try:
            dt = datetime.strptime(path.stem, "motion_%Y%m%d_%H%M%S")
            if dt < cutoff:
                path.unlink()
                deleted += 1
        except ValueError:
            pass
    if deleted:
        log.info(f"Cleanup: deleted {deleted} captures older than {retention_days} days")
    return deleted


class DiskCleaner:
    def __init__(self, get_settings):
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
        while not self._stop.is_set():
            s = self.get_settings()
            retention = s.get("retention_days", 30)
            run_cleanup(retention)
            # Check once per hour
            self._stop.wait(3600)
