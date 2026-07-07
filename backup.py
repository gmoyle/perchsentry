import subprocess
import logging
import threading
import time
from pathlib import Path

CAPTURES_DIR = Path(__file__).parent / "captures"
SLOWMO_DIR = Path(__file__).parent / "slowmo"
TIMELAPSE_DIR = Path(__file__).parent / "timelapse"

log = logging.getLogger("perchsentry")


def run_backup(destination):
    """rsync captures, slowmo, and timelapse to destination (user@host:/path or /local/path)."""
    if not destination.strip():
        return False, "No backup destination configured"
    # No trailing slashes on sources: rsync then creates captures/, slowmo/,
    # and timelapse/ subdirectories at the destination. (With trailing slashes
    # it merges the *contents* of all three flat into one folder.)
    cmd = [
        "rsync", "-av", "--ignore-existing",
        str(CAPTURES_DIR),
        str(SLOWMO_DIR),
        str(TIMELAPSE_DIR),
        destination.rstrip("/") + "/",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            log.info(f"Backup completed to {destination}")
            return True, "Backup completed"
        else:
            log.warning(f"Backup failed: {result.stderr.strip()}")
            return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Backup timed out after 5 minutes"
    except FileNotFoundError:
        return False, "rsync not found — install with: sudo apt install rsync"
    except Exception as e:
        return False, str(e)


class BackupScheduler:
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
            interval_h = s.get("backup_interval", 0)
            dest = s.get("backup_path", "").strip()

            if interval_h > 0 and dest:
                run_backup(dest)
                self._stop.wait(interval_h * 3600)
            else:
                # Check again in 5 minutes in case settings change
                self._stop.wait(300)
