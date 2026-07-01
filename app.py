import sys
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, Response, render_template, request, jsonify, send_file, abort

sys.stdout.reconfigure(line_buffering=True)

import settings as cfg
from camera import Camera, generate_stream
from detector import MotionDetector
from timelapse import TimelapseCapturer, build_video, TIMELAPSE_DIR
from slowmo import SLOWMO_DIR, is_capturing
from cleanup import DiskCleaner, disk_usage
from daynight import DayNightManager
from backup import BackupScheduler, run_backup
from verify_slowmo import SlowMoVerifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent / "logs" / "birdbuddy.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("birdbuddy")

app = Flask(__name__)
import threading


@app.before_request
def _track_client_activity():
    global _last_request_at
    from flask import request as _rq
    if _rq.path not in _ACTIVITY_EXCLUDE:
        _last_request_at = _time.time()


# ── Maintenance mode ────────────────────────────────────────────────────────
# While a heavy nightly job runs, lock the UI so no browser traffic competes
# with it. Auto-expires as a safety net so a crash can't leave the site locked.
_maintenance = {"active": False, "since": 0.0, "reason": ""}
MAINT_MAX_SECS = 3 * 3600


def set_maintenance(active, reason=""):
    _maintenance["active"] = bool(active)
    _maintenance["since"] = _time.time() if active else 0.0
    _maintenance["reason"] = reason if active else ""
    log.info("Maintenance mode " + ("ON: " + reason if active else "OFF"))


def _maintenance_active():
    if not _maintenance["active"]:
        return False
    if _time.time() - _maintenance["since"] > MAINT_MAX_SECS:
        _maintenance["active"] = False
        return False
    return True


@app.route("/api/maintenance")
def api_maintenance():
    return jsonify({
        "active": _maintenance_active(),
        "since": _maintenance["since"],
        "reason": _maintenance["reason"],
    })


@app.before_request
def _maintenance_gate():
    from flask import request as _rq
    if _rq.path == "/api/maintenance":
        return  # status is always reachable
    if not _maintenance_active():
        return
    if _rq.path.startswith("/api/") or _rq.path.startswith("/stream"):
        return jsonify({"maintenance": True, "reason": _maintenance["reason"]}), 503
    html = (
        "<html><head><meta charset='utf-8'><meta http-equiv='refresh' content='30'>"
        "<title>BirdBuddy — Maintenance</title></head>"
        "<body style='background:#1a1a1a;color:#ccc;font-family:system-ui;"
        "text-align:center;padding:80px 20px'>"
        "<h2>&#128736; BirdBuddy is doing nightly maintenance</h2>"
        "<p>" + (_maintenance["reason"] or "") + "</p>"
        "<p style='color:#666'>The live view and gallery will be back shortly.</p>"
        "</body></html>"
    )
    return Response(html, status=503, mimetype="text/html")

Path("logs").mkdir(exist_ok=True)
log.info("BirdBuddy starting")
_settings = cfg.load()
_camera = Camera(cam_id=0)
_camera1 = Camera(cam_id=1)
import time as _time
_activity_lock = threading.Lock()
_active_streams = 0
_last_request_at = 0.0
CLIENT_GRACE_SECS = 20.0
# Endpoints that poll on a timer — they shouldn't count as "someone browsing"
# or an idle open tab would suppress slow-mo forever.
_ACTIVITY_EXCLUDE = {"/api/motion-status", "/api/slowmo-status"}


def clients_active():
    """True if the live stream is open or a page was served very recently.
    Used to suppress slow-mo bursts while someone is viewing the site, since
    the 120fps camera reconfiguration both interrupts the live feed and has
    been the trigger for camera-pipeline hangs under concurrent load."""
    with _activity_lock:
        if _active_streams > 0:
            return True
    return (_time.time() - _last_request_at) < CLIENT_GRACE_SECS


_detector = MotionDetector(_camera, lambda: _settings, clients_active=clients_active)
_timelapse = TimelapseCapturer(_camera, lambda: _settings)
_cleaner = DiskCleaner(lambda: _settings)
_daynight = DayNightManager(_camera, lambda: _settings)
_backup = BackupScheduler(lambda: _settings)
_slowmo_verifier = SlowMoVerifier(lambda: _settings, clients_active=clients_active, set_maintenance=set_maintenance)
_camera.start()
_camera.apply_settings(_settings)
if _camera1.available:
    _camera1.start()
_detector.start()
_timelapse.start()
_cleaner.start()
_daynight.start()
_backup.start()
_slowmo_verifier.start()
log.info("Camera ready — http://birdbuddy.local:8080/")


@app.route("/")
def index():
    return render_template("index.html", settings_json=json.dumps(_settings))


def _counted_stream(cam_id):
    """Wrap generate_stream to track how many live viewers are connected."""
    global _active_streams
    with _activity_lock:
        _active_streams += 1
    try:
        for chunk in generate_stream(cam_id):
            yield chunk
    finally:
        with _activity_lock:
            _active_streams -= 1


@app.route("/stream")
def stream():
    return Response(
        _counted_stream(0),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stream/<int:cam_id>")
def stream_cam(cam_id):
    if cam_id not in (0, 1):
        abort(404)
    return Response(
        _counted_stream(cam_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stream/mobile")
def stream_mobile():
    return Response(
        _counted_stream("mobile"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/settings", methods=["GET"])
def get_settings():
    return jsonify(_settings)


@app.route("/settings", methods=["POST"])
def post_settings():
    global _settings
    data = request.get_json(force=True)
    _settings.update(data)
    cfg.save(_settings)
    _camera.apply_settings(_settings)
    log.info(f"Settings updated: {data}")
    return jsonify({"ok": True})


CAPTURES_DIR = Path(__file__).parent / "captures"
LOG_FILE = Path(__file__).parent / "logs" / "birdbuddy.log"
BIRD_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*BIRD DETECTED: (.+?) \((\d+\.\d+)%\) → (motion_\S+\.jpg)")

# Filenames the user has explicitly deleted. Stats and sightings are derived
# from the log file, which keeps the BIRD DETECTED line even after the image
# is gone — so we filter these out to keep counts in sync with deletions.
# (Auto-retention cleanup deliberately does NOT add here, so historical totals
# survive space-saving cleanup; only explicit user deletes subtract.)
DELETED_FILE = Path(__file__).parent / "deleted.json"
_deleted_lock = threading.Lock()


def _load_deleted():
    try:
        with open(DELETED_FILE) as f:
            return set(json.load(f))
    except Exception:
        return set()


def _add_deleted(filename):
    with _deleted_lock:
        s = _load_deleted()
        s.add(filename)
        try:
            with open(DELETED_FILE, "w") as f:
                json.dump(sorted(s), f)
        except Exception as e:
            log.warning(f"Failed to record deleted file {filename}: {e}")


@app.route("/sightings")
def sightings():
    entries = []
    deleted = _load_deleted()
    if LOG_FILE.exists():
        for line in LOG_FILE.read_text().splitlines():
            m = BIRD_RE.search(line)
            if m:
                ts, species, confidence, filename = m.groups()
                if filename in deleted:
                    continue
                entries.append({
                    "timestamp": ts,
                    "species": species,
                    "confidence": float(confidence),
                    "filename": filename,
                    "has_image": (CAPTURES_DIR / filename).exists(),
                })
    # Most recent first, cap at 50
    return jsonify(list(reversed(entries))[:50])


@app.route("/captures/<filename>")
def capture(filename):
    path = CAPTURES_DIR / filename
    if not path.exists() or not path.suffix == ".jpg":
        abort(404)
    return send_file(path, mimetype="image/jpeg")


THUMBS_DIR = Path(__file__).parent / "thumbnails"
THUMBS_DIR.mkdir(exist_ok=True)

@app.route("/captures/thumb/<filename>")
def capture_thumb(filename):
    if not filename.endswith(".jpg"):
        abort(404)
    thumb_path = THUMBS_DIR / filename
    if not thumb_path.exists():
        src = CAPTURES_DIR / filename
        if not src.exists():
            abort(404)
        from PIL import Image as _Img
        with _Img.open(src) as im:
            im.thumbnail((320, 180))
            im.save(thumb_path, "JPEG", quality=70)
    return send_file(thumb_path, mimetype="image/jpeg")


@app.route("/api/captures/<filename>", methods=["DELETE"])
def delete_capture(filename):
    if "/" in filename or "\\" in filename or ".." in filename or not filename.endswith(".jpg"):
        abort(400)
    path = CAPTURES_DIR / filename
    if not path.exists():
        abort(404)
    path.unlink()
    (THUMBS_DIR / filename).unlink(missing_ok=True)
    _add_deleted(filename)
    log.info(f"Deleted capture: {filename}")
    return jsonify({"ok": True})


@app.route("/gallery")
def gallery():
    return render_template("gallery.html")



@app.route("/slowmo-page")
def slowmo_page():
    return render_template("slowmo.html")


@app.route("/help")
def help_page():
    return render_template("help.html")


@app.route("/api/captures")
def api_captures():
    page = int(request.args.get("page", 0))
    per_page = 48
    filter_type = request.args.get("filter", "all")  # "all", "bird", "motion"

    # Build bird sightings index from log for labelling
    bird_index = {}
    if LOG_FILE.exists():
        for line in LOG_FILE.read_text().splitlines():
            m = BIRD_RE.search(line)
            if m:
                ts, species, confidence, filename = m.groups()
                bird_index[filename] = {"species": species, "confidence": float(confidence)}

    all_files = sorted(CAPTURES_DIR.glob("*.jpg"), reverse=True)

    # Apply server-side filter before pagination
    if filter_type == "bird":
        files = [f for f in all_files if f.name in bird_index]
    elif filter_type == "motion":
        files = [f for f in all_files if f.name not in bird_index]
    else:
        files = all_files

    total = len(files)
    page_files = files[page * per_page:(page + 1) * per_page]

    items = []
    for f in page_files:
        bird = bird_index.get(f.name)
        try:
            dt = datetime.strptime(f.stem, "motion_%Y%m%d_%H%M%S")
            ts = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            ts = ""
        items.append({
            "filename": f.name,
            "timestamp": ts,
            "is_bird": bird is not None,
            "species": bird["species"] if bird else None,
            "confidence": bird["confidence"] if bird else None,
        })

    return jsonify({"items": items, "total": total, "page": page, "per_page": per_page, "filter": filter_type})


@app.route("/stats")
def stats_page():
    return render_template("stats.html")


@app.route("/api/stats")
def api_stats():
    entries = []
    deleted = _load_deleted()
    today = datetime.now().date()
    if LOG_FILE.exists():
        for line in LOG_FILE.read_text().splitlines():
            m = BIRD_RE.search(line)
            if m:
                ts, species, confidence, filename = m.groups()
                if filename in deleted:
                    continue
                entries.append({"ts": datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"), "species": species})

    species_counts = Counter(e["species"] for e in entries)
    top_species = [{"species": s, "count": c} for s, c in species_counts.most_common(20)]
    today_count = sum(1 for e in entries if e["ts"].date() == today)

    by_hour = [0] * 24
    for e in entries:
        by_hour[e["ts"].hour] += 1

    # Last 30 days
    day_counts = defaultdict(int)
    for e in entries:
        day_counts[e["ts"].strftime("%Y-%m-%d")] += 1
    by_day = []
    for i in range(29, -1, -1):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        by_day.append({"date": d, "count": day_counts.get(d, 0)})

    total_captures = len(list(CAPTURES_DIR.glob("*.jpg"))) if CAPTURES_DIR.exists() else 0

    return jsonify({
        "total_sightings": len(entries),
        "unique_species": len(species_counts),
        "today_sightings": today_count,
        "total_captures": total_captures,
        "by_hour": by_hour,
        "by_day": by_day,
        "top_species": top_species,
    })


@app.route("/api/disk")
def api_disk():
    return jsonify(disk_usage())


@app.route("/api/backup", methods=["POST"])
def api_backup():
    dest = _settings.get("backup_path", "").strip()
    if not dest:
        return jsonify({"ok": False, "error": "No backup_path configured in settings"}), 400
    ok, msg = run_backup(dest)
    return jsonify({"ok": ok, "message": msg})


def _slowmo_entry(v):
    meta = {}
    sc = v.with_suffix(".json")
    if sc.exists():
        try:
            meta = json.loads(sc.read_text())
        except Exception:
            meta = {}
    size = v.stat().st_size
    species = meta.get("best_species") or meta.get("trigger_species")
    conf = meta.get("best_confidence")
    if conf is None:
        conf = meta.get("trigger_confidence")
    return {
        "filename": v.name,
        "timestamp": v.stem.replace("slowmo_", ""),
        "size_mb": round(size / 1024 / 1024, 1),
        "size_bytes": size,
        "species": species,
        "confidence": conf,
        "is_hummingbird": meta.get("trigger_is_hummingbird"),
        "verified": meta.get("verified"),
    }


@app.route("/slowmo")
def slowmo_list():
    videos = sorted(SLOWMO_DIR.glob("slowmo_*.mp4"), reverse=True)
    return jsonify([_slowmo_entry(v) for v in videos])


@app.route("/api/slowmo-rejected")
def slowmo_rejected_list():
    rej = SLOWMO_DIR / "rejected"
    if not rej.exists():
        return jsonify([])
    videos = sorted(rej.glob("slowmo_*.mp4"), reverse=True)
    return jsonify([_slowmo_entry(v) for v in videos])


@app.route("/slowmo/rejected/<filename>")
def slowmo_rejected_video(filename):
    if "/" in filename or "\\" in filename or ".." in filename or not filename.endswith(".mp4"):
        abort(400)
    path = SLOWMO_DIR / "rejected" / filename
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="video/mp4")


@app.route("/api/slowmo-rejected/<filename>", methods=["DELETE"])
def delete_slowmo_rejected(filename):
    if "/" in filename or "\\" in filename or ".." in filename or not filename.endswith(".mp4"):
        abort(400)
    path = SLOWMO_DIR / "rejected" / filename
    if not path.exists():
        abort(404)
    path.unlink()
    path.with_suffix(".json").unlink(missing_ok=True)
    log.info(f"Deleted quarantined slow-mo: {filename}")
    return jsonify({"ok": True})


@app.route("/api/slowmo/restore/<filename>", methods=["POST"])
def slowmo_restore(filename):
    if "/" in filename or "\\" in filename or ".." in filename or not filename.endswith(".mp4"):
        abort(400)
    src = SLOWMO_DIR / "rejected" / filename
    if not src.exists():
        abort(404)
    src.rename(SLOWMO_DIR / filename)
    sc = src.with_suffix(".json")
    if sc.exists():
        try:
            m = json.loads(sc.read_text())
        except Exception:
            m = {}
        m["verified"] = True  # so the verifier won't re-quarantine it
        m["restored_at"] = datetime.now().isoformat(timespec="seconds")
        (SLOWMO_DIR / sc.name).write_text(json.dumps(m))
        sc.unlink(missing_ok=True)
    log.info(f"Restored slow-mo from quarantine: {filename}")
    return jsonify({"ok": True})


@app.route("/api/slowmo/verify", methods=["POST"])
def slowmo_verify_now():
    import threading as _t
    _t.Thread(target=lambda: _slowmo_verifier.run_pass(force=True), daemon=True).start()
    return jsonify({"ok": True, "message": "Verification pass started in background"})


@app.route("/slowmo/<filename>")
def slowmo_video(filename):
    path = SLOWMO_DIR / filename
    if not path.exists() or path.suffix != ".mp4":
        abort(404)
    return send_file(path, mimetype="video/mp4")


@app.route("/api/slowmo/<filename>", methods=["DELETE"])
def delete_slowmo(filename):
    if "/" in filename or "\\" in filename or ".." in filename or not filename.endswith(".mp4"):
        abort(400)
    path = SLOWMO_DIR / filename
    if not path.exists():
        abort(404)
    path.unlink()
    log.info(f"Deleted slow-mo video: {filename}")
    return jsonify({"ok": True})


@app.route("/api/slowmo-status")
def slowmo_status():
    return jsonify({"capturing": is_capturing()})


@app.route("/api/motion-status")
def motion_status():
    return jsonify(_detector.get_status())


@app.route("/timelapse/build", methods=["POST"])
def timelapse_build():
    fps = int(request.args.get("fps", 10))
    output, msg = build_video(fps=fps)
    if output is None:
        return jsonify({"ok": False, "error": msg}), 500
    return jsonify({"ok": True, "message": msg, "url": "/timelapse/video"})


@app.route("/timelapse/video")
def timelapse_video():
    path = TIMELAPSE_DIR / "timelapse.mp4"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="video/mp4")


@app.route("/timelapse/status")
def timelapse_status():
    frames = sorted(TIMELAPSE_DIR.glob("tl_*.jpg"))
    video = TIMELAPSE_DIR / "timelapse.mp4"
    return jsonify({
        "frame_count": len(frames),
        "oldest": frames[0].stem.replace("tl_", "") if frames else None,
        "newest": frames[-1].stem.replace("tl_", "") if frames else None,
        "video_exists": video.exists(),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
