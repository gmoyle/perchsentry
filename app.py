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
from slowmo import SLOWMO_DIR
from cleanup import DiskCleaner, disk_usage
from daynight import DayNightManager
from backup import BackupScheduler, run_backup

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

Path("logs").mkdir(exist_ok=True)
log.info("BirdBuddy starting")
_settings = cfg.load()
_camera = Camera(cam_id=0)
_camera1 = Camera(cam_id=1)
_detector = MotionDetector(_camera, lambda: _settings)
_timelapse = TimelapseCapturer(_camera, lambda: _settings)
_cleaner = DiskCleaner(lambda: _settings)
_daynight = DayNightManager(_camera, lambda: _settings)
_backup = BackupScheduler(lambda: _settings)
_camera.start()
_camera.apply_settings(_settings)
if _camera1.available:
    _camera1.start()
_detector.start()
_timelapse.start()
_cleaner.start()
_daynight.start()
_backup.start()
log.info("Camera ready — http://birdbuddy.local:8080/")


@app.route("/")
def index():
    return render_template("index.html", settings_json=json.dumps(_settings))


@app.route("/stream")
def stream():
    return Response(
        generate_stream(0),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stream/<int:cam_id>")
def stream_cam(cam_id):
    if cam_id not in (0, 1):
        abort(404)
    return Response(
        generate_stream(cam_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stream/mobile")
def stream_mobile():
    return Response(
        generate_stream("mobile"),
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


@app.route("/sightings")
def sightings():
    entries = []
    if LOG_FILE.exists():
        for line in LOG_FILE.read_text().splitlines():
            m = BIRD_RE.search(line)
            if m:
                ts, species, confidence, filename = m.groups()
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
    today = datetime.now().date()
    if LOG_FILE.exists():
        for line in LOG_FILE.read_text().splitlines():
            m = BIRD_RE.search(line)
            if m:
                ts, species, confidence, filename = m.groups()
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


@app.route("/slowmo")
def slowmo_list():
    videos = sorted(SLOWMO_DIR.glob("slowmo_*.mp4"), reverse=True)
    return jsonify([{
        "filename": v.name,
        "timestamp": v.stem.replace("slowmo_", ""),
        "size_mb": round(v.stat().st_size / 1024 / 1024, 1),
    } for v in videos])


@app.route("/slowmo/<filename>")
def slowmo_video(filename):
    path = SLOWMO_DIR / filename
    if not path.exists() or path.suffix != ".mp4":
        abort(404)
    return send_file(path, mimetype="video/mp4")


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
