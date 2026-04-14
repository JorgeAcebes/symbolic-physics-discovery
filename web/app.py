"""
symbolic-physics-discovery / web / app.py
Flask backend — serves the UI and exposes the API.
All outputs are isolated to web/outputs/<job_id>/  (never touches results/ or data/).
"""
import pysr #<--- No tocar. Debe ser llamado lo primero de todo 
import os
import sys
import json
import uuid
import threading
import io
from pathlib import Path

from flask import Flask, request, jsonify, send_file, render_template

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_WEB  = Path(__file__).parent
SRC_DIR   = BASE_WEB.parent / "src"
DATA_DIR  = BASE_WEB.parent / "data"
UPLOADS   = BASE_WEB / "uploads"
OUTPUTS   = BASE_WEB / "outputs"

UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB upload limit

# In-memory job store  { job_id: { status, log, files, summary, error } }
_jobs: dict = {}
_lock = threading.Lock()

ALLOWED_EXT = {"csv", "json", "txt"}


# ── HTML ───────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── API: datasets ──────────────────────────────────────────────────────────────
@app.route("/api/datasets")
def list_datasets():
    """Return the CSV files found in data/."""
    files = []
    if DATA_DIR.exists():
        for f in sorted(DATA_DIR.glob("*.csv")):
            try:
                import pandas as pd
                df = pd.read_csv(f, nrows=1)
                files.append({
                    "name":    f.name,
                    "columns": df.columns.tolist(),
                    "path":    str(f),
                })
            except Exception:
                files.append({"name": f.name, "columns": [], "path": str(f)})
    return jsonify(files)


# ── API: parse uploaded file ───────────────────────────────────────────────────
@app.route("/api/parse", methods=["POST"])
def parse_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    ext  = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported type .{ext}. Use CSV, JSON or TXT."}), 400

    content = file.read().decode("utf-8", errors="replace")

    try:
        import pandas as pd

        if ext == "csv":
            df = pd.read_csv(io.StringIO(content))
        elif ext == "json":
            df = pd.read_json(io.StringIO(content))
        else:  # txt — try tab / space / comma
            df = pd.read_csv(io.StringIO(content), sep=None, engine="python")

        # Detect if first row was numeric (no headers)
        def _looks_numeric(s):
            try:
                float(s)
                return True
            except (ValueError, TypeError):
                return False

        has_headers = not all(_looks_numeric(c) for c in df.columns.astype(str))

        # Persist as CSV so runner can reload it
        file_id   = str(uuid.uuid4())
        save_path = UPLOADS / f"{file_id}.csv"
        df.to_csv(save_path, index=False)

        return jsonify({
            "file_id":     file_id,
            "columns":     df.columns.tolist(),
            "preview":     df.head(5).values.tolist(),
            "n_rows":      len(df),
            "has_headers": has_headers,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── API: run ───────────────────────────────────────────────────────────────────
@app.route("/api/run", methods=["POST"])
def run_experiment():
    config = request.json or {}
    job_id = str(uuid.uuid4())

    with _lock:
        _jobs[job_id] = {
            "status":  "queued",
            "log":     [],
            "files":   [],
            "summary": [],
            "error":   None,
        }

    t = threading.Thread(target=_execute_job, args=(job_id, config), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


# ── API: poll job ──────────────────────────────────────────────────────────────
@app.route("/api/job/<job_id>")
def get_job(job_id):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


# ── API: serve output image ────────────────────────────────────────────────────
@app.route("/api/job/<job_id>/image/<path:filepath>")
def serve_image(job_id, filepath):
    path = OUTPUTS / job_id / filepath
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(str(path), mimetype="image/png")


# ── API: download output file ──────────────────────────────────────────────────
@app.route("/api/job/<job_id>/download/<path:filepath>")
def download_file(job_id, filepath):
    path = OUTPUTS / job_id / filepath
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(str(path), as_attachment=True, download_name=path.name)


# ── Job worker ─────────────────────────────────────────────────────────────────
def _execute_job(job_id: str, config: dict):
    def log(msg: str):
        with _lock:
            _jobs[job_id]["log"].append(msg)

    job_dir = OUTPUTS / job_id
    job_dir.mkdir(exist_ok=True)

    try:
        with _lock:
            _jobs[job_id]["status"] = "running"

        # Make src/ importable
        src = str(SRC_DIR)
        if src not in sys.path:
            sys.path.insert(0, src)

        # Delegate to runner
        from runner import run_experiment_internal
        summary = run_experiment_internal(job_id, config, job_dir, log)

        # Collect generated files
        files = []
        for f in sorted(job_dir.rglob("*")):
            if f.is_file() and f.suffix in (".png", ".txt", ".json"):
                rel = f.relative_to(job_dir)
                files.append({
                    "name": f.name,
                    "path": str(rel).replace("\\", "/"),
                    "type": "image" if f.suffix == ".png" else "text",
                })

        with _lock:
            _jobs[job_id]["status"]  = "done"
            _jobs[job_id]["files"]   = files
            _jobs[job_id]["summary"] = summary

        log("Experimento completado.")

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log(f"{exc}")
        log(tb)
        with _lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = str(exc)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"  Project root : {BASE_WEB.parent}")
    print(f"  Data dir     : {DATA_DIR}")
    print(f"  Outputs dir  : {OUTPUTS}")
    print()
    app.run(debug=True, use_reloader=False, port=5050, threaded=True)

