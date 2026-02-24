import os
import uuid
import json
import sqlite3
from datetime import datetime

from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------- CONFIG -----------------
UPLOAD_VIDEO_DIR = "uploads/videos"
UPLOAD_MODEL_DIR = "uploads/models"
RESULTS_DIR = "results"
DB_PATH = "traffic_runs.db"

DEFAULT_MODEL_PATH = "yolov8s.pt"  # put a YOLO model here or change to your fine-tuned .pt

PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}  # COCO: bicycle, car, motorcycle, bus, truck

os.makedirs(UPLOAD_VIDEO_DIR, exist_ok=True)
os.makedirs(UPLOAD_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = Flask(__name__)


# ----------------- DB UTILS -----------------
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            video_path TEXT,
            model_path TEXT,
            people_count INTEGER,
            vehicle_count INTEGER,
            suggestion TEXT,
            summary_json TEXT,
            preview_path TEXT,
            heatmap_path TEXT
        )
        """
    )
    conn.commit()
    conn.close()


# ----------------- CV UTILS -----------------
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def draw_detections(frame, dets):
    img = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cls = d["class"]
        conf = d["conf"]
        if cls == PERSON_CLASS_ID:
            color = (0, 255, 0)
            label = f"person {conf:.2f}"
        elif cls in VEHICLE_CLASS_IDS:
            color = (0, 0, 255)
            label = f"vehicle {conf:.2f}"
        else:
            color = (255, 255, 0)
            label = f"class{cls} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def compute_heatmap(frame_shape, centroids):
    h, w = frame_shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for (cx, cy) in centroids:
        if 0 <= cx < w and 0 <= cy < h:
            heat[cy, cx] += 1.0
    # blur to make density smooth
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=25, sigmaY=25)
    if heat.max() > 0:
        heat_norm = (heat / heat.max() * 255).astype(np.uint8)
    else:
        heat_norm = heat.astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    return heat_color


def suggest_traffic_logic(people_count, vehicle_count,
                          person_threshold=10, vehicle_threshold=15, extend_seconds=10):
    suggestions = []
    if people_count > person_threshold:
        suggestions.append(
            f"High pedestrian volume ({people_count} > {person_threshold}): "
            f"extend pedestrian green by ~{extend_seconds} seconds."
        )
    else:
        suggestions.append(
            f"Pedestrian count ({people_count}) is below threshold ({person_threshold})."
        )

    if vehicle_count > vehicle_threshold:
        suggestions.append(
            f"High vehicle density ({vehicle_count} > {vehicle_threshold}): "
            f"prioritize vehicle green or adjust cycle splits."
        )
    else:
        suggestions.append(
            f"Vehicle count ({vehicle_count}) is below threshold ({vehicle_threshold})."
        )

    if people_count > person_threshold and vehicle_count > vehicle_threshold:
        suggestions.append(
            "Both pedestrians and vehicles are high: consider adaptive phasing "
            "or alternating extended phases for each direction."
        )

    return " ".join(suggestions)


def process_video(run_id, video_path, model_path,
                  do_preprocess=False,
                  target_width=None,
                  sample_fps=5.0,
                  conf_thresh=0.35):
    model = load_model(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Decide sampling step
    sample_step = max(int(round(orig_fps / sample_fps)), 1)

    total_people = 0
    total_vehicles = 0
    all_centroids = []

    preview_frame = None
    frame_idx = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_step != 0:
            continue

        # optional preprocess
        if do_preprocess and target_width is not None:
            scale = target_width / frame_w
            new_h = int(frame_h * scale)
            frame_proc = cv2.resize(frame, (target_width, new_h))
        else:
            frame_proc = frame

        results = model(frame_proc, conf=conf_thresh, imgsz=640)[0]

        dets = []
        people_this = 0
        vehicles_this = 0

        for box in results.boxes:
            cls = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            det = {
                "class": cls,
                "conf": conf,
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            }
            dets.append(det)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if cls == PERSON_CLASS_ID:
                people_this += 1
                all_centroids.append((int(cx), int(cy)))
            elif cls in VEHICLE_CLASS_IDS:
                vehicles_this += 1
                all_centroids.append((int(cx), int(cy)))

        total_people += people_this
        total_vehicles += vehicles_this

        if preview_frame is None:
            preview_frame = draw_detections(frame_proc, dets)

        processed_frames += 1

    cap.release()

    if preview_frame is None:
        # no frames processed, just take first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("No frames in video")
        preview_frame = frame

    heatmap = compute_heatmap(preview_frame.shape, all_centroids)
    overlay = cv2.addWeighted(preview_frame, 0.6, heatmap, 0.4, 0)

    preview_path = os.path.join(RESULTS_DIR, f"{run_id}_preview.png")
    heatmap_path = os.path.join(RESULTS_DIR, f"{run_id}_heatmap.png")
    cv2.imwrite(preview_path, overlay)
    cv2.imwrite(heatmap_path, heatmap)

    summary = {
        "frames_sampled": processed_frames,
        "orig_fps": orig_fps,
        "sample_fps": sample_fps,
        "total_people_counted": int(total_people),
        "total_vehicles_counted": int(total_vehicles),
    }

    suggestion = suggest_traffic_logic(total_people, total_vehicles)

    return {
        "people": int(total_people),
        "vehicles": int(total_vehicles),
        "preview_path": preview_path,
        "heatmap_path": heatmap_path,
        "summary": summary,
        "suggestion": suggestion,
    }


# ----------------- ROUTES -----------------
@app.route("/", methods=["GET", "POST"])
def index():
    conn = get_db()
    cur = conn.cursor()

    latest_run = None

    if request.method == "POST":
        video_file = request.files.get("video")
        model_file = request.files.get("model")
        do_preprocess = request.form.get("preprocess") == "on"
        target_width = request.form.get("target_width") or ""
        sample_fps = request.form.get("sample_fps") or "5"
        conf_thresh = request.form.get("conf_thresh") or "0.35"

        try:
            target_width = int(target_width) if target_width else None
        except ValueError:
            target_width = None

        try:
            sample_fps = float(sample_fps)
        except ValueError:
            sample_fps = 5.0

        try:
            conf_thresh = float(conf_thresh)
        except ValueError:
            conf_thresh = 0.35

        if not video_file:
            conn.close()
            return "No video uploaded", 400

        run_id = uuid.uuid4().hex
        ts = datetime.utcnow().isoformat()

        video_ext = os.path.splitext(video_file.filename)[1]
        video_name = f"{run_id}{video_ext}"
        video_path = os.path.join(UPLOAD_VIDEO_DIR, video_name)
        video_file.save(video_path)

        if model_file and model_file.filename:
            model_ext = os.path.splitext(model_file.filename)[1]
            model_name = f"{run_id}{model_ext}"
            model_path = os.path.join(UPLOAD_MODEL_DIR, model_name)
            model_file.save(model_path)
        else:
            model_path = DEFAULT_MODEL_PATH

        result = process_video(
            run_id=run_id,
            video_path=video_path,
            model_path=model_path,
            do_preprocess=do_preprocess,
            target_width=target_width,
            sample_fps=sample_fps,
            conf_thresh=conf_thresh,
        )

        cur.execute(
            """
            INSERT INTO runs (
                id, created_at, video_path, model_path,
                people_count, vehicle_count, suggestion,
                summary_json, preview_path, heatmap_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                ts,
                video_path,
                model_path,
                result["people"],
                result["vehicles"],
                result["suggestion"],
                json.dumps(result["summary"]),
                result["preview_path"],
                result["heatmap_path"],
            ),
        )
        conn.commit()

        latest_run = {
            "id": run_id,
            "created_at": ts,
            "video_path": video_path,
            "model_path": model_path,
            "people_count": result["people"],
            "vehicle_count": result["vehicles"],
            "suggestion": result["suggestion"],
            "summary": result["summary"],
            "preview_file": os.path.basename(result["preview_path"]),
            "heatmap_file": os.path.basename(result["heatmap_path"]),
        }

    cur.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 10")
    rows = cur.fetchall()
    history = []
    for r in rows:
        history.append({
            "id": r["id"],
            "created_at": r["created_at"],
            "people_count": r["people_count"],
            "vehicle_count": r["vehicle_count"],
            "suggestion": r["suggestion"],
        })

    conn.close()

    return render_template("index.html", latest_run=latest_run, history=history)


@app.route("/results/<path:filename>")
def results_file(filename):
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
