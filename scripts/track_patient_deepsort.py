# scripts/track_patient_deepsort.py

import cv2, numpy as np, os, sys
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from insightface.app import FaceAnalysis

# --- Dynamic Input Reading ---
# The script expects three arguments: <video_in_path> <ref_emb_npy_path> <video_out_path>
if len(sys.argv) < 4:
    # If run standalone, provide a clear error. In app.py, this is handled by subprocess.
    sys.exit("Usage: python script.py <video_in> <ref_emb_npy> <video_out>")

VIDEO_IN_PATH     = sys.argv[1]
REF_EMB_NPY_PATH  = sys.argv[2]
VIDEO_OUT_PATH    = sys.argv[3]
# -----------------------------

# ---------------- Config ----------------
PERSON_CONF_MIN = 0.35
ARC_SIM_THRESHOLD = 0.40
SIM_SMOOTH = 0.7
LOST_PATIENCE = 30

# ---------------- Utils ----------------
def iou(a, b):
    # Calculates Intersection over Union (IoU)
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2,bx2) - max(ax1,bx1))
    inter_h = max(0, min(ay2,by2) - max(ay1,by1))
    inter = inter_w * inter_h
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by2) - inter
    return inter / ua if ua > 0 else 0.0

def box_contains_point(box, x, y):
    x1,y1,x2,y2 = box
    return (x1 <= x <= x2) and (y1 <= y <= y2)

# ---------------- Init models ----------------
# Use ctx_id=-1 to force CPU usage on Cloud Run
print("[Init] Loading ArcFace/InsightFace...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1, det_size=(640, 640)) 

print("[Init] Loading YOLOv8 person detector...")
detector = YOLO("yolov8n.pt") 

print("[Init] Loading Deep SORT tracker...")
tracker = DeepSort(
    max_age=LOST_PATIENCE,
    n_init=3,
    max_cosine_distance=0.4,
    nms_max_overlap=1.0,
    embedder="mobilenet",
    half=True
)

print("[Init] Loading reference embedding...")
# Check if embedding file exists before attempting to load
if not os.path.exists(REF_EMB_NPY_PATH):
    sys.exit(f"Error: Reference embedding not found at {REF_EMB_NPY_PATH}")

REF = np.load(REF_EMB_NPY_PATH).astype("float32")

# ---------------- Video IO ----------------
cap = cv2.VideoCapture(VIDEO_IN_PATH)
if not cap.isOpened():
    sys.exit(f"Cannot open video file: {VIDEO_IN_PATH}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
# Ensure the output directory exists
Path(os.path.dirname(VIDEO_OUT_PATH) or ".").mkdir(parents=True, exist_ok=True)

# Use 'mp4v' or 'avc1' for H.264 compatibility on servers
writer = cv2.VideoWriter(VIDEO_OUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# ---------------- State ----------------
patient_track_id = None
frames_since_face = LOST_PATIENCE + 1
last_face_bbox = None
ema_sim = 0.0

# ---------------- Main loop ----------------
while True:
    ok, frame = cap.read()
    if not ok:
        break

    # 1) Detect faces & compute best ArcFace similarity
    faces = face_app.get(frame)
    best_face, best_sim = None, -1.0
    for f in faces:
        emb = f.normed_embedding.astype("float32")
        sim = float(np.dot(emb, REF))
        if sim > best_sim:
            best_sim, best_face = sim, f

    if best_face is not None:
        ema_sim = SIM_SMOOTH * ema_sim + (1 - SIM_SMOOTH) * max(0.0, best_sim)
    else:
        ema_sim = SIM_SMOOTH * ema_sim

    face_is_patient = (best_face is not None) and (ema_sim >= ARC_SIM_THRESHOLD)

    # 2) Detect persons (YOLO)
    yres = detector(frame, verbose=False)[0]
    detections = []
    for b in yres.boxes:
        cls = int(b.cls[0].item())
        if cls != 0:
            continue
        conf = float(b.conf[0].item())
        if conf < PERSON_CONF_MIN:
            continue
        x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
        detections.append(([x1,y1,x2,y2], conf, "person"))

    # 3) Track with Deep SORT
    tracks = tracker.update_tracks(detections, frame=frame)

    # 4) Bind recognized face -> track 
    if face_is_patient:
        x1,y1,x2,y2 = map(int, best_face.bbox)
        last_face_bbox = (x1,y1,x2,y2)
        fx = (x1 + x2) / 2.0; fy = (y1 + y2) / 2.0

        candidates = []
        for t in tracks:
            if not t.is_confirmed(): 
                continue
            l,t_,r,b = map(int, t.to_ltrb())
            contains = box_contains_point((l,t_,r,b), fx, fy)
            ov = 1.0 if contains else iou((x1,y1,x2,y2), (l,t_,r,b))
            candidates.append((t.track_id, (l,t_,r,b), ov))
        if candidates:
            candidates.sort(key=lambda z: z[2], reverse=True)
            patient_track_id = candidates[0][0]
            frames_since_face = 0
    else:
        frames_since_face += 1

    # 5) Draw + isolate patient
    blurred = cv2.GaussianBlur(frame, (31,31), 0)
    mask = np.zeros((h, w), dtype=np.uint8)
    patient_box_drawn = False

    for t in tracks:
        if not t.is_confirmed():
            continue
        l,t_,r,b = map(int, t.to_ltrb())
        tid = t.track_id

        if patient_track_id is not None and tid == patient_track_id:
            # Patient region: keep sharp and mark
            cv2.rectangle(frame, (l,t_), (r,b), (255,200,0), 2)
            cv2.rectangle(mask, (l,t_), (r,b), 255, -1)
            patient_box_drawn = True
        
        # Drawing boxes/text is mainly for visual inspection, omit or simplify for server speed
        # cv2.putText(frame, f"id={tid}", (l, max(20, t_-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Composite: keep patient sharp; blur others
    if patient_track_id is not None and patient_box_drawn:
        composite = np.where(mask[...,None]==255, frame, blurred)
    else:
        composite = frame

    # Add text overlay to indicate status
    status_text = f"Sim: {ema_sim:.2f}"
    if patient_track_id is not None:
        status_text += f" | Track: {patient_track_id}"
        if frames_since_face > 0 and frames_since_face <= LOST_PATIENCE:
            status_text += f" (Reacq: {frames_since_face})"
    
    cv2.putText(composite, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,180,255), 2)


    writer.write(composite)

cap.release(); writer.release()
print(f"[Done] Processed video saved to: {VIDEO_OUT_PATH}")