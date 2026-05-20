from ultralytics import YOLO
import cv2
import time
import numpy as np
import csv
import os
from collections import defaultdict
from norfair import Tracker, Detection

# CONFIG

PPE_MODEL_PATH  = "11s.pt"
POSE_MODEL_PATH = "yolov8n-pose.pt"
SOURCE          = "mixed_comp2.mp4"

OUTPUT_DIR      = "evaluation_resultsrow1"
CAPTURE_DIR     = os.path.join(OUTPUT_DIR, "compliant_captures")
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(CAPTURE_DIR, exist_ok=True)

IMG_W, IMG_H = 800, 800
PPE_CONF     = 0.25
POSE_CONF    = 0.65

PPE_CONF_THRESH = {
    "helmet": 0.25,
    "vest":   0.25,
    "gloves": 0.01,
    "boots":  0.05,
}

PPE_EVERY_N_FRAMES  = 2
POSE_EVERY_N_FRAMES = 2

TH_HELMET = 40
TH_VEST   = 80
TH_GLOVE  = 45
TH_BOOT   = 45

MANDATORY_FRAMES = 5
OPTIONAL_FRAMES  = 3
MEMORY_TTL       = 60

# PPE COLORS

PPE_COLORS = {
    "helmet": (0, 255, 255),
    "vest":   (0, 165, 255),
    "gloves": (255, 0, 255),
    "boots":  (255, 255, 0),
}

# LOAD MODELS

ppe_model  = YOLO(PPE_MODEL_PATH)
pose_model = YOLO(POSE_MODEL_PATH)

# NORFAIR

def center_iou_distance(det, track):
    d_x1, d_y1, d_x2, d_y2 = det.data
    if track.last_detection is None:
        return 1.0
    t_x1, t_y1, t_x2, t_y2 = track.last_detection.data
    cx, cy = track.estimate.flatten()[:2]
    w = t_x2 - t_x1
    h = t_y2 - t_y1
    p_x1, p_y1 = cx - w / 2, cy - h / 2
    p_x2, p_y2 = cx + w / 2, cy + h / 2
    ix1, iy1 = max(d_x1, p_x1), max(d_y1, p_y1)
    ix2, iy2 = min(d_x2, p_x2), min(d_y2, p_y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_d = (d_x2 - d_x1) * (d_y2 - d_y1)
    area_p = w * h
    union = area_d + area_p - inter
    return 1.0 - (inter / union) if union > 0 else 1.0

tracker = Tracker(
    distance_function=center_iou_distance,
    distance_threshold=0.7,
    hit_counter_max=200,
    initialization_delay=1,
)

# HELPERS

def near(p, q, th):
    dx = p[0] - float(q[0])
    dy = p[1] - float(q[1])
    return (dx * dx + dy * dy) ** 0.5 < th

def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)

def find_pose_idx(track_box, pose_boxes):
    best_iou, best_idx = 0.0, None
    for i, pb in enumerate(pose_boxes):
        iou = bbox_iou(track_box, pb)
        if iou > best_iou:
            best_iou, best_idx = iou, i
    return best_idx if best_iou > 0 else None

def draw_keypoints_only(img, pose_result, conf_thres):
    if pose_result is None or pose_result.keypoints is None:
        return img
    kpts_xy   = pose_result.keypoints.xy
    kpts_conf = pose_result.keypoints.conf
    for p in range(len(kpts_xy)):
        for k in range(kpts_xy.shape[1]):
            if float(kpts_conf[p, k]) < conf_thres:
                continue
            x, y = int(kpts_xy[p, k][0]), int(kpts_xy[p, k][1])
            cv2.circle(img, (x, y), 3, (0, 255, 255), -1)
    return img

def draw_ppe_boxes(img, ppe_labels):
    for label, cx, cy, x1, y1, x2, y2, conf in ppe_labels:
        color = PPE_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img, f"{label} {conf:.2f}",
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )
    return img

# COMPLIANT CAPTURE

def save_compliant_capture(frame, tid, frame_count, box, ppe_labels):
    """
    Draws a highlighted box + PPE labels on a clean copy of
    the frame and saves it to CAPTURE_DIR.
    """
    capture = frame.copy()

    # draw PPE boxes
    capture = draw_ppe_boxes(capture, ppe_labels)

    # draw prominent person box
    x1, y1, x2, y2 = box
    cv2.rectangle(capture, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # banner above box
    banner = f"ID {tid} | COMPLIANT | frame {frame_count}"
    cv2.putText(
        capture, banner,
        (x1, max(y1 - 12, 15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2,
    )

    # timestamp watermark
    cv2.putText(
        capture, f"frame {frame_count}",
        (10, IMG_H - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
    )

    path = os.path.join(CAPTURE_DIR, f"compliant_ID{tid}_frame{frame_count}.jpg")
    cv2.imwrite(path, capture)
    print(f"[CAPTURE] ID {tid} compliant → {path}")

# CSV SETUP

csv_path   = os.path.join(OUTPUT_DIR, "frame_metrics.csv")
csv_file   = open(csv_path, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "frame",
    "total_detected",
    "compliant_detected",
    "non_compliant_detected",
    "non_compliance_rate_pct",
    "fps",
    "proc_time_ms",
    "id_switches_cumulative",
])

# METRICS ACCUMULATORS

total_frames        = 0
sum_detected        = 0
sum_compliant       = 0
sum_non_compliant   = 0
fps_list            = []
proc_time_list      = []

id_switch_count     = 0
track_last_state    = {}
track_seen_frames   = defaultdict(int)

# tracks already captured — save once per unique ID
saved_compliant_tids = set()

# VIDEO

cap         = cv2.VideoCapture(SOURCE)
frame_count = 0
prev_time   = time.time()
fps         = 0

last_pose_result = None
last_ppe_labels  = []
pose_boxes       = []
mem              = {}

print("Press 'q' to quit")

# MAIN LOOP

while cap.isOpened():

    frame_start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (IMG_W, IMG_H))

    # PPE DETECTION
    
    if frame_count % PPE_EVERY_N_FRAMES == 0 or frame_count == 1:
        last_ppe_labels = []
        ppe_res = ppe_model.predict(
            frame, conf=PPE_CONF, imgsz=IMG_W, verbose=False
        )[0]
        if ppe_res.boxes is not None:
            for b in ppe_res.boxes:
                conf  = float(b.conf[0])
                cls   = int(b.cls[0])
                label = ppe_model.names[cls].lower()
                if label not in PPE_CONF_THRESH:
                    continue
                if conf < PPE_CONF_THRESH[label]:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                last_ppe_labels.append((label, cx, cy, x1, y1, x2, y2, conf))
    
    # POSE ESTIMATION
    
    if frame_count % POSE_EVERY_N_FRAMES == 0 or last_pose_result is None:
        last_pose_result = pose_model.predict(
            frame, conf=POSE_CONF, imgsz=IMG_W, verbose=False
        )[0]
        pose_boxes = []
        if (
            last_pose_result.boxes is not None and
            last_pose_result.keypoints is not None
        ):
            for box, cls in zip(
                last_pose_result.boxes.xyxy.cpu().numpy(),
                last_pose_result.boxes.cls.cpu().numpy(),
            ):
                if int(cls) != 0:
                    continue
                pose_boxes.append(tuple(box.astype(float)))

    annotated = frame.copy()
    annotated = draw_keypoints_only(annotated, last_pose_result, POSE_CONF)
    annotated = draw_ppe_boxes(annotated, last_ppe_labels)

    # NORFAIR DETECTIONS
    
    detections = []
    for pb in pose_boxes:
        x1, y1, x2, y2 = pb
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        detections.append(
            Detection(
                points=np.array([[cx, cy]], dtype=np.float32),
                data=(x1, y1, x2, y2),
            )
        )

    tracked_objects = tracker.update(detections)

    kpts_xy = (
        last_pose_result.keypoints.xy
        if last_pose_result.keypoints is not None
        else None
    )
    
    # COMPLIANCE PER TRACK
    
    frame_compliant     = 0
    frame_non_compliant = 0

    for obj in tracked_objects:
        if obj.last_detection is None:
            continue

        tid = obj.id
        x1, y1, x2, y2 = map(int, obj.last_detection.data)

        if tid not in mem:
            mem[tid] = {
                "helmet": 0, "vest": 0,
                "gloves": 0, "boots": 0,
                "last_seen": frame_count,
            }

        pidx = find_pose_idx((x1, y1, x2, y2), pose_boxes)

        if pidx is not None and kpts_xy is not None and pidx < len(kpts_xy):
            kp = kpts_xy[pidx]

            for lbl, px, py, *_ in last_ppe_labels:
                if lbl == "helmet":
                    if near((px, py), kp[0], TH_HELMET):
                        mem[tid]["helmet"] += 1
                elif lbl == "vest":
                    if (
                        near((px, py), kp[5], TH_VEST) or
                        near((px, py), kp[6], TH_VEST)
                    ):
                        mem[tid]["vest"] += 1
                elif lbl == "gloves":
                    if (
                        near((px, py), kp[9],  TH_GLOVE) or
                        near((px, py), kp[10], TH_GLOVE)
                    ):
                        mem[tid]["gloves"] += 1
                elif lbl == "boots":
                    if (
                        near((px, py), kp[15], TH_BOOT) or
                        near((px, py), kp[16], TH_BOOT)
                    ):
                        mem[tid]["boots"] += 1

        mem[tid]["last_seen"] = frame_count

        compliant = (
            mem[tid]["helmet"] >= MANDATORY_FRAMES and
            mem[tid]["vest"]   >= MANDATORY_FRAMES and
            mem[tid]["gloves"] >= OPTIONAL_FRAMES  and
            mem[tid]["boots"]  >= OPTIONAL_FRAMES
        )
        # CAPTURE — first time this ID becomes compliant
        if compliant and tid not in saved_compliant_tids:
            save_compliant_capture(
                frame,               # raw frame (no annotations)
                tid,
                frame_count,
                (x1, y1, x2, y2),
                last_ppe_labels,
            )
            saved_compliant_tids.add(tid)

        # TRACKING STABILITY
        
        track_seen_frames[tid] += 1
        if tid in track_last_state and track_last_state[tid] != compliant:
            id_switch_count += 1
        track_last_state[tid] = compliant

        if compliant:
            frame_compliant += 1
        else:
            frame_non_compliant += 1

        color      = (0, 255, 0) if compliant else (0, 0, 255)
        label_text = "COMPLIANT"  if compliant else "NOT COMPLIANT"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated, f"ID {tid}: {label_text}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2,
        )

    
    # CLEANUP
    
    for tid in list(mem.keys()):
        if frame_count - mem[tid]["last_seen"] > MEMORY_TTL:
            del mem[tid]

    
    # FPS + PROC TIME
    
    now          = time.time()
    fps          = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
    prev_time    = now
    proc_time_ms = (now - frame_start) * 1000

    
    # ACCUMULATE
    
    frame_detected    = frame_compliant + frame_non_compliant
    non_comp_rate     = (
        frame_non_compliant / frame_detected * 100
        if frame_detected > 0 else 0.0
    )

    total_frames      += 1
    sum_detected      += frame_detected
    sum_compliant     += frame_compliant
    sum_non_compliant += frame_non_compliant
    fps_list.append(fps)
    proc_time_list.append(proc_time_ms)

    
    # CSV ROW
    
    csv_writer.writerow([
        frame_count,
        frame_detected,
        frame_compliant,
        frame_non_compliant,
        round(non_comp_rate, 2),
        round(fps, 2),
        round(proc_time_ms, 2),
        id_switch_count,
    ])

    
    # HUD
    
    cv2.putText(annotated, f"FPS: {fps:.1f}",
                (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(annotated, f"Detected: {frame_detected}",
                (20, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f"Compliant: {frame_compliant}",
                (20, 98),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, f"Non-Compliant: {frame_non_compliant}",
                (20, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(annotated, f"ID Switches: {id_switch_count}",
                (20, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    cv2.putText(annotated, f"Captures saved: {len(saved_compliant_tids)}",
                (20, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 255, 180), 2)

    cv2.imshow("PPE + Pose + Norfair", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# CLEANUP

cap.release()
csv_file.close()
cv2.destroyAllWindows()


# FINAL SUMMARY

avg_fps     = sum(fps_list) / len(fps_list) if fps_list else 0
avg_proc_ms = sum(proc_time_list) / len(proc_time_list) if proc_time_list else 0
avg_non_comp= (sum_non_compliant / sum_detected * 100) if sum_detected > 0 else 0

print("\n========== EVALUATION SUMMARY ==========")
print(f"Total Frames Processed  : {total_frames}")
print(f"Total Workers Detected  : {sum_detected}")
print(f"Compliant Detected      : {sum_compliant}")
print(f"Non-Compliant Detected  : {sum_non_compliant}")
print(f"Avg Non-Compliance Rate : {avg_non_comp:.2f}%")
print(f"Avg FPS                 : {avg_fps:.2f}")
print(f"Avg Processing Time     : {avg_proc_ms:.2f} ms")
print(f"ID Switches (stability) : {id_switch_count}")
print(f"Compliant Captures Saved: {len(saved_compliant_tids)}")
print("-----------------------------------------")
print("Fill manually: Actual Compliant, Actual Non-Compliant,")
print("               Precision, Recall, F1, Accuracy,")
print("               Conclusion, Notes")
print("=========================================")
print(f"Frame CSV  → {csv_path}")
print(f"Captures   → {CAPTURE_DIR}/")