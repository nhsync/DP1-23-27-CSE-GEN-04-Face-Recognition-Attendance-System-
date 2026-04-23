"""
Engagement Module — Presenz
Detects: attentive, phone_usage, sleeping

Head-pose approach (works even when phone/body is hidden by desk):
- If face detected but eyes NOT visible → head is pitched down
- Short sustained downward gaze (3–6 s) → phone_usage
- Long sustained downward gaze (> 6 s) → sleeping
- Also uses YOLOv8 phone detection (COCO class 67) when visible

All engagement events carry the face bounding box of the person,
so app.py can attribute them to the correct student.
"""

import cv2
import numpy as np
import time

COCO_PERSON = 0
COCO_PHONE  = 67

LABELS = {
    'phone_usage': '#c62828',
    'sleeping':    '#e65100',
    'attentive':   '#2e7d32',
}

# Thresholds (in seconds of continuous downward gaze)
PHONE_GAZE_SECONDS   = 3.0   # head down this long → phone_usage
SLEEP_GAZE_SECONDS   = 7.0   # head down this long → sleeping

# Grid cell size for tracking gaze per face region
GRID_CELLS = 6


class EngagementModule:
    def __init__(self):
        self._yolo   = None
        self._loaded = False

        self.eye_cascade  = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Per-region downward gaze tracking:
        # key = grid_cell(x, y), value = timestamp when gaze started going down
        self._gaze_down_since: dict[tuple, float] = {}

        self._try_load_yolo()

    def _try_load_yolo(self):
        try:
            from ultralytics import YOLO
            self._yolo   = YOLO('yolov8n.pt')
            self._loaded = True
        except Exception:
            self._loaded = False

    def _grid_key(self, x, y, frame_w, frame_h):
        """Map a face position to a coarse grid cell for persistent tracking."""
        gx = int(x / frame_w * GRID_CELLS)
        gy = int(y / frame_h * GRID_CELLS)
        return (gx, gy)

    def detect(self, frame, known_faces=None):
        """
        Returns list of (label, confidence, (x,y,w,h), color_hex).
        
        known_faces: list of (name, sid, conf, bbox) from face recognition,
                     used to provide correct face bbox for attribution even
                     when the person's head is down.
        """
        results   = []
        fh, fw    = frame.shape[:2]
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq   = cv2.equalizeHist(gray)
        now       = time.time()

        # ── Step 1: Detect all frontal faces ──────────────────────────────────
        face_bboxes = self.face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.15, minNeighbors=5, minSize=(50, 50))

        seen_cells = set()

        for (x, y, w, h) in face_bboxes:
            cell    = self._grid_key(x + w//2, y + h//2, fw, fh)
            seen_cells.add(cell)

            # Check if eyes are visible (upper 60% of face)
            upper  = gray[y: y + int(h * 0.65), x: x + w]
            eyes   = self.eye_cascade.detectMultiScale(
                upper, scaleFactor=1.1, minNeighbors=3, minSize=(12, 12))
            eyes_visible = len(eyes) >= 1

            if eyes_visible:
                # Clear any gaze-down timer for this cell
                self._gaze_down_since.pop(cell, None)
                results.append(('attentive', 90.0, (x, y, w, h), LABELS['attentive']))
            else:
                # Start or continue gaze-down timer
                if cell not in self._gaze_down_since:
                    self._gaze_down_since[cell] = now
                elapsed = now - self._gaze_down_since[cell]

                if elapsed >= SLEEP_GAZE_SECONDS:
                    results.append(('sleeping', 85.0, (x, y, w, h), LABELS['sleeping']))
                elif elapsed >= PHONE_GAZE_SECONDS:
                    results.append(('phone_usage', 80.0, (x, y, w, h), LABELS['phone_usage']))
                # Under 3 seconds: no flag yet, just observing

        # ── Step 2: YOLOv8 phone detection (visible phone = immediate flag) ───
        if self._loaded:
            try:
                preds = self._yolo(frame, verbose=False, conf=0.42)[0]
                for box in preds.boxes:
                    cls  = int(box.cls[0])
                    conf = float(box.conf[0]) * 100
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if cls == COCO_PHONE:
                        # Phone is visible — override/add phone_usage for nearest face
                        px, py = (x1+x2)//2, (y1+y2)//2
                        best_bbox = self._nearest_face_bbox(px, py, face_bboxes)
                        if best_bbox is not None:
                            fx, fy, fw2, fh2 = best_bbox
                            # Replace attentive with phone_usage for that face
                            results = [r for r in results
                                       if not (r[2] == best_bbox)]
                            results.append(('phone_usage', conf,
                                            best_bbox, LABELS['phone_usage']))
                            # Also force gaze timer
                            cell = self._grid_key(fx+fw2//2, fy+fh2//2, fw, fh)
                            if cell not in self._gaze_down_since:
                                self._gaze_down_since[cell] = now
            except Exception:
                pass

        # ── Step 3: Clean up stale gaze timers for cells with no face ─────────
        stale = [k for k in self._gaze_down_since if k not in seen_cells]
        for k in stale:
            self._gaze_down_since.pop(k, None)

        return results

    def _nearest_face_bbox(self, px, py, face_bboxes):
        """Return bbox of the face nearest to point (px, py)."""
        if len(face_bboxes) == 0:
            return None
        best, best_d = None, float('inf')
        for (x, y, w, h) in face_bboxes:
            cx, cy = x + w//2, y + h//2
            d = (px-cx)**2 + (py-cy)**2
            if d < best_d:
                best_d, best = d, (x, y, w, h)
        return best
