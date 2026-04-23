"""
Face Recognition Module — Presenz
- Haar Cascade detection + LBPH recognition
- Temporal vote buffer to eliminate flicker and false positives
- Streams live video frames during capture via Socket.IO
"""

import os, cv2, pickle, csv, base64, time
import numpy as np
from datetime import datetime

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR     = os.path.join(BASE_DIR, 'data', 'student_images')
LABELS_DIR  = os.path.join(BASE_DIR, 'data', 'training_labels')
DETAILS_CSV = os.path.join(BASE_DIR, 'data', 'students.csv')
HAAR_PATH   = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

for d in [IMG_DIR, LABELS_DIR, os.path.join(BASE_DIR, 'data', 'encodings')]:
    os.makedirs(d, exist_ok=True)

# ── Tuning constants ───────────────────────────────────────────────────────────
# LBPH distance: lower = more similar. Range is 0-∞, typical match < 80.
# At dist=0 it's a perfect match; at dist=100+ it's basically wrong.
# We map: confidence_pct = max(0, 100 - dist * 0.7)
# Then require confidence > CONF_THRESHOLD to accept as known face.
CONF_THRESHOLD = 62       # Raised significantly from old 45 — reduces false positives
VOTE_WINDOW    = 6        # Must see same identity in 6 out of VOTE_REQUIRED frames
VOTE_REQUIRED  = 4        # At least 4 consistent votes before confirming identity
MIN_FACE_SIZE  = (70, 70) # Ignore tiny faces (likely false detections far away)


class FaceRecognitionModule:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(HAAR_PATH)
        self.eye_cascade  = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.recognizer   = cv2.face.LBPHFaceRecognizer_create()
        self.label_map    = {}
        self.trained      = False

        # Per-face-region vote history: region_key -> deque of (name, sid, conf)
        self._vote_history = {}   # managed by recognize_faces caller (app.py)

        self._try_load()

    def _try_load(self):
        model_path = os.path.join(LABELS_DIR, 'lbph_model.yml')
        map_path   = os.path.join(LABELS_DIR, 'label_map.pkl')
        if os.path.exists(model_path) and os.path.exists(map_path):
            try:
                self.recognizer.read(model_path)
                with open(map_path, 'rb') as f:
                    self.label_map = pickle.load(f)
                self.trained = True
            except Exception:
                self.trained = False

    # ── Image Capture — streams live video during registration ─────────────────
    def capture_student_images(self, student_id: str, name: str,
                               socketio=None, n_images=80):
        """
        Capture face images for a new student.
        Streams live annotated video frames to browser via Socket.IO.
        Captures varied samples: normal, slight left, slight right, slight up/down.
        """
        folder = os.path.join(IMG_DIR, f"{student_id}_{name}")
        os.makedirs(folder, exist_ok=True)

        cap   = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        count       = 0
        no_face_streak = 0

        while count < n_images:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # equalizeHist improves detection under varying lighting
            gray_eq = cv2.equalizeHist(gray)

            faces = self.face_cascade.detectMultiScale(
                gray_eq, scaleFactor=1.2, minNeighbors=6,
                minSize=MIN_FACE_SIZE)

            captured_this_frame = False
            for (x, y, w, h) in faces:
                # Draw box on display
                cv2.rectangle(display, (x, y), (x+w, y+h), (34, 139, 34), 2)
                label_txt = f"Capturing {name}: {count}/{n_images}"
                cv2.rectangle(display, (x, y-28), (x+w, y), (34, 139, 34), -1)
                cv2.putText(display, label_txt, (x+4, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if not captured_this_frame:
                    face_gray = gray[y:y+h, x:x+w]
                    face_gray = cv2.equalizeHist(face_gray)
                    face_resized = cv2.resize(face_gray, (200, 200))
                    path = os.path.join(folder, f"{count}.jpg")
                    cv2.imwrite(path, face_resized)
                    count += 1
                    captured_this_frame = True
                    no_face_streak = 0
                    if socketio:
                        socketio.emit('capture_progress', {
                            'count': count, 'total': n_images, 'name': name
                        })
                break  # one face per frame is enough

            if not captured_this_frame:
                no_face_streak += 1
                cv2.putText(display, "Position your face in the camera",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 80, 220), 2)

            # Draw progress bar overlay
            bar_w = int((count / n_images) * frame.shape[1])
            cv2.rectangle(display, (0, frame.shape[0]-8),
                          (bar_w, frame.shape[0]), (34, 139, 34), -1)

            # Stream frame to register page
            _, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 72])
            b64 = base64.b64encode(buf).decode('utf-8')
            if socketio:
                socketio.emit('register_frame', {'frame': b64})

            time.sleep(0.05)

        cap.release()
        if socketio:
            socketio.emit('register_frame', {'frame': ''})  # signal done

        self._save_student_details(student_id, name)
        return {'success': True, 'count': count,
                'message': f'Captured {count} images for {name}'}

    def _save_student_details(self, sid, name):
        rows = self._load_students()
        if any(r['id'] == sid for r in rows):
            return
        is_new = not os.path.exists(DETAILS_CSV) or os.path.getsize(DETAILS_CSV) == 0
        with open(DETAILS_CSV, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['id', 'name', 'registered_at'])
            if is_new:
                w.writeheader()
            w.writerow({'id': sid, 'name': name,
                        'registered_at': datetime.now().isoformat()})

    def _load_students(self):
        if not os.path.exists(DETAILS_CSV):
            return []
        with open(DETAILS_CSV, newline='') as f:
            return list(csv.DictReader(f))

    # ── Training ───────────────────────────────────────────────────────────────
    def train_model(self, socketio=None):
        faces, labels = [], []
        label_map = {}
        label_idx = 0

        for folder_name in sorted(os.listdir(IMG_DIR)):
            folder_path = os.path.join(IMG_DIR, folder_name)
            if not os.path.isdir(folder_path):
                continue
            parts = folder_name.split('_', 1)
            sid   = parts[0]
            name  = parts[1] if len(parts) > 1 else folder_name
            label_map[label_idx] = {'id': sid, 'name': name}

            for img_file in sorted(os.listdir(folder_path)):
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.equalizeHist(cv2.resize(img, (200, 200)))
                faces.append(img)
                labels.append(label_idx)

            label_idx += 1
            if socketio:
                socketio.emit('train_progress', {'student': name})

        if not faces:
            return {'success': False, 'message': 'No training images found'}

        self.recognizer.train(faces, np.array(labels))
        model_path = os.path.join(LABELS_DIR, 'lbph_model.yml')
        map_path   = os.path.join(LABELS_DIR, 'label_map.pkl')
        self.recognizer.save(model_path)
        with open(map_path, 'wb') as f:
            pickle.dump(label_map, f)
        self.label_map = label_map
        self.trained   = True
        return {'success': True,
                'message': f'Trained on {len(faces)} images, {label_idx} students'}

    # ── Recognition ───────────────────────────────────────────────────────────
    def recognize_faces(self, frame):
        """
        Returns list of (name, student_id, confidence_pct, (x,y,w,h)).
        Uses histogram equalisation + tighter threshold for better accuracy.
        Does NOT apply temporal vote buffer here — that's done in app.py
        so it persists across frames.
        """
        if not self.trained:
            return []

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.15, minNeighbors=6,
            minSize=MIN_FACE_SIZE)

        results = []
        for (x, y, w, h) in faces:
            roi    = cv2.equalizeHist(cv2.resize(gray[y:y+h, x:x+w], (200, 200)))
            try:
                label, dist = self.recognizer.predict(roi)
                # Better confidence mapping: dist < 50 → ~96%, dist=100 → ~70%, dist=150 → ~55%
                confidence = max(0.0, 100.0 - dist * 0.7)
                if confidence >= CONF_THRESHOLD and label in self.label_map:
                    info = self.label_map[label]
                    results.append((info['name'], info['id'], confidence, (x, y, w, h)))
                else:
                    results.append(('Unknown', '', 0.0, (x, y, w, h)))
            except Exception:
                results.append(('Unknown', '', 0.0, (x, y, w, h)))
        return results

    def check_eyes_visible(self, frame, face_bbox):
        """
        Returns True if eyes are detectable inside the face region.
        If False → head is likely pitched downward.
        """
        x, y, w, h = face_bbox
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Only look in upper 60% of face for eyes
        upper_face = gray[y: y + int(h * 0.6), x: x + w]
        eyes = self.eye_cascade.detectMultiScale(
            upper_face, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15))
        return len(eyes) >= 1

    def get_registered_students(self):
        return self._load_students()
