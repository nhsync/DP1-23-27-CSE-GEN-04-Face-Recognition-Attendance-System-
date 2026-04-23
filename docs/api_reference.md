# API Reference

## Presenz — Module Reference
**Project ID:** DP1-23-27-CSE-GEN-04

All source modules reside under `src/modules/`. The Flask entry point is `src/app.py`.

---

## `face_recognition_module.py`

### `FaceRecognitionModule`

Core class managing student enrollment, LBPH model training, real-time recognition, and student record management.

---

### `capture_student_images(student_id, name, socketio=None)`
Captures 60 grayscale enrollment images for a new student via webcam.

**Parameters:**
- `student_id` (str) — Unique student identifier
- `name` (str) — Student's full name
- `socketio` (SocketIO, optional) — SocketIO instance for emitting live progress events to the browser

**Behaviour:**
- Opens webcam, captures frames at 960×540, detects face via Haar Cascade, crops and resizes to 200×200 px
- Saves images to `data/student_images/{student_id}_{name}/`
- Appends student record to the students CSV
- Invalidates any previously trained model

**Returns:**
- `dict` — `{success: bool, message: str}`

---

### `train_model()`
Trains the LBPH recogniser on all enrolled student images.

**Behaviour:**
- Loads all images from `data/student_images/`
- Applies Haar Cascade face detection and histogram equalisation per image
- Trains `cv2.face.LBPHFaceRecognizer_create()`
- Serialises model to `data/training_labels/lbph_model.yml`
- Serialises label-to-student mapping to `data/training_labels/label_map.pkl`
- Training time: approximately 3–5 seconds for 10 students (600 images)

**Returns:**
- `dict` — `{success: bool, message: str, student_count: int}`

---

### `recognize_faces(frame)`
Runs LBPH recognition on all faces detected in a video frame.

**Parameters:**
- `frame` (numpy.ndarray) — BGR image frame from `cv2.VideoCapture`

**Behaviour:**
- Converts to grayscale, applies histogram equalisation
- Detects faces via Haar Cascade (`scaleFactor=1.1`, `minNeighbors=8`, `minSize=(80,80)`)
- For each face ROI (resized to 200×200), calls `recognizer.predict()` and maps result to confidence: `max(0, 100 − distance × 0.7)`
- Rejects predictions with confidence below 52

**Returns:**
- `list` of `(name, student_id, confidence, (x, y, w, h))` tuples; unknown faces return `('Unknown', '', 0.0, bbox)`

---

### `delete_student(student_id)`
Removes a student's images and record, and invalidates the trained model.

**Parameters:**
- `student_id` (str) — Student identifier to delete

**Returns:**
- `dict` — `{success: bool, message: str}`

---

### `update_student_info(student_id, new_name)`
Renames a student and updates their image folder accordingly. Invalidates the trained model.

**Parameters:**
- `student_id` (str) — Student identifier
- `new_name` (str) — New display name

**Returns:**
- `dict` — `{success: bool, message: str}`

---

### `get_registered_students()`
Returns the list of all currently enrolled students.

**Returns:**
- `list` of `dict` — Each entry contains `id`, `name`, and enrollment metadata

---

## `engagement_module.py`

### `EngagementModule`

Detects phone usage and sleeping behaviour via three parallel channels: Haar Eye Cascade heuristic, face disappearance tracking, and YOLOv8n phone detection.

---

### `detect(frame, confirmed_ids=None)`
Runs all three engagement detection channels on a video frame.

**Parameters:**
- `frame` (numpy.ndarray) — BGR image frame
- `confirmed_ids` (dict, optional) — Mapping of grid cell keys to `{name, sid, bbox, ts}` for students whose identity has been confirmed by the face recognition module. Used for face-disappearance sleeping detection.

**Detection Channels:**

**Channel 1 — Eye Visibility Heuristic:**
Haar Eye Cascade (`haarcascade_eye.xml`, `minNeighbors=6`) on the upper 55% of each face ROI.
- Eyes absent ≥ 6 s → `phone_usage` flag
- Eyes absent ≥ 18 s → `sleeping` flag (with 30 s cooldown)
- Any eye detection resets the absence timer

**Channel 2 — Face Disappearance:**
If a confirmed student's grid cell has no detected face for ≥ 20 continuous seconds, a `sleeping` alert is raised using the student's last known bounding box coordinates.

**Channel 3 — YOLOv8n Phone Detection:**
Full-frame inference with YOLOv8n (COCO pre-trained weights). Phone detections (COCO class 67, confidence ≥ 0.45) are attributed to the nearest confirmed face via bounding box centre proximity. A matched phone detection overrides the eye heuristic result for that student.

**Cooldown:** All alerts enforce a 15-second per-student cooldown to prevent duplicate demotions from a single incident.

**Returns:**
- `list` of `(label, confidence, (x, y, w, h), color_hex, student_id_or_None)` tuples
- `label` is one of: `'phone_usage'`, `'sleeping'`

---

## `attendance_manager.py`

### `mark_attendance(student_id, name, subject)`
Logs a student as Present in the daily subject CSV file.

**Parameters:**
- `student_id` (str) — Student identifier
- `name` (str) — Student's display name
- `subject` (str) — Subject/session name

**Behaviour:**
- Appends a row to `data/attendance/{subject}_{YYYY-MM-DD}.csv` with columns: `Student_ID`, `Name`, `Date`, `Time`, `Status`, `Reason`, `Snapshots`
- Silently ignores duplicate calls for the same student within the same session

**Returns:**
- `bool` — `True` if newly marked, `False` if already marked this session

---

### `update_engagement(student_id, subject, alert_type, snapshot_filename)`
Updates an existing attendance record to Absent and logs the engagement evidence.

**Parameters:**
- `student_id` (str) — Student identifier
- `subject` (str) — Subject/session name
- `alert_type` (str) — `'Phone Usage'` or `'Sleeping'`
- `snapshot_filename` (str) — Filename of the saved JPEG evidence snapshot in `data/snapshots/`

**Behaviour:**
- Finds the student's row in the current session CSV
- Sets `Status` to `Absent`, `Reason` to `alert_type`
- Appends `snapshot_filename` to the `Snapshots` column (semicolon-separated if multiple events)

---

### `get_report(subject, date)`
Returns attendance data for a given subject and date.

**Parameters:**
- `subject` (str) — Subject name
- `date` (str) — Date string in `YYYY-MM-DD` format

**Returns:**
- `pandas.DataFrame` — Full attendance table for that subject/date, or an empty DataFrame if no records exist

---

## `app.py` — Flask Routes & Socket.IO Events

### HTTP Routes

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Dashboard view (live session) |
| GET | `/students` | Student management view |
| GET | `/reports` | Reports view |
| POST | `/api/register` | Register a new student (triggers `capture_student_images`) |
| POST | `/api/train` | Trigger model re-training |
| POST | `/api/session/start` | Start an attendance session for a given subject |
| POST | `/api/session/stop` | Stop the active session |
| GET | `/api/report/download` | Download CSV for a given subject and date |
| DELETE | `/api/student/<id>` | Delete a student record |
| PUT | `/api/student/<id>` | Rename a student |

### Socket.IO Events (Server → Client)

| Event | Payload | Description |
|-------|---------|-------------|
| `frame` | `{image: base64_jpeg_string}` | Annotated live video frame (~30 FPS, JPEG quality 60) |
| `attendance_marked` | `{student_id, name, time, subject}` | Fired when a student is newly marked Present |
| `engagement_alert` | `{student_id, name, alert_type, snapshot, subject}` | Fired when a misconduct event is detected and attendance is downgraded |

---

## Data Storage Reference

| Path | Contents |
|------|----------|
| `data/student_images/{id}_{name}/` | 60 grayscale 200×200 px JPEG enrollment images per student |
| `data/training_labels/lbph_model.yml` | Serialised LBPH recogniser |
| `data/training_labels/label_map.pkl` | Pickle dict mapping LBPH labels to student ID and name |
| `data/attendance/{subject}_{YYYY-MM-DD}.csv` | Daily per-subject attendance records |
| `data/snapshots/{id}_{type}_{subject}_{ts}.jpg` | Timestamped JPEG evidence of engagement events |
