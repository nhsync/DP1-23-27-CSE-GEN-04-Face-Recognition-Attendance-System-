"""
Presenz — Face Recognition Attendance System
Team: Nitin H R, PS Lakshwin Ruup, Royce Biju Thomas, Gokul S
Mentor: Dr. Ganga Holi | 6CS1991 Design Project-2
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO
import cv2, os, base64, threading, time
from datetime import datetime
from collections import deque
import numpy as np
from modules.face_recognition_module import FaceRecognitionModule, VOTE_WINDOW, VOTE_REQUIRED
from modules.engagement_module import EngagementModule
from modules.attendance_manager import AttendanceManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'presenz_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

face_module    = FaceRecognitionModule()
engage_module  = EngagementModule()
attend_manager = AttendanceManager()

camera_active   = False
camera_thread   = None
current_subject = ""
session_stats   = {}

# ── Pages ──────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/records')
def records():
    return render_template('records.html')

@app.route('/reports')
def reports():
    return render_template('reports.html')

# ── Registration ───────────────────────────────────────────────────────────────
@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    sid  = data.get('student_id', '').strip()
    name = data.get('name', '').strip()
    if not sid or not name:
        return jsonify({'success': False, 'message': 'ID and name required'})
    return jsonify(face_module.capture_student_images(sid, name, socketio))

@app.route('/api/train', methods=['POST'])
def api_train():
    return jsonify(face_module.train_model(socketio))

# ── Session ────────────────────────────────────────────────────────────────────
@app.route('/api/start_session', methods=['POST'])
def start_session():
    global camera_active, camera_thread, current_subject, session_stats
    data = request.json
    current_subject = data.get('subject', '').strip()
    if not current_subject:
        return jsonify({'success': False, 'message': 'Subject name required'})
    if camera_active:
        return jsonify({'success': False, 'message': 'Session already running'})
    camera_active = True
    session_stats = {}
    camera_thread = threading.Thread(target=run_session, daemon=True)
    camera_thread.start()
    return jsonify({'success': True})

@app.route('/api/stop_session', methods=['POST'])
def stop_session():
    global camera_active
    camera_active = False
    time.sleep(1.5)
    summary = attend_manager.get_session_summary(current_subject)
    return jsonify({'success': True, 'summary': summary})

# ── Data APIs ──────────────────────────────────────────────────────────────────
@app.route('/api/subjects')
def get_subjects():
    return jsonify({'subjects': attend_manager.get_all_subjects()})

@app.route('/api/attendance/<subject>')
def get_attendance(subject):
    return jsonify({'records': attend_manager.get_attendance_records(subject)})

@app.route('/api/students')
def get_students():
    return jsonify({'students': face_module.get_registered_students()})

@app.route('/api/stats')
def get_stats():
    return jsonify(attend_manager.get_global_stats())

@app.route('/api/export/<subject>')
def export_csv(subject):
    path = attend_manager.get_csv_path(subject)
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({'error': 'Not found'}), 404

@app.route('/api/engagement/<subject>')
def get_engagement(subject):
    return jsonify(attend_manager.get_engagement_by_student(subject))

@app.route('/api/snapshots/<path:filename>')
def serve_snapshot(filename):
    snap_dir = os.path.join(os.path.dirname(__file__), 'data', 'snapshots')
    full = os.path.join(snap_dir, filename)
    if os.path.exists(full):
        return send_file(full, mimetype='image/jpeg')
    return '', 404

@app.route('/api/report_summary/<subject>')
def report_summary(subject):
    return jsonify(attend_manager.get_report_summary(subject))

# ── Camera session loop ────────────────────────────────────────────────────────
def run_session():
    """
    Main attendance session loop.

    Key design decisions:
    1. VOTE BUFFER: Each face region accumulates predictions across frames.
       A name is only DISPLAYED and MARKED when the same identity appears
       VOTE_REQUIRED times within VOTE_WINDOW consecutive frames.
       This eliminates flickering and false positives from a single bad frame.

    2. DISPLAY: While face is detected but not yet confirmed, show a plain
       white rectangle with "Identifying…" — never show raw ID/name until confirmed.

    3. LAST-KNOWN POSITION TRACKING: When a face disappears from its grid region
       (head down / occluded), we remember the last confirmed identity at that
       location for engagement attribution.

    4. ENGAGEMENT: Eye-visibility heuristic + YOLOv8 phone detection. Duration
       thresholds distinguish phone use vs sleeping. Attribution uses nearest
       confirmed face or falls back to last-known identity in that region.
    """
    global camera_active, session_stats

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    fh_cap = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fw_cap = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    marked_today   = set()   # sids that have been marked this session
    frame_count    = 0
    snap_cooldown  = {}
    SNAP_INTERVAL  = 12      # seconds between snapshots per student-label

    # Vote buffer: region_key -> deque of (name, sid, conf)
    # region_key is a coarse grid cell so nearby faces share the buffer
    vote_buffers: dict = {}   # region_key -> deque(maxlen=VOTE_WINDOW)

    # Confirmed identities: region_key -> {name, sid, conf, bbox, ts}
    # Persists even when face briefly disappears (for engagement attribution)
    confirmed_ids: dict = {}

    # Last known face bboxes — for drawing in current frame
    last_raw_faces = []

    def _region_key(x, y):
        GRID = 6
        return (int(x / fw_cap * GRID), int(y / fh_cap * GRID))

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display = frame.copy()

        # ── Face recognition every 3 frames ──────────────────────────────────
        if frame_count % 3 == 0:
            raw_faces = face_module.recognize_faces(frame)
            last_raw_faces = raw_faces

            # Update vote buffers
            for (name, sid, conf, bbox) in raw_faces:
                x, y, w, h = bbox
                key = _region_key(x + w//2, y + h//2)
                if key not in vote_buffers:
                    vote_buffers[key] = deque(maxlen=VOTE_WINDOW)
                vote_buffers[key].append((name, sid, conf))

            # Evaluate votes → confirm identities
            newly_confirmed = set()
            for key, buf in vote_buffers.items():
                if len(buf) < VOTE_REQUIRED:
                    continue
                # Count how many of last VOTE_REQUIRED are the same identity
                recent = list(buf)[-VOTE_REQUIRED:]
                names  = [r[0] for r in recent]
                if len(set(names)) == 1 and names[0] != 'Unknown':
                    # All recent predictions agree → confirmed
                    best = max(recent, key=lambda r: r[2])
                    name, sid, conf = best
                    # Find matching bbox from current raw faces
                    bbox = None
                    for (rn, rs, rc, rb) in raw_faces:
                        rx, ry, rw, rh = rb
                        if _region_key(rx+rw//2, ry+rh//2) == key:
                            bbox = rb
                            break
                    if bbox is None and key in confirmed_ids:
                        bbox = confirmed_ids[key]['bbox']  # use last known
                    confirmed_ids[key] = {
                        'name': name, 'sid': sid, 'conf': conf,
                        'bbox': bbox, 'ts': time.time()
                    }
                    newly_confirmed.add(key)

            # Expire confirmed IDs older than 8 seconds with no face signal
            active_keys = set()
            for (rn, rs, rc, (rx,ry,rw,rh)) in last_raw_faces:
                active_keys.add(_region_key(rx+rw//2, ry+rh//2))
            now = time.time()
            expired = [k for k, v in confirmed_ids.items()
                       if k not in active_keys and now - v['ts'] > 8]
            for k in expired:
                confirmed_ids.pop(k, None)
                vote_buffers.pop(k, None)

            # Mark attendance for newly confirmed faces
            for key in newly_confirmed:
                info = confirmed_ids[key]
                sid  = info['sid']
                if sid and sid not in marked_today:
                    marked_today.add(sid)
                    attend_manager.mark_attendance(sid, info['name'], current_subject)
                    socketio.emit('attendance_marked', {
                        'name': info['name'], 'id': sid,
                        'time': datetime.now().strftime('%H:%M:%S')
                    })

        # ── Draw overlays ─────────────────────────────────────────────────────
        # Draw confirmed identities with solid green boxes + name
        confirmed_bboxes = set()
        for key, info in confirmed_ids.items():
            bbox = info.get('bbox')
            if not bbox:
                continue
            x, y, w, h = bbox
            confirmed_bboxes.add(bbox)
            color = (34, 139, 34)
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
            lbl = f"{info['name']} ({info['conf']:.0f}%)"
            cv2.rectangle(display, (x, y-26), (x+w, y), color, -1)
            cv2.putText(display, lbl, (x+5, y-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw unconfirmed faces (no name — just "Identifying...")
        for (name, sid, conf, (x, y, w, h)) in last_raw_faces:
            key = _region_key(x + w//2, y + h//2)
            if key not in confirmed_ids:
                color = (180, 130, 30)
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 1)
                cv2.rectangle(display, (x, y-22), (x+w, y), color, -1)
                cv2.putText(display, "Identifying...", (x+4, y-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # ── Engagement detection every 5 frames ──────────────────────────────
        if frame_count % 5 == 0:
            eng_results = engage_module.detect(frame)

            for (eng_label, eng_conf, eng_bbox, color_hex) in eng_results:
                ex, ey, ew, eh = eng_bbox
                h6  = color_hex.lstrip('#')
                bgr = tuple(int(h6[i:i+2], 16) for i in (4, 2, 0))

                # Only draw alert labels on display (don't litter screen with attentive boxes)
                if eng_label != 'attentive':
                    cv2.rectangle(display, (ex, ey), (ex+ew, ey+eh), bgr, 2)
                    cv2.putText(display, eng_label.replace('_', ' '), (ex, ey-7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)

                # Attribute to student — nearest confirmed face first
                linked_sid  = 'unknown'
                linked_name = 'Unidentified'
                ec = (ex + ew//2, ey + eh//2)

                # Try confirmed identities first
                best_d = float('inf')
                for key, info in confirmed_ids.items():
                    bbox = info.get('bbox')
                    if not bbox:
                        continue
                    fx, fy, fw2, fh2 = bbox
                    fc = (fx + fw2//2, fy + fh2//2)
                    d  = (ec[0]-fc[0])**2 + (ec[1]-fc[1])**2
                    if d < best_d:
                        best_d      = d
                        linked_sid  = info['sid']
                        linked_name = info['name']

                # Fallback: raw faces
                if linked_sid == 'unknown' and last_raw_faces:
                    for (fn, fs, fc_, (fx, fy, fw2, fh2)) in last_raw_faces:
                        if fn == 'Unknown':
                            continue
                        fc = (fx+fw2//2, fy+fh2//2)
                        d  = (ec[0]-fc[0])**2 + (ec[1]-fc[1])**2
                        if d < best_d:
                            best_d = d
                            linked_sid  = fs
                            linked_name = fn

                # Tally per student
                if linked_sid not in session_stats:
                    session_stats[linked_sid] = {'name': linked_name}
                session_stats[linked_sid][eng_label] = \
                    session_stats[linked_sid].get(eng_label, 0) + 1

                # Snapshot on alert events with cooldown
                if eng_label in ('phone_usage', 'sleeping'):
                    snap_key = f"{linked_sid}_{eng_label}"
                    now_ts   = time.time()
                    if now_ts - snap_cooldown.get(snap_key, 0) >= SNAP_INTERVAL:
                        snap_cooldown[snap_key] = now_ts
                        snap_frame = frame.copy()
                        ts_str = datetime.now().strftime('%H:%M:%S')
                        cv2.putText(snap_frame, ts_str,
                                    (10, snap_frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        attend_manager.save_snapshot(
                            snap_frame, linked_sid, linked_name,
                            eng_label, current_subject
                        )

            if eng_results:
                socketio.emit('engagement_update', {'stats': session_stats})

        # ── Stream frame ──────────────────────────────────────────────────────
        _, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 74])
        socketio.emit('video_frame',
                      {'frame': base64.b64encode(buf).decode('utf-8')})
        time.sleep(0.04)

    cap.release()
    socketio.emit('session_ended', {'stats': session_stats})


if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000, use_reloader=False)
