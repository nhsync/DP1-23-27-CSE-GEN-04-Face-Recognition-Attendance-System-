"""
Attendance Manager — CSV logging, snapshots, per-student engagement, reports

Attendance CSV columns:
  Student_ID, Name, Date, Time, Status, Reason, Snapshots

Status values:
  Present  — face confirmed, no violations
  Absent   — not seen OR marked absent due to phone_usage / sleeping

When a snapshot alert fires for a student:
  • If they are marked Present  → downgrade to Absent, write reason + snapshot
  • If they are not yet marked  → create an Absent row immediately
  • Reason column: human-readable string, e.g. "Using phone (detected 14:23:05)"
  • Snapshots column: semicolon-separated list of snapshot filenames
"""

import os, cv2, csv
from datetime import datetime, date

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTENDANCE_DIR = os.path.join(BASE_DIR, 'data', 'attendance')
ENGAGE_DIR     = os.path.join(BASE_DIR, 'data', 'engagement')
SNAP_DIR       = os.path.join(BASE_DIR, 'data', 'snapshots')
STUDENTS_CSV   = os.path.join(BASE_DIR, 'data', 'students.csv')

FIELDS = ['Student_ID', 'Name', 'Date', 'Time', 'Status', 'Reason', 'Snapshots']

for d in [ATTENDANCE_DIR, ENGAGE_DIR, SNAP_DIR]:
    os.makedirs(d, exist_ok=True)

REASON_LABELS = {
    'phone_usage': 'Using phone',
    'sleeping':    'Sleeping in class',
}


class AttendanceManager:

    # ── Attendance ─────────────────────────────────────────────────────────────
    def mark_attendance(self, sid, name, subject):
        """Mark a student Present (if not already in the CSV for today)."""
        today    = date.today().isoformat()
        now      = datetime.now().strftime('%H:%M:%S')
        csv_path = self._att_path(subject, today)

        existing = self._read_rows(csv_path)
        if any(r.get('Student_ID') == sid for r in existing):
            return False   # already recorded

        existing.append({'Student_ID': sid, 'Name': name,
                         'Date': today, 'Time': now,
                         'Status': 'Present', 'Reason': '', 'Snapshots': ''})
        self._write_rows(csv_path, existing)
        return True

    def mark_absent_for_behaviour(self, sid, name, subject, label, snapshot_fname):
        """
        Called whenever a phone_usage or sleeping snapshot is saved.
        Inserts or updates the student's row to Absent with reason + snapshot.
        """
        today    = date.today().isoformat()
        now      = datetime.now().strftime('%H:%M:%S')
        csv_path = self._att_path(subject, today)
        reason   = f"{REASON_LABELS.get(label, label)} (detected {now})"

        rows = self._read_rows(csv_path)
        found = False
        for r in rows:
            if r.get('Student_ID') == sid:
                found = True
                # Always downgrade to Absent regardless of current status
                r['Status'] = 'Absent'
                # Append reason (may have multiple over the session)
                existing_reason = r.get('Reason', '')
                if existing_reason and reason not in existing_reason:
                    r['Reason'] = existing_reason + '; ' + reason
                elif not existing_reason:
                    r['Reason'] = reason
                # Append snapshot filename
                existing_snaps = r.get('Snapshots', '')
                snaps = [s for s in existing_snaps.split(';') if s] if existing_snaps else []
                if snapshot_fname not in snaps:
                    snaps.append(snapshot_fname)
                r['Snapshots'] = ';'.join(snaps)
                break

        if not found:
            # Student was never seen (head always down) — create Absent row
            rows.append({'Student_ID': sid, 'Name': name,
                         'Date': today, 'Time': now,
                         'Status': 'Absent', 'Reason': reason,
                         'Snapshots': snapshot_fname})

        self._write_rows(csv_path, rows)

    def get_attendance_records(self, subject):
        records = []
        prefix  = subject.replace(' ', '_').replace('/', '-') + '_'
        for fname in sorted(os.listdir(ATTENDANCE_DIR)):
            if fname.startswith(prefix) and fname.endswith('.csv'):
                records.extend(self._read_rows(
                    os.path.join(ATTENDANCE_DIR, fname)))
        return records

    def get_all_subjects(self):
        subjects = set()
        for fname in os.listdir(ATTENDANCE_DIR):
            if fname.endswith('.csv'):
                parts = fname.rsplit('_', 1)
                if parts:
                    subjects.add(parts[0].replace('_', ' '))
        return sorted(subjects)

    def get_csv_path(self, subject):
        today = date.today().isoformat()
        p     = self._att_path(subject, today)
        if os.path.exists(p):
            return p
        prefix = subject.replace(' ', '_').replace('/', '-') + '_'
        files  = sorted(f for f in os.listdir(ATTENDANCE_DIR)
                        if f.startswith(prefix))
        return os.path.join(ATTENDANCE_DIR, files[-1]) if files else None

    def get_session_summary(self, subject):
        today   = date.today().isoformat()
        records = self.get_attendance_records(subject)
        today_r = [r for r in records if r.get('Date') == today]
        present = [r for r in today_r if r.get('Status') == 'Present']
        return {'subject': subject, 'date': today,
                'total_present': len(present), 'students': today_r}

    # ── Snapshots ──────────────────────────────────────────────────────────────
    def save_snapshot(self, frame, sid, name, label, subject):
        """
        Save JPEG snapshot, log engagement event, and immediately update
        the attendance record to Absent with reason.
        """
        ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_s  = subject.replace(' ', '_')
        safe_id = sid.replace('/', '-')
        fname   = f"{safe_s}_{safe_id}_{label}_{ts}.jpg"
        path    = os.path.join(SNAP_DIR, fname)
        cv2.imwrite(path, frame)

        # Log engagement event
        log_path = os.path.join(ENGAGE_DIR,
                                f"{safe_s}_{date.today().isoformat()}.csv")
        eng_rows = self._read_rows(log_path, default_fields=[
            'Student_ID', 'Name', 'Subject', 'Label', 'Timestamp', 'Snapshot'])
        eng_rows.append({'Student_ID': sid, 'Name': name,
                         'Subject': subject, 'Label': label,
                         'Timestamp': datetime.now().strftime('%H:%M:%S'),
                         'Snapshot': fname})
        self._write_rows(log_path, eng_rows,
                         fields=['Student_ID', 'Name', 'Subject',
                                 'Label', 'Timestamp', 'Snapshot'])

        # Update attendance to Absent
        self.mark_absent_for_behaviour(sid, name, subject, label, fname)

        return fname

    # ── Per-student engagement ─────────────────────────────────────────────────
    def get_engagement_by_student(self, subject):
        result = {}
        safe_s = subject.replace(' ', '_').replace('/', '-')
        prefix = safe_s + '_'
        for fname in sorted(os.listdir(ENGAGE_DIR)):
            if not (fname.startswith(prefix) and fname.endswith('.csv')):
                continue
            for row in self._read_rows(os.path.join(ENGAGE_DIR, fname)):
                sid  = row.get('Student_ID', 'unknown')
                name = row.get('Name', 'Unidentified')
                if sid not in result:
                    result[sid] = {'name': name, 'events': [], 'counts': {}}
                label = row.get('Label', '')
                result[sid]['events'].append({
                    'label':     label,
                    'timestamp': row.get('Timestamp', ''),
                    'snapshot':  row.get('Snapshot', '')
                })
                result[sid]['counts'][label] = \
                    result[sid]['counts'].get(label, 0) + 1
        return result

    # ── Report summary ─────────────────────────────────────────────────────────
    def get_report_summary(self, subject):
        """
        Returns per-student rows with status, reason, and snapshots list.
        Students not in the attendance CSV are also included as Absent (not seen).
        """
        all_students = self._load_all_students()
        records      = self.get_attendance_records(subject)

        # Build lookup by sid → latest record
        rec_by_sid = {}
        for r in records:
            rec_by_sid[r['Student_ID']] = r

        rows = []
        present_count = 0
        absent_count  = 0

        for s in all_students:
            rec = rec_by_sid.get(s['id'])
            if rec:
                status    = rec.get('Status', 'Present')
                reason    = rec.get('Reason', '')
                snap_str  = rec.get('Snapshots', '')
                snapshots = [sn for sn in snap_str.split(';') if sn] if snap_str else []
            else:
                status    = 'Absent'
                reason    = 'Not detected during session'
                snapshots = []

            if status == 'Present':
                present_count += 1
            else:
                absent_count += 1

            rows.append({
                'id':        s['id'],
                'name':      s['name'],
                'status':    status,
                'reason':    reason,
                'snapshots': snapshots,
            })

        return {
            'subject':        subject,
            'total_students': len(all_students),
            'total_present':  present_count,
            'total_absent':   absent_count,
            'rows':           rows,
        }

    def get_global_stats(self):
        students   = self._load_all_students()
        total_sess = 0
        total_rec  = 0
        subjects   = set()
        for fname in os.listdir(ATTENDANCE_DIR):
            if fname.endswith('.csv'):
                total_sess += 1
                parts = fname.rsplit('_', 1)
                if parts:
                    subjects.add(parts[0])
                rows = self._read_rows(os.path.join(ATTENDANCE_DIR, fname))
                total_rec += sum(1 for r in rows if r.get('Status') == 'Present')
        return {
            'total_students': len(students),
            'total_sessions': total_sess,
            'total_present':  total_rec,
            'total_subjects': len(subjects),
        }

    # ── CSV helpers ────────────────────────────────────────────────────────────
    def _att_path(self, subject, date_str):
        safe = subject.replace(' ', '_').replace('/', '-')
        return os.path.join(ATTENDANCE_DIR, f"{safe}_{date_str}.csv")

    def _read_rows(self, path, default_fields=None):
        if not os.path.exists(path):
            return []
        with open(path, newline='') as f:
            return [dict(r) for r in csv.DictReader(f)]

    def _write_rows(self, path, rows, fields=None):
        if fields is None:
            fields = FIELDS
        # Ensure all rows have all fields (backfill missing keys)
        for r in rows:
            for f in fields:
                r.setdefault(f, '')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)

    def _load_all_students(self):
        if not os.path.exists(STUDENTS_CSV):
            return []
        with open(STUDENTS_CSV, newline='') as f:
            return list(csv.DictReader(f))
