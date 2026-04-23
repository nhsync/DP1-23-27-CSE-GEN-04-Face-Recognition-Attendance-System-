# User Manual

## Presenz: A Face Recognition and Engagement Detection Based Attendance System

---

## For Instructors / Administrators

### Registering a New Student
1. Open the Presenz dashboard in your browser at `http://127.0.0.1:5000`
2. Navigate to the **Students** tab
3. Click **Register New Student** and enter the student's name and ID
4. The student should sit in front of the webcam and look directly at the camera
5. The system will automatically capture 60 images — a live progress indicator will display on screen
6. Registration is complete when you see the confirmation message. The model will need to be re-trained before the new student can be recognised (see below)

### Training the Recognition Model
After registering one or more new students, or after any student record change (rename, delete, re-enroll):
1. Go to the **Students** tab
2. Click **Train Model**
3. Wait for the confirmation message ("Model trained successfully") — this typically takes 3–5 seconds for 10 students

### Starting an Attendance Session
1. From the **Dashboard** tab, enter the subject name in the Session field
2. Click **Start Session**
3. The live annotated camera feed will appear. Students should sit facing the webcam
4. As faces are recognised, names will appear in the attendance table on the right side of the dashboard
5. Engagement alerts (phone usage, sleeping) will appear in the alert log below the attendance table

### Stopping a Session
1. Click **Stop Session** on the Dashboard
2. The session is closed and attendance is saved to the CSV for that subject

### Viewing and Downloading Reports
1. Navigate to the **Reports** tab
2. Select a subject and a date
3. The attendance table will display each student's status (Present / Absent), reason (if absent), and evidence snapshot thumbnails
4. Click **Download CSV** to export the full report for that subject/date combination

### Managing Students
From the **Students** tab you can:
- **Rename** a student — updates their name across all future records (re-trains model automatically)
- **Re-enroll** a student — re-captures 60 fresh images (useful if recognition accuracy drops for a particular student)
- **Delete** a student — removes all their images and records from the system

---

## For Students

1. Sit in front of the webcam when the instructor starts the session
2. Look directly at the camera — keep your face well-lit and unobstructed
3. Once your name appears in the attendance table on the instructor's screen, your attendance has been recorded automatically. No action is needed from you
4. Stay attentive during the session — phone usage and sleeping are detected automatically and will downgrade your attendance status from Present to Absent

---

## Engagement Monitoring Explained

Presenz continuously monitors each recognised student's behaviour using two methods:

**Phone Usage Detection:**
- If your eyes are not visible for more than 6 seconds, the system flags a potential phone use
- If YOLOv8n directly detects a smartphone near your face with ≥ 45% confidence, this overrides the eye heuristic and is treated as a confirmed phone usage event

**Sleeping Detection:**
- If your eyes are continuously absent for more than 18 seconds, the system flags sleeping
- If your face disappears entirely from your seat position for more than 20 seconds, sleeping is also flagged (for cases where the head is fully rested on the desk)

When either event is detected, your attendance status changes from Present to Absent and a timestamped snapshot is saved as evidence.

---

## Notes and Tips

- Ensure good, even lighting — avoid sitting with a bright window behind you
- Remove sunglasses or caps that shade the face if recognition fails
- The recognition system requires 4 consecutive frames to confirm your identity — stay still for a moment when first sitting down
- Each student can only be marked Present once per session regardless of how many times the face is detected
- Recognition accuracy may reduce at distances greater than 3 metres from the webcam — aim to sit within 1.5–2.5 m for best results
- If a student is not being recognised, use the **Re-enroll** option under the Students tab and retake images in the actual room lighting

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera feed not appearing | Check that no other application is using the webcam; try restarting the session |
| Student not being recognised | Re-train the model after registration; ensure the student is within 1.5–3 m of the webcam |
| "Model not trained" error | Click **Train Model** in the Students tab before starting a session |
| Engagement alerts firing incorrectly | Ensure adequate lighting; brief eye closures (blinking) should not trigger alerts as the timer requires sustained absence |
| Dashboard not loading | Ensure the Flask server is running (`python src/app.py`) and navigate to `http://127.0.0.1:5000` |
