# Setup Instructions

## Presenz: A Face Recognition and Engagement Detection Based Attendance System

---

## Prerequisites

- Python 3.11
- pip (Python package manager)
- A working webcam (USB or built-in, 720p or higher recommended)
- Git
- Minimum 8 GB RAM, 2 GB free disk space
- OS: Windows 10/11, Ubuntu 20.04+, or macOS 12+

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/nhsync/DP1-23-27-CSE-GEN-04-Face-Recognition-Attendance-System-.git
cd DP1-23-27-CSE-GEN-04-Face-Recognition-Attendance-System-
```

---

## Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** `opencv-python` includes Haar Cascade classifiers and the LBPH recogniser out of the box — no separate dlib or face_recognition installation is needed.

> ⚠️ **YOLOv8n weights:** On first run, Ultralytics will automatically download `yolov8n.pt` (approx. 6 MB) if it is not already present. Ensure you have an internet connection for the first launch.

---

## Step 4: Verify the Data Directory Structure

Presenz uses flat-file storage. Ensure the following directories exist (they will be created automatically on first run, but you can create them manually):

```
data/
├── student_images/     # Enrollment images are stored here
├── training_labels/    # Trained LBPH model files are saved here
├── attendance/         # CSV attendance logs are saved here
└── snapshots/          # JPEG engagement evidence is saved here
```

No database setup is required.

---

## Step 5: Run the Application

```bash
python src/app.py
```

The Flask server will start with Eventlet. Open your browser and navigate to:

```
http://127.0.0.1:5000
```

---

## Step 6: Register Students

1. In the browser, go to the **Students** tab
2. Click **Register New Student**, enter the student's name and ID
3. Have the student sit in front of the webcam and look at the camera
4. The system captures 60 images automatically — wait for the confirmation message
5. Repeat for all students

---

## Step 7: Train the Recognition Model

1. After registering all students, go to the **Students** tab
2. Click **Train Model**
3. Wait for the "Model trained successfully" message (approx. 3–5 seconds for 10 students)

---

## Step 8: Start an Attendance Session

1. Go to the **Dashboard** tab
2. Enter the subject name and click **Start Session**
3. The live camera feed will appear with face recognition and engagement monitoring active
4. Click **Stop Session** when done

---

## Step 9: View and Export Reports

1. Go to the **Reports** tab
2. Select a subject and date
3. View the attendance table with statuses, reasons, and snapshot thumbnails
4. Click **Download CSV** to export

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not detected | Check webcam connection; try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `app.py` |
| `yolov8n.pt` not found | Ensure internet access on first run; Ultralytics downloads it automatically |
| Face not recognised | Re-enroll the student in the actual room lighting and re-train the model |
| "Model not trained" error | Click **Train Model** in the Students tab before starting a session |
| Module not found | Ensure virtual environment is activated and `pip install -r requirements.txt` was run |
| Port 5000 already in use | Change the port in `app.py`: `socketio.run(app, port=5001)` |

---

## Project Structure Reference

```
src/
├── app.py                          # Flask entry point — run this to start Presenz
└── modules/
    ├── face_recognition_module.py  # LBPH enrollment, training, recognition
    ├── engagement_module.py        # YOLOv8n + Haar eye/face engagement detection
    └── attendance_manager.py       # CSV read/write for attendance and reports

data/
├── student_images/                 # Enrollment images (auto-created)
├── training_labels/                # lbph_model.yml + label_map.pkl (auto-created)
├── attendance/                     # subject_YYYY-MM-DD.csv files (auto-created)
└── snapshots/                      # JPEG evidence files (auto-created)

templates/                          # Jinja2 HTML templates
static/                             # CSS and JS assets
```
