# Presenz: A Face Recognition and Engagement Detection Based Attendance System
### Project ID: DP1-23-27-CSE-GEN-04

Presenz is a CPU-powered, web-based classroom attendance system that combines real-time face recognition with per-student behavioural engagement monitoring — all from a single standard webcam, with no GPU or external database required.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Setup Instructions](#setup-instructions)
- [Demo](#demo)
- [Results](#results)
- [Team](#team)

---

## Overview

Presenz solves two problems in one system. First, it automates attendance marking by using an LBPH face recogniser that confirms a student's identity across four consecutive video frames before logging them as present — eliminating false positives from motion blur or momentary occlusions. Second, it monitors each recognised student's behaviour in real time using a YOLOv8n object detector (phone detection) and a Haar Cascade eye-visibility heuristic (sleeping detection). When misconduct is detected, the student's attendance status is automatically downgraded from Present to Absent, and a timestamped JPEG snapshot is saved as evidence.

Everything runs on a standard CPU laptop. No GPU, no external database server, and no specialised hardware beyond a USB webcam.

---

## Features

- 🎥 Real-time face detection and LBPH-based recognition with 4-frame confirmation streak
- 📋 Automatic attendance logging with timestamp, subject, and status
- 🔒 Prevents proxy attendance and duplicate entries per session
- 😴 Engagement monitoring: detects phone usage (YOLOv8n + eye heuristic) and sleeping (eye absence + face disappearance)
- 📸 Timestamped JPEG snapshot evidence saved per engagement event
- 📊 Per-subject CSV attendance reports with status, reason, and snapshot columns
- 🌐 Browser-accessible dashboard (Flask + Socket.IO) with live annotated video feed
- 👤 Student management: register, rename, delete, and re-enroll via web UI
- 💻 No GPU, no database server — runs entirely on CPU with flat-file storage

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Face Detection | OpenCV 4.8 — Haar Cascade |
| Face Recognition | OpenCV LBPH (`cv2.face.LBPHFaceRecognizer`) |
| Engagement Detection | Ultralytics YOLOv8n (COCO pre-trained) + Haar Eye Cascade |
| Web Framework | Flask 2.3 + Flask-SocketIO 5.3 |
| Async Worker | Eventlet 0.33 |
| Frontend | HTML5, CSS3, JavaScript (ES6+), Socket.IO client, Jinja2 |
| Data Handling | NumPy 1.24, Pandas 2.0 |
| Storage | Flat-file CSV (attendance) + JPEG files (snapshots) — no database |

---

## Project Structure

```
Presenz/
├── src/
│   ├── app.py                        # Flask entry point, SocketIO session loop
│   └── modules/
│       ├── face_recognition_module.py  # LBPH enrollment, training, recognition
│       ├── engagement_module.py        # YOLOv8n + eye/face heuristics
│       └── attendance_manager.py       # CSV read/write, mark & update attendance
├── docs/
│   ├── project_report.md
│   ├── user_manual.md
│   └── api_reference.md
├── data/
│   ├── student_images/               # 60 enrollment images per student
│   ├── training_labels/              # lbph_model.yml + label_map.pkl
│   ├── attendance/                   # subject_YYYY-MM-DD.csv files
│   └── snapshots/                    # Evidence JPEG files
├── templates/                        # Jinja2 HTML templates (Dashboard, Students, Reports)
├── static/                           # CSS, JS assets
├── README.md
├── requirements.txt
├── architecture.png
├── demo_video_link.txt
└── setup_instructions.md
```

---

## Architecture

See [`architecture.png`](./architecture.png) for the full system architecture diagram.

**Processing Flow:**
```
Webcam → Frame Capture (960×540) → Grayscale + Histogram Equalisation
  ├── Face Detection (Haar Cascade)
  │     └── LBPH Recognition → 4-frame confirmation streak → Attendance Logged (CSV)
  └── Engagement Detection (every 5th frame)
        ├── Eye Visibility (Haar Eye Cascade) → Phone / Sleeping heuristic
        ├── Face Disappearance (grid cell tracking) → Sleeping alert
        └── YOLOv8n Phone Detection (COCO class 67) → Phone Usage alert
              └── Engagement Alert → Status downgrade + JPEG snapshot → CSV updated
Socket.IO → Live browser dashboard (annotated feed + attendance table + alert log)
```

---

## Setup Instructions

See [`setup_instructions.md`](./setup_instructions.md) for full step-by-step setup.

---

## Demo

See [`demo_video_link.txt`](./demo_video_link.txt) for the project demo video link.

---

## Results

| Metric | Value |
|--------|-------|
| Attendance Recognition Accuracy | 91.3% |
| Engagement Detection mAP50 | 83.6% |
| Phone Detection AP50 | 87.2% |
| Sleeping Detection AP50 | 76.4% |
| Avg. Attendance Mark Time (10 students) | 4.8 minutes |
| Avg. Frame Latency (CPU) | < 120 ms |
| Test Setup | 10 students × 5 sessions |

---

## Team

| Name | Reg. No | Role |
|------|---------|------|
| Nitin Hanumantha Rao | 2023BCSE07AED617 | Face Recognition Module |
| P S Lakshwin Ruup | 2023BCSE07AED503 | Engagement Detection Module |
| Royce Biju Thomas | 2023BCSE07AED354 | Web Interface & Dashboard |
| Gokul S | 2023BCSE07AED428 | Attendance Management & Testing |

> **Supervisor:** Dr. Ganga Holi
> **Institution:** Alliance School of Advanced Computing, Alliance University, Bangalore — 562106
> **Department:** Computer Science and Engineering (CSE-General)
> **Academic Year:** 2023–2027
