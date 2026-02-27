# 🦴 FHP Detection System

> **Real-time Forward Head Posture Detection & Correction** using Computer Vision + Deep Learning + Graph Convolutional Networks

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)

---

## 🎯 What Does This Do?

This system detects **Forward Head Posture (FHP)** — a common postural issue where the head juts forward relative to the shoulders — using only a webcam. It tracks key anatomical landmarks (head top → ear tragus → neck → shoulder → hand) and uses a **Spatio-Temporal Graph Convolutional Network (ST-GCN)** to classify posture as Normal or FHP in real-time.

### Why GCN Instead of Simple Rules?

Traditional approaches use hardcoded angle thresholds (e.g., CVA < 46° = FHP). This fails because:
- Camera angles vary → angles change
- Body proportions differ → thresholds don't generalize
- Single angles miss complex postural patterns

Our **GCN learns from data** — it captures the full spatial relationship between connected joints (shoulder→elbow→wrist chain, spine→neck→head chain) and temporal patterns across frames. The result: **genuinely adaptive detection, not predefined rules**.

---

## 🏗️ Architecture

```
Webcam → MediaPipe 2D Pose → VideoPose3D (2D→3D Lift) → Normalization → ST-GCN → FHP/Normal
                                                        ↓
                                              Biomechanical Features
                                              (CVA proxy, angles)
```

| Component | Purpose |
|---|---|
| **MediaPipe Pose** | Real-time 2D keypoint detection (33 landmarks → 17 H36M joints) |
| **VideoPose3D** | Lifts 2D joints to 3D space (pretrained on Human3.6M) |
| **Preprocessing** | Pelvis centering, torso-length scaling, spine alignment |
| **Biomechanical Module** | Computes CVA proxy, shoulder rounding, head displacement |
| **ST-GCN** | Classifies 3D skeleton sequences as Normal/FHP |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Real-time Detection
```bash
python src/realtime/app.py --config config.yaml
```
> ⚠️ First run will use demo mode (random weights). Train a model first for real detection.

### 3. Full Pipeline (Automated)
```bash
# Run the entire data pipeline
python scripts/run_pipeline.py --config config.yaml --stage all

# Or run individual stages:
python scripts/run_pipeline.py --stage collect      # Check raw data
python scripts/run_pipeline.py --stage detect_2d    # 2D pose estimation
python scripts/run_pipeline.py --stage lift_3d      # 2D→3D lifting
python scripts/run_pipeline.py --stage preprocess   # Normalize
python scripts/run_pipeline.py --stage label        # Interactive labeling
python scripts/run_pipeline.py --stage split        # Train/val/test split
python scripts/run_pipeline.py --stage verify       # Readiness check
```

### 4. Train the Model
```bash
python scripts/train.py --config config.yaml --epochs 200
```

---

## 📁 Project Structure

```
├── config.yaml                  # Master configuration
├── requirements.txt             # Dependencies
├── src/
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset + DataLoader
│   │   ├── preprocessing.py     # 3D pose normalization
│   │   ├── augmentation.py      # 3D skeleton augmentation
│   │   └── label_tools.py       # Visual labeling guide + tool
│   ├── models/
│   │   ├── pose_estimator.py    # MediaPipe wrapper
│   │   ├── videopose3d.py       # 2D→3D lifting
│   │   └── stgcn.py             # ST-GCN classifier
│   ├── realtime/
│   │   └── app.py               # Real-time webcam app
│   └── utils/
│       ├── skeleton.py          # Skeleton graph definitions
│       ├── angles.py            # Biomechanical computations
│       └── metrics.py           # Evaluation metrics
├── scripts/
│   ├── run_pipeline.py          # End-to-end automation
│   └── train.py                 # Training script
├── notebooks/                   # Colab notebooks
├── data/                        # Raw + processed data
├── models/                      # Checkpoints + exports
└── docs/                        # Labeling guides
```

---

## 📐 Key Anatomical Landmarks

| # | Landmark | Clinical Role |
|---|---|---|
| 1 | **Head Top** | Cranium position tracking |
| 2 | **Ear Tragus** | Gold standard CVA reference point |
| 3 | **Top Neck (C1-C2)** | Upper cervical flexion |
| 4 | **Bottom Neck (C7)** | CVA pivot point |
| 5 | **Shoulder (Acromion)** | Shoulder rounding + alignment |
| 6 | **Wrist/Hand** | Activity context (typing vs phone) |

---

## 🧠 Model Details

**Spatio-Temporal GCN (ST-GCN)**:
- **Spatial**: 3-layer GCN respecting skeleton connectivity (13 upper body joints)
- **Temporal**: 2-layer 1D convolution across 30-frame windows
- **Fusion**: Biomechanical features (6 angles) merged with learned embeddings
- **Output**: Binary classification (Normal vs FHP) with confidence score

---

## 📊 Data Labeling

The labeling tool provides a visual guide and interactive interface. Key rule:

> **If the ear is FORWARD of the shoulder line → FHP. If aligned or behind → Normal.**

Run the labeling tool:
```bash
python -c "from src.data.label_tools import ImageLabeler; ImageLabeler('data/raw').run()"
```

---

## 📚 References

- [PMC: FHP Recognition with GCN](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384178/) — GCN-based FHP detection methodology
- [Don't Be Turtle](https://github.com/motlabs/dont-be-turtle) — Mobile posture detection project
- [Human3.6M](http://vision.imar.ro/human3.6m) — 3.6M 3D human poses dataset
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) — 3D pose estimation from video

---

## 📄 License

Apache License 2.0
