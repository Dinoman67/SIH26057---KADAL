# KADAL: Kilohertz Acoustic Debris & Anomaly Localization
*Autonomous multi-sensor marine debris & threat intelligence system (SIH 2026)*

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18%2B-61DAFB?logo=react&logoColor=black)](https://reactjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0%2B-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16%2B-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai)
<[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model%20(v6)-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/Dinoman1221/sonarvision-yolov8-esi-v6)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset%20(v6)-blue?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/Dinoman1221/sonarvision-multisource-v6)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Real Data](https://img.shields.io/badge/Real_Data-No_Synthetic_Renders-brightgreen)](#data-provenance-every-pixel-from-real-sonar-zero-synthetic-renders)
[![SIH 2026](https://img.shields.io/badge/Smart_India_Hackathon-2026-orange)](https://www.sih.gov.in/)

**Smart India Hackathon 2026 | Problem Statement: SIH26057 (Ministry of Earth Sciences / NIOT)**  
*Real-time AI for Marine Debris, Naval Mine Countermeasures (MCM), Shipwrecks, and Submerged Aircraft Localization in Side-Scan Sonar (SSS) Imagery.*

[Model (Hugging Face)](https://huggingface.co/Dinoman1221/sonarvision-yolov8-esi-v6) • [Dataset (Hugging Face)](https://huggingface.co/datasets/Dinoman1221/sonarvision-multisource-v6) • [Live Demo](#quick-start) • [Architecture](#solution-yolov8-esi-architecture) • [Benchmarks](#empirical-benchmarks) • [Provenance](#data-provenance-every-pixel-from-real-sonar-zero-synthetic-renders) • [Report Engine](#automated-intelligence-reporting) • [Roadmap](#roadmap)

</div>

> **Results at a glance:** mAP50 **0.6042** on **962** held-out test images • **0 false alarms** on clean seabed • **5.9 MB** FP16 ONNX • **~2.4 ms** GPU inference • trained on **100% real survey acoustics**.

---

## The Problem: Why Standard Computer Vision Fails on Sonar

Underwater marine debris, lost cargo containers, unexploded naval mines, and submerged aircraft are completely invisible from satellite optical cameras. Oceanographic vessels rely on **Side-Scan Sonar (SSS)**, which creates acoustic intensity maps of the seafloor.

Standard computer vision models (COCO-trained YOLOv8, Faster R-CNN) fail catastrophically on sonar:
* **No Color Information**: Sonar outputs single-channel acoustic backscatter intensity.
* **Acoustic Shadow Physics**: Objects are characterized not just by bright highlights, but by the **acoustic shadows** cast directly behind them based on towfish altitude and sound grazing angle.
* **Speckle Noise & Clutter**: Natural sand ripples, seafloor mud, and rocky reefs produce intense false alarms for brightness-dependent detectors.
* **Manual Review Bottleneck**: Surveyors spend days reviewing multi-gigabyte continuous waterfall records.

---

## Solution: YOLOv8-ESI Architecture

**YOLOv8-ESI** (Edge Sonar Intelligence) introduces **Squeeze-and-Excitation (SE)** channel attention directly into the C2f feature bottleneck of a lightweight CSPDarknet backbone:

```
[Raw SSS Imagery (256x256)]
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│              CSPDarknet Feature Extractor               │
│  ┌───────────────────────────────────────────────────┐  │
│  │   C2f Feature Block + SE Channel Attention       │  │
│  │   • Global Average Pooling (Spatial Squeeze)      │  │
│  │   • Two-Layer MLP Recalibration (Channel Excite) │  │
│  │   • Multiplies Highlight Features × Shadow Context│  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│           Decoupled Anchor-Free Detection Head          │
│   • Bounding Box Regression (CIoU Loss)                 │
│   • Multi-Class Classification (Class-Weighted BCE)     │
└─────────────────────────────────────────────────────────┘
           │
           ▼
[FP16 ONNX Engine] ──> [Real-Time Geospatial WGS84 Solver] ──> [PDF Report]
```

### Key Architectural Advantages
1. **Highlight-Shadow Coupling**: Channel attention forces the network to only trigger when an acoustic highlight is spatially correlated with a corresponding acoustic shadow.
2. **Compact Edge Footprint**: Only **3.03M parameters** and **5.9 MB** (FP16 ONNX), requiring zero high-end marine GPUs.
3. **Ultra-Low Latency**: **~2.4 ms** on NVIDIA GPUs, **<45 ms** on edge ARM CPUs (single-pass 256×256; tiled inference on large surveys is slower — see `backend/inference/engine.py`).

---

## Two-Model Operational Architecture

To provide maximum operational flexibility for maritime authorities and environmental teams, KADAL supports a two-model strategy:

| Component | Model 1: Debris Specialist (deprecated reference) | Model 2: Multi-Sensor Target Classifier (production champion) |
| :--- | :--- | :--- |
| **Model Type** | YOLOv8-ESI Single-Class | YOLOv8-ESI 4-Class Multi-Source |
| **Classes** | `marine_debris` | `unknown_debris`, `airplane`, `mine`, `wreck` |
| **Primary Domain** | High-density coastal cleanup & plastic mapping | Naval MCM, port security, maritime SAR & salvage |
| **mAP50 Score** | **0.8837 (reference only — train/test share survey passes, so this score is optimistic by construction; retained for pass-level analysis, not a deployment candidate)** | **0.6042** across 962 unseen multi-sensor test images |
| **Runtime Size** | 6.2 MB (FP16 ONNX) | 5.9 MB (FP16 ONNX) |
| **Deployment** | Archived reference only | Survey ships, Naval AUVs, coastal defense command |

---

## Empirical Benchmarks

### Unseen Test Split (962 Images, Zero File Overlap)

Evaluated strictly on independent, held-out side-scan sonar images (zero file overlap across splits; debris train/test share survey passes — see `reports/debris_feature_learning_report.md` for pass-level analysis):

**Benchmark methodology.** The 962-image test split was built with zero file overlap against train/val, so no test frame was seen in training. Per-sensor breakdowns (NOAA / Kaggle / MILCO below) matter because each sonar model has distinct speckle, gain, and shadow statistics — a single pooled score would hide sensor-specific failure modes.

| Object Type / Class | Benchmark Target | Baseline YOLO | **KADAL YOLOv8-ESI** | Detection Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Marine Debris** (`unknown_debris`) | $\ge 0.10$ | 0.0005 | **0.8185** | **81.9%** | **78.7%** |
| **Naval Mine** (`mine`) | $\ge 0.30$ | 0.1833 | **0.3862** | **70.7%** | **29.6%** |
| **Shipwreck** (`wreck`) | $\ge 0.70$ | 0.6698 | **0.7173** | **80.0%** | **60.4%** |
| **Submerged Aircraft** (`airplane`) | $\ge 0.70$ | 0.6206 | **0.4950** *(Val: 0.654)* | **57.2%** | **69.2%** |
| **Overall Model mAP50** | $\ge 0.50$ | 0.3685 | **0.6042** | **66.9%** | **62.4%** |
| **Peak F1 Score** | $\ge 0.60$ | 0.5100 | **0.6454** (@ conf 0.40) | — | — |
| **mAP50-95 (localization)** | — | — | **0.3348** | — | — |

### Multi-Sensor Cross-Validation
* **NOAA Klein 5000 SSS** (833 test images): **0.7578 mAP50** (P: 81.9%, R: 78.7%, F1: 0.8025)
* **KAGGLE High-Res Sonar** (65 test images): **0.6061 mAP50** (P: 80.0%, R: 60.3%, F1: 0.6881)
* **MILCO Klein 3500 MCM Sonar** (64 test images): **0.2714 mAP50** (P: 70.7% — high precision prevents false mine alerts)
* **Clean Seabed Validation**: **0 False Alarms** across natural seafloor sand ripples and mud textures.

*Reading the debris row: `unknown_debris` P/R (81.9%/78.7%) matches the NOAA row because debris test frames are the NOAA test frames — debris appears only in the NOAA split, so the two rows describe the same images, not two independent results.*

### Scope Notes: Coverage by Design

**Statement fidelity (SIH26057).** The statement asks for shipwrecks, pipes, cylinders, and entangled debris nets, plus anomaly reporting with geotagging, across debris, MCM, and SAR threat types. Our four classes (`unknown_debris`, `mine`, `wreck`, `airplane`), together with the GeoTIFF→WGS84 solver and the PDF/CSV/JSON engine (with optional NMEA/KML helpers), map directly onto those clauses.

**A note on ghost nets.** Ghost-net SSS detection is established science — e.g. GhostNetZero (Microsoft Research + WWF, 2025) reports ~90% detection from 412 expert-annotated Baltic/Puget Sound segments. That dataset is private, and no open student-usable ghost-net benchmark exists today. Our R&D decision: train on real survey acoustics only, using physics-preserving augmentation of real frames and no rendered objects (speckle/shadow statistics mismatch is documented to degrade real-survey transfer). Net-like contacts are therefore served through `unknown_debris`; a dedicated head becomes a fine-tune the day an open benchmark appears.

**Reading our numbers.** Headline metrics are held-out test (962 images) with per-class and per-sensor breakdowns. See `reports/debris_feature_learning_report.md` for our frame-vs-pass generalization analysis.

### Problem Statement Mapping

How each SIH26057 clause maps onto the implementation (partial coverage stated honestly — see Roadmap):

| SIH26057 Clause | Our Implementation | Coverage |
| :--- | :--- | :---: |
| Shipwrecks | `wreck` class (0.7173 AP50) | Full |
| Pipes, cylinders | Served via `unknown_debris` today | Partial → Roadmap |
| Entangled debris nets | Served via `unknown_debris` today (see ghost-net note above) | Partial → Roadmap |
| Anomaly reporting with geotagging | PDF/CSV/JSON reports + GeoTIFF→WGS84 solver (+ optional NMEA/KML helpers) | Full |
| Naval MCM coverage | `mine` class + threat triage + C2 exports | Full |
| Maritime SAR coverage | `airplane` class + geospatial reporting | Full |

---

## Data Provenance: Every Pixel From Real Sonar, Zero Synthetic Renders

Every training, validation, and test frame comes from real survey acoustics — **no synthetic renders, GAN-generated targets, or composited objects in any split** (4,033 train / 563 val / 962 test; 5,558 images total):

| Source | Sensor | Held-Out Test Images | Role |
| :--- | :---: | :---: | :--- |
| NOAA Hydrographic Surveys | Klein 5000 | 833 | Marine-debris evaluation |
| NATO STO CMRE MILCO | Klein 3500 | 64 | Mine-class (MCM) evaluation |
| Kaggle SSS Benchmark | High-res SSS | 65 | Airplane/wreck cross-sensor validation |

**Why this matters.** Synthetic sonar renders mismatch real acoustic physics — speckle statistics, TVG gain behavior, shadow gradients, and nadir geometry all differ from survey data — so models trained on renders degrade on real surveys. Training on real frames only is the reason the held-out numbers above transfer to the field. See `reports/debris_feature_learning_report.md` for how pass-memorization (not rendering) remains the hard generalization problem.

---

## Automated Intelligence Reporting

KADAL bridges raw AI detections with hydrographic GIS operations by generating instant intelligence deliverables:

1. **Publication-Ready PDF Reports**: Complete with executive summary, primary target type badge (`Naval Mine`, `Shipwreck`, `Marine Debris`), embedded high-res annotated imagery, detection inventory table, and narrative technical assessment.
2. **Tabular CSV Exports**: Detailed spreadsheets with target ID, object type, class name, bounding box bounds, center coordinates, and resolved WGS84 latitude/longitude.
3. **Machine-Readable JSON**: Complete API response schema for seamless integration into C2 (Command & Control) naval systems.
4. **Geospatial GeoTIFF Solver**: Solves the embedded affine transform matrix (`EPSG:26916` $\to$ `WGS84`) to project pixel bounding boxes into real-world geographic coordinates.
5. **Optional Field Exports (supporting)**: NMEA 0183 `$GPWPL` waypoints (`waypoints.txt`) and Google Earth KML dive-plan (`dive_plan.kml`) helpers under `/api/export`, for teams that already use ECDIS/chartplotters.
6. **Supporting Detection Context (optional)**: Acoustic backscatter/shadow context, threat ordering, and slant-range / XTF ingestion helpers (`backend/inference/`, `backend/reports/`) are available alongside the core detector.
7. **Optional Live Waterfall View (supporting demo)**: The viewer also offers a simulated transect mode alongside static analysis, reusing the same table/map panels; simulated contacts are labeled and kept separate from real analysis exports.
8. **Operator Review Queue**: Confirm/Reject verdicts per detection in the inventory table (persisted per analysis via `PATCH /api/export/{id}/verdicts`); verdicts are stamped into `detections.csv`/`results.json`, and the summary shows confirmed/rejected/pending counts.
9. **Evidence Bundle**: One-click ZIP (`GET /api/export/{id}/bundle`) with annotated/evidence/colormap imagery plus CSV, JSON, PDF, NMEA, and KML.
10. **Position Uncertainty**: Every georeferenced detection carries a conservative search radius (half max box-dimension × pixel resolution, method-labeled) shown on the map, in the table, and in CSV — a planning aid, never field-validated.

---

## Hugging Face Model & Dataset Downloads

To download the trained production model weights or access the acoustic side-scan sonar benchmark dataset, visit our official Hugging Face repositories:

| Resource | Hugging Face Repository | Description & Contents |
| :--- | :--- | :--- |
| **Model Weights (v6)** | [`Dinoman1221/sonarvision-yolov8-esi-v6`](https://huggingface.co/Dinoman1221/sonarvision-yolov8-esi-v6) | **YOLOv8-ESI v6 ONNX models** (`yolo_esi_v6_fp16.onnx` @ 5.9 MB, `yolo_esi_v6_fp32.onnx` @ 11.67 MB, and `yolo_esi_core_debris_fp16.onnx` @ 6.2 MB — leaked reference only), model cards with test benchmarks, and standalone ONNX inference code. |
| **Multi-Source Dataset (v6)** | [`Dinoman1221/sonarvision-multisource-v6`](https://huggingface.co/datasets/Dinoman1221/sonarvision-multisource-v6) | **924 MB archive** containing **5,558 side-scan sonar images** (4,033 train, 563 val, 962 strictly held-out test), `dataset.yaml`, 4 tactical target classes, and zero-leakage split protocol. |

### CLI Download Commands:
```bash
# Download production v6 FP16 model weights into models/
hf download Dinoman1221/sonarvision-yolov8-esi-v6 yolo_esi_v6_fp16.onnx --local-dir models/

# Download the complete multi-source v6 dataset (5,558 images)
hf download Dinoman1221/sonarvision-multisource-v6 sonarvision_multisource_v6.zip --repo-type dataset --local-dir datasets/
```

---

## Quick Start

### 1. Clone & Run (One-Command Startup)

Prerequisites: **Python 3.10+** and **Node.js 20+** (`node --version`). No manual setup beyond that:

```bash
git clone https://github.com/Dinoman67/sonarvision.git
cd sonarvision

# Launch unified application (FastAPI backend + React frontend)
./start.sh
```
`start.sh` automatically creates `.venv`, installs Python dependencies, builds the
frontend, and starts the server. For real (non-simulated) detections, download weights once:

```bash
.venv/bin/hf download Dinoman1221/sonarvision-yolov8-esi-v6 yolo_esi_v6_fp16.onnx --local-dir models/
cp models/yolo_esi_v6_fp16.onnx models/yolo_esi_fp16.onnx
./start.sh
```

Open your browser to:
**`http://localhost:8000`**

*(If Node.js is not installed, the backend immediately serves built production assets and starts in Demonstration Mode with preloaded multi-class test crops).*

The imagery workspace includes Static Analysis and an optional Live Waterfall (simulated transect) toggle; both feed the same inventory/map panels.

### 2. Activate Production ONNX Model

To run live GPU/CPU ONNX tensor inference:
1. Download or copy your trained model weights from Hugging Face into the `models/` folder:
   ```bash
   # Download directly from Hugging Face
   hf download Dinoman1221/sonarvision-yolov8-esi-v6 yolo_esi_v6_fp16.onnx --local-dir models/
   cp models/yolo_esi_v6_fp16.onnx models/yolo_esi_fp16.onnx

   # Or if you already have the file locally in Downloads:
   cp ~/Downloads/yolo_esi_v6_fp16.onnx models/yolo_esi_fp16.onnx
   ```
2. Restart the app (`./start.sh`). The backend will automatically bind the model and display execution provider details (`CUDAExecutionProvider` or `CPUExecutionProvider`).

### 3. Uploads With Geolocation
`POST /api/analyze` auto-resolves real-world coordinates, no model changes needed:
* **GeoTIFF** (embedded transform/CRS) and **EXIF-GPS JPGs** work out of the box.
* Plain PNG/JPG + georeferencing sidecars (`.tfw`, `.jgw`, `.pgw`, `.wld`, `.prj`, `.aux.xml`) via the optional `sidecars` form field.
* Native `.XTF` waterfall files are also accepted where available (slant-range helper applied when telemetry is present).
* Detections from files with no survey metadata return pixel boxes with an explicit non-georeferenced status — coordinates are never fabricated.

## Alignment with National & Global Goals

* **UN SDG 14: Life Below Water**: Autonomous spatial mapping of benthic plastics and ghost gear to direct cleanup vessels to high-density debris hotspots.
* **Atmanirbhar Bharat & Blue Economy**: Indigenous, sovereign deep-tech AI for naval port security, mine countermeasures, and economic zone surveillance without foreign dependencies.

---

## License & Acknowledgements

* Released under the **Apache 2.0 License** (see [LICENSE](LICENSE)).

## Roadmap

* **Temporal change detection**: epoch-over-epoch comparison of repeat survey passes to flag new, moved, or missing contacts.
* **Dedicated pipe/cylinder class**: split out of `unknown_debris` under the same no-synthetic-data policy — real survey frames only.
* **Ghost-net head**: a dedicated detection head pending an open real-acoustics benchmark (see the GhostNetZero note under Scope Notes); until then, net-like contacts stay served through `unknown_debris`.

---

## Team Cold Start

* Ashish S
* Sanjeev kumar S
* Kamlesh Y
* Prajan SS
* Sangamithra B
* Sudhishna P

---

* Developed for **Smart India Hackathon 2026** by Team **Cold Start**.
* Acoustic data sources: NOAA Hydrographic Survey Archives, NATO STO CMRE MILCO Benchmark, and Kaggle SSS Object Detection.
