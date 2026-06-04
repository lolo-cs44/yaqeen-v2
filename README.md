# Yaqeen — Smart AI-Powered Platform for Fake Media and News Detection

## Graduation Project

Department of Computer Science and Artificial Intelligence  
College of Computing  
Umm Al-Qura University, Saudi Arabia

Academic Year: 2025–2026  
Project ID: CSAI-472-P2-F04

---

## Team Members

| Name | Student ID |
|--------|------------|
| Layan Khalid Al-Hazmi | 443010340 |
| Alluluwah Salem Alqthamy | 444000440 |
| Haya Naif Alnefaie | 443000680 |
| Lana Eidha Alsalmi | 444000780 |

### Supervisor

Dr. Huda Alhazmi

---

## Project Overview

Yaqeen is an AI-powered Arabic platform developed to combat misinformation and manipulated digital media.

The platform combines:

- Arabic News Verification
- Deepfake Image Detection
- Deepfake Video Detection
- Knowledge-Based Evidence Retrieval

into a single unified web application.

Users can submit Arabic news articles, images, or videos and receive AI-generated verification results supported by confidence scores and trusted evidence sources.

The project addresses the growing challenge of fake news and AI-generated media by providing a reliable and accessible verification platform tailored for Arabic content.

---

## Key Features

### Arabic News Verification

- Verifies Arabic news articles and claims
- Uses Natural Language Processing (NLP)
- Retrieves supporting evidence from trusted Arabic sources
- Returns:
  - True
  - False
  - No Response

### Deepfake Detection

- Detects manipulated images
- Detects manipulated videos
- Provides confidence scores
- Supports visual content authentication

### Knowledge Base Retrieval

- FAISS-powered semantic search
- Trusted Arabic news sources
- Evidence-backed verification

### Additional Features

- Arabic Text-to-Speech (TTS)
- Share verification results
- User reporting system
- Rating and feedback system
- Admin dashboard

---

## System Architecture

The platform follows a Model-View-Controller (MVC) architecture.

```
Frontend (Firebase Hosting)
        ↓
FastAPI Backend
        ↓
    AI Models
    ├── CAMeLBERT
    ├── Xception
    └── FAISS Knowledge Base
        ↓
Verification Results
```

---

## AI Models

### 1. Arabic News Verification

**Model:** CAMeLBERT

**Purpose:** Arabic fake-news detection and credibility classification.

**Output:** True / False / No Response

**Performance:**

| Metric | Score |
|----------|----------|
| Accuracy | 85.04% |
| AUC-ROC | 94.21% |

---

### 2. Deepfake Detection

**Model:** Xception

**Purpose:** Detection of manipulated images and videos.

**Performance:**

| Metric | Score |
|----------|----------|
| Accuracy | 98.00% |
| AUC-ROC | 99.74% |

---

## Knowledge Base

The platform uses a curated Arabic knowledge base consisting of:

- Government announcements
- Official Saudi sources
- Trusted Arabic news agencies
- Verified news records

Semantic retrieval is performed using **FAISS** and **Sentence Transformers** to provide supporting evidence alongside prediction results.

---

## Technology Stack

| Layer | Technology |
|---------|-------------|
| Frontend | HTML, CSS, JavaScript |
| Hosting | Firebase Hosting |
| Backend | FastAPI |
| Server | Uvicorn |
| Deepfake Detection | Xception, PyTorch |
| Face Processing | MTCNN |
| NLP | CAMeLBERT |
| Transformers | Hugging Face |
| Semantic Search | FAISS |
| Database | Firebase Firestore |
| Model Hosting | Hugging Face Hub |
| Backend Hosting | Hugging Face Spaces |

---

## Repository Structure

```text
yaqeen/
│
├── assets/
│   └── screenshots/
│
├── frontend/
│   ├── index.html
│   ├── css/
│   └── js/
│
├── backend/
│   ├── main.py
│   ├── routes/
│   ├── services/
│   └── utils/
│
├── models/
│   ├── camelbert/
│   └── xception/
│
├── knowledge_base/
│   ├── faiss_index.bin
│   └── knowledge_base.json
│
├── firebase.json
├── requirements.txt
└── README.md
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

---

### News Verification

```http
POST /predict
```

Request:

```json
{
  "text": "النص الإخباري"
}
```

Response:

```json
{
  "label": "TRUE",
  "confidence": 92.4
}
```

---

### Deepfake Detection

```http
POST /predict-media
```

Request: Multipart Form Data

```
file=image.jpg
```

Response:

```json
{
  "label": "REAL",
  "confidence": 98.7
}
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/yaqeen.git
cd yaqeen
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the backend:

```bash
uvicorn main:app --reload
```

Open:

```
http://localhost:8000
```

---

## Deployment

| Component | Platform |
|-----------|----------|
| Frontend | Firebase Hosting |
| Backend | Hugging Face Spaces |
| Model Storage | Hugging Face Hub |

---

## Screenshots

### Home Page
![Home Page](assets/screenshots/home.png)

---

### News Verification — True
![News Verification True](assets/screenshots/news-verification-true.jpeg)

---

### News Verification — False
![News Verification False](assets/screenshots/news-verification-false.jpeg)

---

### Deepfake Detection — Fake
![Deepfake Fake](assets/screenshots/deepfake-fake.jpeg)

---

### Deepfake Detection — Real
![Deepfake Real](assets/screenshots/deepfake-real.jpeg)

---

### Admin Dashboard
![Admin Dashboard](assets/screenshots/admin-dashboard.png)

---

## Future Work

- Audio deepfake detection
- Browser extension
- Mobile application
- Multi-language support
- Real-time social media verification
- Continuous learning from verified reports

---

## Research Contribution

Yaqeen contributes by combining Arabic fake news verification, deepfake media detection, evidence retrieval using FAISS, and accessibility features within a single unified platform specifically designed for Arabic users.

---

## Acknowledgments

We would like to express our sincere gratitude to **Dr. Huda Alhazmi**, the Department of Computer Science and Artificial Intelligence, and the College of Computing at **Umm Al-Qura University** for their guidance and support throughout this project.

---

## License

Academic Use Only  
© 2026 Yaqeen Team — All rights reserved.
