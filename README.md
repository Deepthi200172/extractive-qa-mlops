# Extractive Question Answering (BERT) — End-to-End MLOps Pipeline

This project builds an **extractive Question Answering** system using **BERT fine-tuned on SQuAD**, then serves it via **FastAPI** and ships it with **Docker**.

Pipeline (high-level):
SQuAD dataset → preprocessing/tokenization → BERT QA training on Colab GPU → save model artifacts (all artifacts in one Google Drive) → download `models/best/` to local → FastAPI loads model → Docker

---

## 1) Project Overview

**Goal:** Given a *question* and a *context paragraph*, return the **answer span** extracted from the context (or “no answer” when not supported).

**Why this project:** It demonstrates an end-to-end workflow that includes:
- reproducible environment & dependencies
- training + evaluation
- model artifact management
- API serving
- containerization
- CI/testing hooks (as applicable per course requirements)

---

## 2) Problem Definition & Data

### Task
Extractive QA:
- Input: `question`, `context`
- Output: `answer_text`, `start_char`, `end_char`, and a confidence score (optional)

### Dataset
- **SQuAD (Stanford Question Answering Dataset)**  
We use SQuAD to fine-tune a pretrained BERT-style model for span prediction.

### Data split
- Train / Validation (standard split provided by dataset)

---

## 3) System Architecture

### Components
1. **Data & preprocessing**
   - Load SQuAD
   - Clean and normalize text (lightweight)
   - Tokenize with a Hugging Face tokenizer
   - Map character-level answer spans → token-level start/end indices

2. **Training (Colab GPU)**
   - Fine-tune a BERT QA model on SQuAD
   - Track metrics during training (loss / EM / F1 if implemented)
   - Save checkpoints and the best model

3. **Artifact management (Google Drive)**
   - All training artifacts saved under one Drive folder:
     - model weights
     - tokenizer files
     - training config
     - metrics logs (if available)

4. **Local inference**
   - Download `models/best/` from Drive to local project folder
   - FastAPI loads the model at startup and serves predictions

5. **Containerization**
   - Docker image for running the inference service consistently

---

## 4) MLOps Practices

This repository aims to follow the course expectations (structure, reproducibility, serving, containerization, testing, etc.). :contentReference[oaicite:1]{index=1}

### Reproducible environment
- Python dependencies are pinned (see `pyproject.toml` / lockfile if included)
- Recommended: `uv` for env & dependency management (course standard)

### Version control (Git/GitHub)
- Feature branches + PR reviews (team workflow)
- Clear commit messages with meaningful changes

### Testing & code quality
- Unit tests (recommended target coverage per course)
- Pre-commit hooks (formatting/linting)

### Model artifacts
Saved artifacts are organized so the inference service can load them directly:
- `models/best/` is the canonical folder used by FastAPI

---

## 5) Monitoring & Reliability

Basic reliability considerations:
- `/health` endpoint to confirm the API is running
- structured logs for requests and inference latency (if implemented)
- input validation via Pydantic models
- graceful error messages for missing/invalid inputs

(You can extend this with Prometheus metrics, request tracing, etc.)

---

## 6) Team Collaboration

- Work is tracked via Git commit history and PR reviews.
- Each member should understand the full pipeline: data → training → artifacts → serving.

> Add your team member names and roles here:
- Member 1 — Data/Preprocessing
- Member 2 — Training/Evaluation
- Member 3 — Serving (FastAPI)
- Member 4 — Docker/CI/CD/Monitoring

---

## 7) Limitations & Future Work

**Limitations**
- Extractive QA depends on the answer being present in the provided context.
- Performance depends on domain similarity between SQuAD and real user contexts.

**Future work**
- Add retrieval (RAG-style) to fetch relevant contexts automatically
- Add better confidence calibration and abstention (“no answer”)
- Improve monitoring (metrics, dashboards)
- Automate model registry + deployment using MLflow

---

# Repository Structure

Example structure (adjust to your repo):

