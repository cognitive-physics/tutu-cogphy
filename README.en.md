# Cognitive Engine

> A research-oriented repository that organizes a three-layer closed-loop architecture — **Encoder → Four-Dimensional Responsibility Model → Decoder** — into readable documents, runnable prototype code, and extensible API components.

## Overview

This repository packages several core Notion documents into a cleaner GitHub project structure:

- **Theory layer**: encoder, four-dimensional responsibility model, decoder, and global formalism
- **Core prototype layer**: `cognitive_engine/engine.py`
- **Interface layer**: FastAPI service in `cognitive_engine/api.py`
- **Demo layer**: Streamlit frontend in `frontend/app.py`

This project should currently be understood as:

- a **research prototype**,
- a **theory-to-engineering mapping skeleton**,
- and a **starting point** for future experiments, refactors, productization, and documentation expansion.

It should **not yet** be treated as a production-ready or experimentally validated scientific software system.

## Core Architecture

### 1. Encoder

The input layer translates fuzzy human signals into structured parameter estimates.

Main features:

- multi-channel sampling
- inverse-problem style estimation
- Bayesian frame-by-frame updating
- cross-channel `D_KL` denoising
- convergence detection

### 2. Four-Dimensional Responsibility Model

The middle layer decomposes events into four orthogonal dimensions:

- behavioral responsibility
- developmental / educational responsibility
- environmental responsibility
- growth responsibility

The fourth dimension is the action-producing layer, formally associated with:

```math
δA = 0
```

### 3. Decoder

The output layer translates an already-computed optimal path into a form the receiver can actually absorb.

Main features:

- conclusion-first output
- high-compression communication
- bandwidth adaptation
- `η` (decode-rate) feedback loop
- dynamic adjustment of compression rate `c`

## Repository Layout

```text
.
├── docs/
│   ├── encoder.md
│   ├── four-dimensional-responsibility.md
│   ├── decoder.md
│   └── formalism.md
├── cognitive_engine/
│   ├── __init__.py
│   ├── engine.py
│   ├── complexity.py
│   ├── store.py
│   └── api.py
├── frontend/
│   └── app.py
├── README.md
├── README.en.md
├── README.zh-CN.md
├── LICENSE
├── requirements.txt
├── Dockerfile
└── .gitignore
```

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the backend API

```bash
uvicorn cognitive_engine.api:app --reload
```

Default address: `http://localhost:8000`

### Run the frontend demo

```bash
streamlit run frontend/app.py
```

Default address: `http://localhost:8501`

## Main API Endpoints

- `POST /chat`
- `POST /feedback`
- `GET /profile/{person_id}`
- `DELETE /profile/{person_id}`
- `GET /health`

## Current Status

This repository is most suitable right now for:

- theory organization and archival
- prototype validation
- API / frontend interaction testing
- use as a base for later productization or paper-adjacent materials

## Recommended Next Steps

- develop a more stable calibration strategy for `eta`
- design clearer experiments around the `DecisionEngine` objective
- replace `complexity.py` with an embedding-based or stronger encoder-based approach
- add tests, notebooks, examples, and expanded documentation

## License

This repository is currently distributed under the [MIT License](LICENSE).
