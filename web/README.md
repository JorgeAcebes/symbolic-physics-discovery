# Web Interface (`web/`)

Interactive Flask dashboard for launching experiments, inspecting results, and downloading outputs — without touching the source code.

---

## Directory Structure

```
web/
├── app.py          ← Flask server: REST API + HTML rendering
├── runner.py       ← Adapter between the web UI and src/
├── templates/
│   └── index.html  ← Main single-page UI
├── static/
│   ├── style.css
│   └── app.js
├── uploads/        ← Temporarily uploaded CSV files   [gitignored]
├── outputs/        ← Per-run results with UUID prefix [gitignored]
└── README.md
```

---

## Starting the Server

```bash
# From the repository root
cd web
python app.py
```

Open [http://localhost:5050](http://localhost:5050) in your browser.

> If you get an error on first load, wait a few seconds for the server to finish initialising, then reload the page.

---

## Usage Workflow

### 1 — Data

Choose your input data using one of two options:

- **Built-in dataset:** Select any CSV from the `data/` folder via the dropdown.
- **Custom file:** Upload an external CSV, JSON, or TXT file.
  - If the file lacks recognisable column headers, a column configurator will prompt you to name each column and mark which one is the target.

### 2 — Models

Click the model cards to toggle which models will be executed. Multiple models can be selected at once.

### 3 — Advanced Options *(optional)*

Expand the advanced panel to override default hyperparameters (number of epochs, learning rate, PySR iterations, etc.). Changes apply only to the current run and do not modify any source file.

### 4 — Launch

Click **Run**. The server executes all selected models as background jobs. A live log streams to the terminal panel in the UI.

### 5 — Results

When the run completes, the UI displays:

- **Metric cards** — MSE, MAE, and discovered equation for each model.
- **Plot gallery** — residual scatter plots and convergence curves.
- **Download buttons** — one per output file.

---

## Output Isolation

Every run gets its own subdirectory under `web/outputs/<uuid>/`. This means:

- Results from different runs never overwrite each other.
- The `results/`, `data/`, and `src/` directories of the main project are never modified by the web interface.
- The `uploads/` and `outputs/` folders are gitignored and will not be committed.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the main UI |
| `POST` | `/api/run` | Start an experiment job |
| `GET` | `/api/status/<job_id>` | Poll job status and live logs |
| `GET` | `/api/results/<job_id>` | Retrieve metrics and plot paths |
| `GET` | `/outputs/<job_id>/<filename>` | Download an output file |

---

## Requirements

The web interface uses the same virtual environment as the main pipeline. No additional dependencies are needed beyond `requirements.txt`.

Flask is pinned to `3.1.x` in `requirements.txt`.
