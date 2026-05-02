# StatSense — Hypothesis Testing App
## Running on Windows with Docker

### Prerequisites (one-time setup)

1. **Install Docker Desktop for Windows**
   - Download from https://www.docker.com/products/docker-desktop/
   - Run the installer and restart your PC when prompted
   - After restart, open Docker Desktop and wait for the whale icon in the taskbar to stop animating — it's ready when it says "Engine running"

2. **WSL 2 backend** (Docker Desktop enables this automatically during install)

---

### Running the app

Open **PowerShell** or **Command Prompt** in the project folder, then:

#### Option A — Docker Compose (recommended, one command)

```powershell
docker compose up --build
```

Then open your browser at: **http://localhost:8501**

To stop:
```powershell
docker compose down
```

#### Option B — Plain Docker commands

```powershell
# Build the image
docker build -t statsense .

# Run the container
docker run -p 8501:8501 statsense
```

Then open your browser at: **http://localhost:8501**

To stop: press `Ctrl+C` in the terminal, or run:
```powershell
docker stop statsense
```

---

### Rebuilding after code changes

If you edit any `.py` file and want the container to pick up the changes:

```powershell
docker compose up --build
```

The `volumes` mount in `docker-compose.yml` means your local files are mapped
into the container, so in most cases a **browser refresh** is enough (Streamlit
auto-reloads on file save). A full rebuild is only needed after changing
`requirements.txt`.

---

### Troubleshooting

| Problem | Fix |
|---|---|
| `docker: command not found` | Docker Desktop isn't installed or not running — check the taskbar icon |
| Port 8501 already in use | Run `docker compose down` first, or change `8501:8501` to e.g. `8502:8501` in `docker-compose.yml` |
| Page doesn't load | Wait ~10 seconds after `docker compose up` for Streamlit to start, then refresh |
| Changes not showing | Save the file, wait 2 seconds, refresh the browser |

---

### Project structure

```
hypothesis_app/
├── app.py                  ← Streamlit entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── core/
│   ├── data_loader.py      ← datasets + upload
│   ├── wizard.py           ← guided question flow
│   ├── assumption_checks.py
│   ├── parametric.py       ← t-tests, ANOVA, Pearson
│   ├── nonparametric.py    ← Mann-Whitney, Wilcoxon, Kruskal, Chi², Spearman
│   ├── test_runner.py
│   └── interpreter.py      ← plain-English verdict
├── plots/
│   ├── distribution_plot.py
│   └── diagnostic_plots.py
└── theory/
    └── cards.py            ← expandable theory cards
```
