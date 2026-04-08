# ── Base image ─────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Working directory ──────────────────────────────────────────────────
WORKDIR /app

# ── Install dependencies ───────────────────────────────────────────────
# Copy requirements first — Docker caches this layer
# Only reinstalls if requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ───────────────────────────────────────────────────
COPY src/       src/
COPY api/       api/
COPY models/    models/

# ── Create empty __init__.py files ────────────────────────────────────
RUN touch src/__init__.py api/__init__.py

# ── Expose port ────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start the Flask API ────────────────────────────────────────────────
# Use gunicorn instead of the development server
# 4 workers = 4 simultaneous requests
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8000", "api.serve:app"]