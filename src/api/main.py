"""
FastAPI application entry point.

Run with:
  uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

API docs available at:
  http://localhost:8000/docs
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routers import portfolio, scenario, loan

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    log.info("Starting Mortgage Credit Risk API ...")
    # Pre-load heavy models and portfolio on startup to avoid cold-start latency
    try:
        from src.models.ecl_engine import ECLEngine
        app.state.engine = ECLEngine()
        log.info("ECL engine loaded successfully.")
    except Exception as e:
        log.warning("Could not pre-load ECL engine: %s", e)

    yield  # ← app is running

    log.info("Shutting down API ...")


app = FastAPI(
    title="Mortgage Credit Risk & Stress Testing Lab API",
    description=(
        "Production-grade ECL computation, PD/LGD modeling, "
        "and stress-scenario analysis for Freddie Mac mortgage loans."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow Next.js frontend) ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 1)
    response.headers["X-Process-Time-ms"] = str(elapsed)
    return response


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(portfolio.router)
app.include_router(scenario.router)
app.include_router(loan.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "Mortgage Credit Risk API"}


@app.get("/", tags=["Root"])
def root():
    return {
        "name":    "Mortgage Credit Risk & Stress Testing Lab",
        "version": "1.0.0",
        "docs":    "/docs",
    }
