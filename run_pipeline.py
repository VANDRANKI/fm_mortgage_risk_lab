"""
Master pipeline runner for the Mortgage Credit Risk & Stress Testing Lab.

Runs all stages in order:
  1. Ingest origination files
  2. Ingest servicing files
  3. Combine years
  4. Pull macro data (FRED)
  5. Build monthly panel
  6. Add macro to panel
  7. Build PD modeling datasets
  8. Build LGD modeling dataset
  9. Train PD models
  10. Train LGD models
  11. Precompute scenario cache

Usage:
  py -3 run_pipeline.py [--start-from STAGE] [--years 2010 2011 ...]

Stages can be skipped with --start-from (useful for re-runs):
  py -3 run_pipeline.py --start-from 4  (skip ingestion, start from macro pull)
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# Make src importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log"),
    ],
)
log = logging.getLogger(__name__)


def stage(n: int, name: str, fn, start_from: int, **kwargs):
    if n < start_from:
        log.info("SKIP  [Stage %02d] %s", n, name)
        return
    log.info("START [Stage %02d] %s", n, name)
    t0 = time.time()
    fn(**kwargs)
    elapsed = time.time() - t0
    log.info("DONE  [Stage %02d] %s  (%.1fs)", n, name, elapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", type=int, default=1,
                        help="Start from this stage number (1-11)")
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Vintage years to process (default: 2010-2016)")
    args = parser.parse_args()

    sf = args.start_from

    Path("logs").mkdir(exist_ok=True)

    # ── Stage 1: Ingest origination ──────────────────────────────────────────
    from src.ingest.load_orig import load_all_orig_years
    stage(1, "Ingest origination files",
          load_all_orig_years, sf, years=args.years)

    # ── Stage 2: Ingest servicing ────────────────────────────────────────────
    from src.ingest.load_svcg import load_all_svcg_years
    stage(2, "Ingest servicing files",
          load_all_svcg_years, sf, years=args.years)

    # ── Stage 3: Combine years ───────────────────────────────────────────────
    from src.ingest.combine_years import combine_all
    stage(3, "Combine years into single tables",
          combine_all, sf, years=args.years)

    # ── Stage 4: Pull macro data ─────────────────────────────────────────────
    from src.macro.pull_fred import save_macro
    stage(4, "Pull / generate macro data (FRED)",
          save_macro, sf)

    # ── Stage 5: Build monthly panel ─────────────────────────────────────────
    from src.features.build_panel import build_all
    stage(5, "Build monthly loan panel + loan outcomes",
          build_all, sf)

    # ── Stage 6: Add macro to panel ──────────────────────────────────────────
    from src.features.add_macro import build_all as macro_build_all
    stage(6, "Merge macro into panel and outcomes",
          macro_build_all, sf)

    # ── Stage 7: Build PD datasets ───────────────────────────────────────────
    from src.features.build_pd_dataset import build_pd_datasets
    stage(7, "Build PD modeling datasets",
          build_pd_datasets, sf)

    # ── Stage 8: Build LGD dataset ───────────────────────────────────────────
    from src.features.build_lgd_dataset import build_lgd_dataset
    stage(8, "Build LGD modeling dataset",
          build_lgd_dataset, sf)

    # ── Stage 9: Train PD models ─────────────────────────────────────────────
    from src.models.train_pd import train_pd_models
    stage(9, "Train PD models",
          train_pd_models, sf)

    # ── Stage 10: Train LGD models ───────────────────────────────────────────
    from src.models.train_lgd import train_lgd_models
    stage(10, "Train LGD models",
          train_lgd_models, sf)

    # ── Stage 11: Precompute scenario cache ──────────────────────────────────
    from src.models.scenarios import precompute_and_cache_scenarios
    stage(11, "Precompute scenario cache for API",
          precompute_and_cache_scenarios, sf)

    log.info("Pipeline complete. You can now start the API with:")
    log.info("  uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
