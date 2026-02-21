"""
Master script to recompute all cached scenario outputs.

Run this after retraining models to refresh the UI cache.
Usage:
  py -3 src/models/recompute_all_scenarios.py
"""
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.scenarios import precompute_and_cache_scenarios

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

if __name__ == "__main__":
    precompute_and_cache_scenarios()
    print("Done. All scenario caches refreshed.")
