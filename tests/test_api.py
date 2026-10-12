"""
Tests for the FastAPI layer in src/api.

The ECL engine is built without model artifacts (heuristic PD and LGD), and the
portfolio and cache files are redirected to a temp directory, so none of these
tests need the trained models or the Freddie Mac data on disk.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("httpx")  # required by fastapi.testclient

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402
from src.api.routers import loan, portfolio, scenario  # noqa: E402
from src.config.settings import SCENARIOS  # noqa: E402
from src.models.ecl_engine import ECLEngine  # noqa: E402

BASELINE_MACRO = pd.Series(
    {
        "unemployment_rate": 4.87,
        "hpi_yoy_chg": 5.6,
        "mortgage_rate_30y": 3.65,
        "financial_stress_idx": -0.5,
        "unrate_chg_mom": 0.0,
    }
)


def _heuristic_engine() -> ECLEngine:
    e = ECLEngine.__new__(ECLEngine)
    e.dataset_name = "pd_12m"
    e.pd_model = None
    e.lgd_model = None
    e.lgd_pre = None
    e._macro_baseline = BASELINE_MACRO.copy()
    return e


def _small_portfolio() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CREDIT_SCORE": [720, 640],
            "ORIGINAL_LTV": [75.0, 95.0],
            "ORIGINAL_DTI": [30.0, 45.0],
            "ORIGINAL_UPB": [150000.0, 250000.0],
            "CURRENT_ACTUAL_UPB": [140000.0, 240000.0],
            "DLQ_STATUS_INT": [0, 0],
            "LOAN_AGE_AT_OBS": [24, 12],
        }
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app.state, "engine", _heuristic_engine(), raising=False)
    return TestClient(app)


class TestRoot:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_root_points_at_docs(self, client):
        assert client.get("/").json()["docs"] == "/docs"

    def test_process_time_header_is_set(self, client):
        assert float(client.get("/health").headers["X-Process-Time-ms"]) >= 0


class TestRiskLevel:
    @pytest.mark.parametrize(
        "pd_value, label",
        [
            (0.0, "LOW"),
            (0.0099, "LOW"),
            (0.01, "MEDIUM"),
            (0.0299, "MEDIUM"),
            (0.03, "HIGH"),
            (0.0799, "HIGH"),
            (0.08, "VERY HIGH"),
            (1.0, "VERY HIGH"),
        ],
    )
    def test_band_edges(self, pd_value, label):
        assert loan._risk_level(pd_value) == label


class TestLoanPredict:
    def test_defaults_return_a_consistent_response(self, client):
        r = client.post("/loan/predict", json={})
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["pd_12m"] <= 1
        assert 0 <= body["lgd"] <= 1
        assert body["ead"] == 200000  # EAD is the original UPB
        assert body["ifrs9_stage"] in (1, 2, 3)
        assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH", "VERY HIGH"}
        assert body["ecl_rate"] == pytest.approx(body["ecl"] / body["ead"], abs=1e-5)

    def test_riskier_loan_has_higher_pd(self, client):
        safe = client.post("/loan/predict", json={"credit_score": 790, "original_ltv": 60}).json()
        risky = client.post("/loan/predict", json={"credit_score": 560, "original_ltv": 97}).json()
        assert risky["pd_12m"] > safe["pd_12m"]

    def test_severe_shock_does_not_lower_ecl(self, client):
        base = client.post("/loan/predict", json={}).json()
        stressed = client.post(
            "/loan/predict",
            json={"unemployment_shock": 8.0, "hpi_shock": -25.0, "rate_shock": 3.0},
        ).json()
        assert stressed["ecl"] >= base["ecl"]

    @pytest.mark.parametrize(
        "field, value",
        [
            ("credit_score", 299),
            ("credit_score", 851),
            ("original_upb", 999),
            ("original_loan_term", 12),
            ("unemployment_shock", 1e6),
            ("hpi_shock", -51),
            ("rate_shock", 11),
        ],
    )
    def test_out_of_range_input_is_a_422(self, client, field, value):
        assert client.post("/loan/predict", json={field: value}).status_code == 422


class TestEngineReuse:
    """main.py preloads app.state.engine; the routers must use it instead of
    building a second engine of their own."""

    def test_loan_router_uses_the_preloaded_engine(self, client, monkeypatch):
        monkeypatch.setattr(loan, "_ENGINE", None)
        assert client.post("/loan/predict", json={}).status_code == 200
        assert loan._ENGINE is None

    def test_scenario_router_uses_the_preloaded_engine(self, client, monkeypatch):
        monkeypatch.setattr(scenario, "_ENGINE", None)
        monkeypatch.setattr(scenario, "_get_portfolio", _small_portfolio)
        monkeypatch.setattr(scenario, "_get_baseline_ecl", lambda: 0.0)
        assert client.post("/scenario/run", json={}).status_code == 200
        assert scenario._ENGINE is None


class TestScenarioEndpoints:
    def test_list_matches_the_configured_scenarios(self, client):
        body = client.get("/scenario/list").json()
        assert [s["name"] for s in body] == list(SCENARIOS)
        assert all({"label", "color", "unemployment_shock"} <= set(s) for s in body)

    def test_run_reports_delta_against_the_cached_baseline(self, client, monkeypatch):
        monkeypatch.setattr(scenario, "_get_portfolio", _small_portfolio)
        monkeypatch.setattr(scenario, "_get_baseline_ecl", lambda: 1000.0)
        body = client.post("/scenario/run", json={"scenario_name": "mine"}).json()
        assert body["scenario_name"] == "mine"
        assert body["baseline_ecl"] == 1000.0
        assert body["ecl_delta"] == pytest.approx(body["total_ecl"] - 1000.0)
        assert body["ecl_delta_pct"] == pytest.approx((body["total_ecl"] - 1000.0) / 1000.0 * 100)

    def test_run_with_zero_baseline_does_not_divide_by_zero(self, client, monkeypatch):
        monkeypatch.setattr(scenario, "_get_portfolio", _small_portfolio)
        monkeypatch.setattr(scenario, "_get_baseline_ecl", lambda: 0.0)
        body = client.post("/scenario/run", json={}).json()
        assert body["ecl_delta_pct"] == pytest.approx(body["total_ecl"] * 100)

    @pytest.mark.parametrize(
        "payload",
        [{"unemployment_shock": 21}, {"hpi_shock": 21}, {"rate_shock": -4}],
    )
    def test_run_rejects_out_of_range_shocks(self, client, payload):
        assert client.post("/scenario/run", json=payload).status_code == 422

    def test_missing_snapshot_is_a_503(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(scenario, "PORTFOLIO_BASELINE", tmp_path)
        monkeypatch.setattr(scenario, "_PORTFOLIO", None)
        assert client.post("/scenario/run", json={}).status_code == 503

    def test_baseline_cache_defaults_to_zero_then_reads_the_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(scenario, "PORTFOLIO_BASELINE", tmp_path)
        monkeypatch.setattr(scenario, "_BASELINE_ECL", None)
        assert scenario._get_baseline_ecl() == 0.0

        monkeypatch.setattr(scenario, "_BASELINE_ECL", None)
        (tmp_path / "all_scenarios.json").write_text(json.dumps({"baseline": {"total_ecl": 42.5}}))
        assert scenario._get_baseline_ecl() == 42.5


class TestPortfolioEndpoints:
    @pytest.fixture(autouse=True)
    def _isolate_cache(self, monkeypatch, tmp_path):
        monkeypatch.setattr(portfolio, "PORTFOLIO_BASELINE", tmp_path)
        monkeypatch.setattr(portfolio, "_SCENARIOS_CACHE", None)
        monkeypatch.setattr(portfolio, "_VINTAGES_CACHE", None)
        self.root = tmp_path

    def test_missing_cache_is_a_503(self, client):
        assert client.get("/portfolio/overview").status_code == 503
        assert client.get("/portfolio/vintages").status_code == 503

    def test_overview_reads_the_baseline_entry(self, client):
        (self.root / "all_scenarios.json").write_text(
            json.dumps({"baseline": {"total_ead": 10.0, "total_ecl": 1.0, "loan_count": 3}})
        )
        body = client.get("/portfolio/overview").json()
        assert body["total_ead"] == 10.0
        assert body["loan_count"] == 3
        assert body["mean_pd"] == 0
        assert body["stage_summary"] == []

    def test_vintages_are_grouped_and_sorted_by_observation_year(self, client):
        pd.DataFrame(
            {
                "vintage_year": [2012, 2012, 2013],
                "obs_year": [2014, 2013, 2014],
                "default_rate": [0.02, 0.01, 0.03],
                "total_loans": [100, 110, 90],
            }
        ).to_parquet(self.root / "vintage_curves.parquet", index=False)
        body = client.get("/portfolio/vintages").json()
        assert [v["vintage_year"] for v in body] == [2012, 2013]
        assert [d["obs_year"] for d in body[0]["data"]] == [2013, 2014]
