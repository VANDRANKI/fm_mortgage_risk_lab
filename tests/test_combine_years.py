"""
Tests for src/ingest/combine_years.py.

The per-year Parquet files are tiny synthetic frames written to a temp
directory, so no Freddie Mac data is needed.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest import combine_years  # noqa: E402


def _write_orig(out_dir, year, loans):
    pd.DataFrame({"LOAN_SEQUENCE_NUMBER": loans, "VINTAGE_YEAR": year}).to_parquet(
        out_dir / f"orig_{year}.parquet", index=False
    )


def _write_svcg(out_dir, year, rows):
    pd.DataFrame(rows, columns=["LOAN_SEQUENCE_NUMBER", "REPORTING_DATE", "UPB"]).to_parquet(
        out_dir / f"svcg_{year}.parquet", index=False
    )


def test_combine_orig_concatenates_every_year(tmp_path):
    _write_orig(tmp_path, 2010, ["A", "B"])
    _write_orig(tmp_path, 2011, ["C"])

    path = combine_years.combine_orig([2010, 2011], out_dir=tmp_path)

    assert path == tmp_path / "orig_all.parquet"
    out = pd.read_parquet(path)
    assert list(out["LOAN_SEQUENCE_NUMBER"]) == ["A", "B", "C"]
    assert list(out.index) == [0, 1, 2]


def test_combine_orig_names_the_missing_year_file(tmp_path):
    _write_orig(tmp_path, 2010, ["A"])

    with pytest.raises(FileNotFoundError, match="orig_2011.parquet"):
        combine_years.combine_orig([2010, 2011], out_dir=tmp_path)

    assert not (tmp_path / "orig_all.parquet").exists()


def test_combine_svcg_sorts_by_loan_then_date(tmp_path):
    _write_svcg(tmp_path, 2010, [
        ("B", pd.Timestamp("2011-02-01"), 90.0),
        ("A", pd.Timestamp("2011-02-01"), 80.0),
    ])
    _write_svcg(tmp_path, 2011, [
        ("A", pd.Timestamp("2011-01-01"), 85.0),
        ("B", pd.Timestamp("2011-01-01"), 95.0),
    ])

    out = pd.read_parquet(combine_years.combine_svcg([2010, 2011], out_dir=tmp_path))

    assert list(out["LOAN_SEQUENCE_NUMBER"]) == ["A", "A", "B", "B"]
    assert list(out["UPB"]) == [85.0, 80.0, 95.0, 90.0]
    assert list(out.index) == [0, 1, 2, 3]


def test_combine_svcg_names_the_missing_year_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="svcg_2010.parquet"):
        combine_years.combine_svcg([2010], out_dir=tmp_path)


def test_combine_all_writes_both_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(combine_years, "DATA_INTERIM", tmp_path)
    _write_orig(tmp_path, 2010, ["A"])
    _write_svcg(tmp_path, 2010, [("A", pd.Timestamp("2011-01-01"), 10.0)])

    combine_years.combine_all([2010])

    assert (tmp_path / "orig_all.parquet").exists()
    assert (tmp_path / "svcg_all.parquet").exists()
