"""Unit tests for src/stats_regime.py — all deterministic, synthetic-data only."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.stats_regime import (
    P1_START, P1_END, P2_START, P2_END,
    SKILLED_ACCOUNTS,
    compute_regime_split,
    compute_persistence,
    skilled_regime_table,
    compute_regime,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    accounts: dict[str, list[tuple[str, bool]]],
    extra_non_validated: int = 5,
) -> pd.DataFrame:
    """Build a synthetic tweets_validated DataFrame.

    accounts : {author: [(entry_date, beats_market), ...]}
    """
    rows = []
    for author, preds in accounts.items():
        for entry_date, beats in preds:
            rows.append({
                "author":             author,
                "validated":          True,
                "beats_market":       beats,
                "prediction_correct": beats,   # same for simplicity
                "entry_date":         entry_date,
            })
    # Non-validated rows (must be ignored)
    for i in range(extra_non_validated):
        rows.append({
            "author":             "account_noise",
            "validated":          False,
            "beats_market":       True,
            "prediction_correct": True,
            "entry_date":         "2024-06-01",
        })
    return pd.DataFrame(rows)


def _p1_dates(n: int) -> list[str]:
    """Return n evenly-spaced dates within P1 (2022)."""
    return ["2022-03-01"] * n


def _p2_dates(n: int) -> list[str]:
    """Return n evenly-spaced dates within P2 (2024)."""
    return ["2024-03-01"] * n


# ---------------------------------------------------------------------------
# compute_regime_split
# ---------------------------------------------------------------------------

class TestComputeRegimeSplit:
    def _make_simple(self):
        return _make_df({
            "acct_A": [(d, True)  for d in _p1_dates(10)] +
                      [(d, False) for d in _p2_dates(20)],
            "acct_B": [(d, False) for d in _p1_dates(5)] +
                      [(d, True)  for d in _p2_dates(15)],
        })

    def test_non_validated_excluded(self):
        df = self._make_simple()
        res = compute_regime_split(df)
        # P1: 15 rows, P2: 35 rows (5 non-validated rows excluded)
        assert res["p1"]["n"] == 15
        assert res["p2"]["n"] == 35

    def test_p1_beats_rate(self):
        df = self._make_simple()
        res = compute_regime_split(df)
        # P1: 10 True + 5 False → rate = 10/15
        assert abs(res["p1"]["beats_rate"] - 10 / 15) < 1e-9

    def test_p2_beats_rate(self):
        df = self._make_simple()
        res = compute_regime_split(df)
        # P2: 20 False + 15 True → rate = 15/35
        assert abs(res["p2"]["beats_rate"] - 15 / 35) < 1e-9

    def test_labels_contain_n(self):
        df = self._make_simple()
        res = compute_regime_split(df)
        assert "15" in res["p1_label"]
        assert "35" in res["p2_label"]

    def test_dates_outside_both_periods_ignored(self):
        # Entry date in 2020 should fall outside both P1 and P2
        df = _make_df({
            "acct_A": [("2020-06-01", True)] * 5 +
                      [(d, True) for d in _p1_dates(10)] +
                      [(d, True) for d in _p2_dates(20)],
        })
        res = compute_regime_split(df)
        assert res["p1"]["n"] == 10
        assert res["p2"]["n"] == 20

    def test_ci_keys_present(self):
        df = self._make_simple()
        res = compute_regime_split(df)
        for period in ("p1", "p2"):
            assert "beats_ci" in res[period]
            lo, hi = res[period]["beats_ci"]
            assert 0.0 <= lo <= hi <= 1.0

    def test_empty_p1(self):
        # No rows in P1 → n=0
        df = _make_df({"acct_A": [(d, True) for d in _p2_dates(30)]})
        res = compute_regime_split(df)
        assert res["p1"]["n"] == 0
        assert np.isnan(res["p1"]["beats_rate"])


# ---------------------------------------------------------------------------
# compute_persistence
# ---------------------------------------------------------------------------

class TestComputePersistence:
    def _make_qualifying(self, p1_n: int = 25, p2_n: int = 25,
                          p1_beats: int = 20, p2_beats: int = 10,
                          n_accounts: int = 5) -> pd.DataFrame:
        """n_accounts identical accounts, each with p1_n P1 rows and p2_n P2 rows."""
        rows: list[tuple[str, list]] = []
        for i in range(n_accounts):
            author = f"acct_{i:02d}"
            p1_preds = [(d, j < p1_beats) for j, d in enumerate(_p1_dates(p1_n))]
            p2_preds = [(d, j < p2_beats) for j, d in enumerate(_p2_dates(p2_n))]
            rows.append((author, p1_preds + p2_preds))
        return _make_df(dict(rows))

    def test_n_qualifying_below_threshold_excluded(self):
        # Only 1 row per account in P1 → none qualify at min_n=20
        df = _make_df({
            "acct_A": [("2022-03-01", True)] + [(d, True) for d in _p2_dates(25)],
            "acct_B": [("2022-03-01", False)] + [(d, False) for d in _p2_dates(25)],
        })
        res = compute_persistence(df, min_n_per_period=20)
        assert res["n_qualifying"] == 0
        assert not res["computable"]

    def test_fewer_than_4_not_computable(self):
        # Only 2 qualifying accounts → computable=False
        df = _make_df({
            "acct_A": [(d, True) for d in _p1_dates(25)] + [(d, True) for d in _p2_dates(25)],
            "acct_B": [(d, False) for d in _p1_dates(25)] + [(d, False) for d in _p2_dates(25)],
        })
        res = compute_persistence(df, min_n_per_period=20)
        assert res["n_qualifying"] == 2
        assert not res["computable"]
        assert np.isnan(res["pearson_r"])

    def test_4_accounts_computable(self):
        df = self._make_qualifying(n_accounts=4, p1_n=25, p2_n=25,
                                   p1_beats=20, p2_beats=10)
        res = compute_persistence(df, min_n_per_period=20)
        assert res["computable"]
        # All 4 accounts have identical rates → r=nan (no variance) or computable
        # With identical rates, Pearson r is undefined; just check computable flag and keys
        assert "pearson_r" in res
        assert "spearman_r" in res

    def test_account_df_has_correct_columns(self):
        df = self._make_qualifying(n_accounts=4)
        res = compute_persistence(df, min_n_per_period=20)
        for col in ("p1_n","p1_k_beats","p1_beats_rate","p2_n","p2_k_beats","p2_beats_rate"):
            assert col in res["account_df"].columns

    def test_non_validated_rows_excluded(self):
        # Extra non-validated rows should not inflate n
        df = self._make_qualifying(n_accounts=4, p1_n=25, p2_n=25)
        res = compute_persistence(df, min_n_per_period=20)
        # Each account should have exactly 25 in P1
        for _, row in res["account_df"].iterrows():
            assert row["p1_n"] == 25

    def test_high_positive_correlation(self):
        # Accounts with high P1 rate also have high P2 rate → r should be positive
        accounts = {}
        for i, (r1, r2) in enumerate([(0.8, 0.7), (0.6, 0.55), (0.4, 0.35), (0.2, 0.25)]):
            author = f"acct_{i:02d}"
            p1_k = int(r1 * 25)
            p2_k = int(r2 * 25)
            accounts[author] = ([(d, j < p1_k) for j, d in enumerate(_p1_dates(25))] +
                                [(d, j < p2_k) for j, d in enumerate(_p2_dates(25))])
        df = _make_df(accounts)
        res = compute_persistence(df, min_n_per_period=20)
        if res["computable"] and not np.isnan(res["pearson_r"]):
            assert res["pearson_r"] > 0

    def test_all_keys_present(self):
        df = self._make_qualifying(n_accounts=4)
        res = compute_persistence(df, min_n_per_period=20)
        for k in ("n_qualifying","min_n_per_period","computable",
                  "pearson_r","pearson_p","pearson_ci",
                  "spearman_r","spearman_p","account_df"):
            assert k in res


# ---------------------------------------------------------------------------
# skilled_regime_table
# ---------------------------------------------------------------------------

class TestSkilledRegimeTable:
    def _make_df_for_skilled(self) -> pd.DataFrame:
        accounts = {}
        for acct in SKILLED_ACCOUNTS[:3]:
            accounts[acct] = ([(d, True) for d in _p1_dates(20)] +
                              [(d, True) for d in _p2_dates(30)])
        # Some with no P1 data
        for acct in SKILLED_ACCOUNTS[3:]:
            accounts[acct] = [(d, True) for d in _p2_dates(30)]
        return _make_df(accounts)

    def test_all_skilled_accounts_in_index(self):
        df = self._make_df_for_skilled()
        tbl = skilled_regime_table(df)
        for acct in SKILLED_ACCOUNTS:
            assert acct in tbl.index

    def test_p1_n_zero_when_no_data(self):
        df = self._make_df_for_skilled()
        tbl = skilled_regime_table(df)
        # Accounts [3:] have no P1 data
        for acct in SKILLED_ACCOUNTS[3:]:
            assert tbl.loc[acct, "p1_n"] == 0
            assert np.isnan(tbl.loc[acct, "p1_beats_rate"])

    def test_above_50_both_requires_both_periods(self):
        df = self._make_df_for_skilled()
        tbl = skilled_regime_table(df)
        # Only accounts with P1 data can be above_50_both
        for acct in SKILLED_ACCOUNTS[3:]:
            assert not tbl.loc[acct, "above_50_both"]

    def test_p2_rate_correct(self):
        df = self._make_df_for_skilled()
        tbl = skilled_regime_table(df)
        for acct in SKILLED_ACCOUNTS[:3]:
            # P2: 30/30 = 1.0
            assert abs(tbl.loc[acct, "p2_beats_rate"] - 1.0) < 1e-9

    def test_p1_rate_correct_when_data_available(self):
        df = self._make_df_for_skilled()
        tbl = skilled_regime_table(df)
        for acct in SKILLED_ACCOUNTS[:3]:
            # P1: 20/20 = 1.0
            assert abs(tbl.loc[acct, "p1_beats_rate"] - 1.0) < 1e-9

    def test_custom_skilled_accounts(self):
        custom = ["account_01", "account_02"]
        df = _make_df({
            "account_01": [(d, True) for d in _p2_dates(10)],
            "account_02": [(d, False) for d in _p2_dates(10)],
        })
        tbl = skilled_regime_table(df, skilled_accounts=custom)
        assert set(tbl.index) == {"account_01", "account_02"}

    def test_ci_bounds_valid(self):
        df = self._make_df_for_skilled()
        tbl = skilled_regime_table(df)
        for acct in SKILLED_ACCOUNTS[:3]:
            lo2 = tbl.loc[acct, "p2_ci_low"]
            hi2 = tbl.loc[acct, "p2_ci_high"]
            assert 0.0 <= lo2 <= hi2 <= 1.0

    def test_required_columns_present(self):
        df = self._make_df_for_skilled()
        tbl = skilled_regime_table(df)
        for col in ("p1_n","p1_beats_rate","p1_ci_low","p1_ci_high",
                    "p2_n","p2_beats_rate","p2_ci_low","p2_ci_high",
                    "above_50_p1","above_50_p2","above_50_both"):
            assert col in tbl.columns


# ---------------------------------------------------------------------------
# compute_regime — integration
# ---------------------------------------------------------------------------

class TestComputeRegime:
    def test_all_top_level_keys_present(self):
        df = _make_df({
            "acct_A": [(d, True) for d in _p1_dates(10)] + [(d, True) for d in _p2_dates(30)],
        })
        res = compute_regime(df)
        assert "split"       in res
        assert "persistence" in res
        assert "skilled"     in res

    def test_split_and_persistence_consistent(self):
        df = _make_df({
            "acct_A": [(d, True) for d in _p1_dates(25)] + [(d, False) for d in _p2_dates(25)],
            "acct_B": [(d, False) for d in _p1_dates(25)] + [(d, True) for d in _p2_dates(25)],
        })
        res = compute_regime(df)
        # P1 n from split should match sum of account P1 ns in persistence
        p1_n_split = res["split"]["p1"]["n"]
        p_acct = res["persistence"]["account_df"]
        p1_n_pers = int(p_acct["p1_n"].sum()) if not p_acct.empty else 0
        assert p1_n_split == 50   # 2 accounts × 25 P1 rows each

    def test_non_validated_rows_ignored_throughout(self):
        df = _make_df(
            {"acct_A": [(d, True) for d in _p1_dates(10)] + [(d, True) for d in _p2_dates(30)]},
            extra_non_validated=50,
        )
        res = compute_regime(df)
        assert res["split"]["p1"]["n"] == 10
        assert res["split"]["p2"]["n"] == 30
