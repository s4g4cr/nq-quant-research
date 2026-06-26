"""
HMM regime detector.

3 states (GaussianHMM, full covariance):
  ranging  — low vol, narrow OR, no trend
  trending — directional, gap-driven
  volatile — high vol, wide OR, erratic

Trained on training features only. Inference runs Viterbi on the
accumulated sequence up to (and including) the current day.
"""
import math
import pickle

import numpy as np
import pandas as pd

try:
    from hmmlearn import hmm as _hmm
except ImportError as exc:
    raise ImportError("hmmlearn not installed — run: pip install hmmlearn") from exc

FEATURE_COLS = ["realized_vol", "or_range_normalized", "overnight_gap", "trend_strength"]


class RegimeHMM:
    LABELS = ("ranging", "trending", "volatile")

    def __init__(self, n_states: int = 3, random_state: int = 42):
        self.n_states     = n_states
        self.random_state = random_state
        self.model        = None
        self.state_labels: dict[int, str] = {}   # state_idx -> label
        self.label_to_state: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, features: pd.DataFrame) -> "RegimeHMM":
        """
        Train on z-scored feature matrix (rows = trading days).
        Drops rows with any NaN before fitting.
        """
        X = features[FEATURE_COLS].dropna().values.astype(float)
        if len(X) < self.n_states * 10:
            raise ValueError(f"Too few valid training days ({len(X)}) for HMM fitting.")

        model = _hmm.GaussianHMM(
            n_components    = self.n_states,
            covariance_type = "full",
            n_iter          = 100,
            random_state    = self.random_state,
        )
        model.fit(X)
        self.model = model
        self._assign_labels(features)
        return self

    def _assign_labels(self, features: pd.DataFrame) -> None:
        valid   = features[FEATURE_COLS].dropna()
        states  = self.model.predict(valid.values.astype(float))

        # Map each HMM state to its mean realized_vol
        vol_means: dict[int, float] = {}
        for s in range(self.n_states):
            mask = states == s
            if mask.any():
                vol_means[s] = float(valid.iloc[mask]["realized_vol"].mean())
            else:
                vol_means[s] = 0.0

        sorted_by_vol = sorted(vol_means, key=lambda k: vol_means[k])
        self.state_labels[sorted_by_vol[0]]  = "ranging"
        self.state_labels[sorted_by_vol[-1]] = "volatile"
        for s in range(self.n_states):
            if s not in self.state_labels:
                self.state_labels[s] = "trending"

        self.label_to_state = {v: k for k, v in self.state_labels.items()}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_regimes(self, features: pd.DataFrame) -> pd.Series:
        """
        Return regime label for each valid row in `features`.
        Runs Viterbi on the full accumulated sequence (causal approximation:
        model trained on training data only; features are causally computed).
        """
        valid     = features[FEATURE_COLS].dropna()
        if len(valid) == 0:
            return pd.Series(dtype=str)

        X      = valid.values.astype(float)
        states = self.model.predict(X)
        labels = pd.Series(
            [self.state_labels.get(int(s), "unknown") for s in states],
            index = valid.index,
            name  = "regime",
        )
        return labels

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "RegimeHMM":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def print_diagnostics(
        self,
        features_raw: pd.DataFrame,
        regime_series: pd.Series,
        split_date,
    ) -> None:
        """
        Print HMM validation block (before any PnL results).

        features_raw : unnormalized features (same index as regime_series)
        split_date   : date separating train from test
        """
        split_date = pd.Timestamp(split_date).date()

        reg_idx  = pd.to_datetime(regime_series.index)
        tr_mask  = pd.to_datetime(regime_series.index).date < split_date  # type: ignore[operator]
        # date comparison fix
        dates_arr = np.array([pd.Timestamp(d).date() for d in regime_series.index])
        tr_mask   = dates_arr < split_date
        te_mask   = dates_arr >= split_date

        train_reg = regime_series[tr_mask]
        test_reg  = regime_series[te_mask]

        W = 64
        print("\n" + "=" * W)
        print("  HMM REGIME VALIDATION")
        print("=" * W)

        # 1. state distribution
        print("\n  1. DISTRIBUCION DE ESTADOS:")
        hdr = f"  {'Estado':<12} | {'Dias Tr':>7} | {'% Tr':>6} | {'Dias Te':>7} | {'% Te':>6}"
        print(hdr)
        print("  " + "-" * 12 + "-+-" + "-" * 7 + "-+-" + "-" * 6 + "-+-" + "-" * 7 + "-+-" + "-" * 6)
        for lbl in ("ranging", "trending", "volatile"):
            n_tr  = int((train_reg == lbl).sum())
            n_te  = int((test_reg  == lbl).sum())
            p_tr  = n_tr / len(train_reg) * 100 if len(train_reg) > 0 else 0.0
            p_te  = n_te / len(test_reg)  * 100 if len(test_reg)  > 0 else 0.0
            print(f"  {lbl:<12} | {n_tr:>7d} | {p_tr:>5.1f}% | {n_te:>7d} | {p_te:>5.1f}%")

        # 2. transition matrix
        print("\n  2. MATRIZ DE TRANSICION:")
        tm      = self.model.transmat_
        ordered = ["ranging", "trending", "volatile"]
        sord    = [self.label_to_state[l] for l in ordered]
        print(f"  {'':14} | {'ranging':>8} | {'trending':>8} | {'volatile':>8}")
        sep_row = "  " + "-" * 14 + "-+-" + "-" * 8 + "-+-" + "-" * 8 + "-+-" + "-" * 8
        print(sep_row)
        for from_lbl, fs in zip(ordered, sord):
            vals = [f"{tm[fs, ts]:.3f}" for ts in sord]
            print(f"  {from_lbl:<14} | {vals[0]:>8} | {vals[1]:>8} | {vals[2]:>8}")

        # 3. mean features per state (raw / unnormalized)
        print("\n  3. CARACTERISTICAS MEDIAS POR ESTADO (sin normalizar):")
        print(f"  {'Estado':<12} | {'Vol real':>9} | {'OR/ATR':>6} | {'Gap/ATR':>7} | {'Trend':>7}")
        sep_row2 = "  " + "-" * 12 + "-+-" + "-" * 9 + "-+-" + "-" * 6 + "-+-" + "-" * 7 + "-+-" + "-" * 7
        print(sep_row2)
        for lbl in ordered:
            mask = regime_series == lbl
            if mask.any():
                sub = features_raw.loc[regime_series.index[mask]]
                rv  = sub["realized_vol"].mean()
                orn = sub["or_range_normalized"].mean()
                gap = sub["overnight_gap"].mean()
                trd = sub["trend_strength"].mean()
                print(f"  {lbl:<12} | {rv:>9.5f} | {orn:>6.2f} | {gap:>7.3f} | {trd:>7.3f}")

        # 4. 2026 analysis
        yrs = np.array([pd.Timestamp(d).year for d in regime_series.index])
        mask_2026 = yrs == 2026
        if mask_2026.any():
            r26 = regime_series[mask_2026]
            print(f"\n  4. ANALISIS 2026 ({len(r26)} dias en test):")
            for lbl in ordered:
                n   = int((r26 == lbl).sum())
                pct = n / len(r26) * 100
                bar = "#" * int(pct / 3)
                print(f"     {lbl:<12}: {n:>3d} dias ({pct:>5.1f}%)  {bar}")

        print("=" * W + "\n")
