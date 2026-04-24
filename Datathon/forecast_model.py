"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   DATATHON 2026 – Round 1  │  forecast_ensemble_v4.py                       ║
║   SEASONAL-HEAVY ENSEMBLE (no recursive ML — preserves spikes)              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Triết lý:
  - Vấn đề cốt lõi của v3 là recursive lag → forecast collapse về mean.
  - v4 loại bỏ HOÀN TOÀN recursive lag khỏi test prediction.
  - Chỉ dùng seasonal mapping + calendar features để preserve spikes.

5 model con (mỗi target):
  A. Last-Year Seasonal Mapping     – dùng pattern năm trước với trend scale
  B. Day-of-Year Robust Profile     – median theo dayofyear
  C. Calendar Profile               – (month×dow) + (weekofyear×dow)
  D. Recent-Year Weighted           – weighted average 2019-2022
  E. LightGBM Calendar-Only Direct  – ML với calendar features only, KHÔNG lag

Cài đặt:
    pip install lightgbm scikit-learn pandas numpy

Chạy:
    python forecast_ensemble_v4.py
    → submission.csv
"""

# ── Kiểm tra lightgbm ─────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
except ImportError:
    print("=" * 60)
    print("  LightGBM chưa cài!  Chạy lệnh:")
    print()
    print("      pip install lightgbm")
    print()
    print("  Sau đó chạy lại script.")
    print("=" * 60)
    raise SystemExit(1)

import os
import warnings
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─── Cấu hình ─────────────────────────────────────────────────────────────────
DATA_PATH    = "."
OUTPUT_FILE  = "submission.csv"
RANDOM_STATE = 42
TRAIN_CUTOFF = "2020-12-31"
VALID_START  = "2021-01-01"
VALID_END    = "2022-12-31"
COGS_CAP     = 0.95

np.random.seed(RANDOM_STATE)


# ══════════════════════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════════════════════

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def print_metrics(name: str, y_true, y_pred) -> float:
    y_pred = np.maximum(y_pred, 0)
    mae_  = mean_absolute_error(y_true, y_pred)
    rmse_ = rmse(y_true, y_pred)
    r2_   = r2_score(y_true, y_pred)
    print(f"    {name:<34}  MAE={mae_:>13,.1f}  RMSE={rmse_:>13,.1f}  R²={r2_:>7.4f}")
    return rmse_

def safe_array(arr: np.ndarray) -> np.ndarray:
    """Thay NaN bằng median của array."""
    arr = np.asarray(arr, dtype=float)
    if np.isnan(arr).any():
        med = np.nanmedian(arr)
        if np.isnan(med):
            med = 0.0
        arr = np.where(np.isnan(arr), med, arr)
    arr = np.where(np.isinf(arr), 0.0, arr)
    return arr

def recent_trend_scale(hist: pd.DataFrame, col: str,
                       ref_date: pd.Timestamp,
                       window_days: int = 90,
                       clip_low: float = 0.6,
                       clip_high: float = 1.4) -> float:
    """
    Scale factor = median(last `window_days` before ref_date) /
                   median(same window one year earlier)
    Dùng để correct trend cho seasonal mapping.
    """
    recent = hist[(hist["Date"] <  ref_date) &
                  (hist["Date"] >= ref_date - pd.Timedelta(days=window_days))]

    past   = hist[(hist["Date"] <  ref_date - pd.DateOffset(years=1)) &
                  (hist["Date"] >= ref_date - pd.DateOffset(years=1)
                                   - pd.Timedelta(days=window_days))]

    if len(recent) < 30 or len(past) < 30:
        return 1.0

    m_recent = recent[col].median()
    m_past   = past[col].median()

    if m_past <= 0:
        return 1.0

    return float(np.clip(m_recent / m_past, clip_low, clip_high))


# ══════════════════════════════════════════════════════════════════════════════
# MODEL A — LAST-YEAR SEASONAL MAPPING
# ══════════════════════════════════════════════════════════════════════════════

class LastYearModel:
    """
    Map forecast date → cùng ngày năm trước.
    - 2023 → 2022
    - 2024 → weighted (2022, 2021) thay vì recursive 2023.
    - Fallback: gần nhất theo dayofyear ±3 ngày.
    - Nhân trend scale factor.
    """

    def __init__(self, search_delta: int = 3):
        self.search_delta = search_delta
        self.hist_        = None

    def fit(self, sales: pd.DataFrame):
        self.hist_ = sales.sort_values("Date").reset_index(drop=True)
        # Index by (year, dayofyear) để lookup nhanh
        self._idx_ = {}
        for _, row in self.hist_.iterrows():
            key = (row["Date"].year, row["Date"].dayofyear)
            self._idx_[key] = (row["Revenue"], row["COGS"])
        return self

    def _lookup(self, year: int, doy: int, col_idx: int):
        """col_idx: 0=Revenue, 1=COGS. Trả về giá trị hoặc None."""
        for delta in range(0, self.search_delta + 1):
            for sign in ([0] if delta == 0 else [1, -1]):
                key = (year, doy + sign * delta)
                if key in self._idx_:
                    return self._idx_[key][col_idx]
        return None

    def predict(self, dates: pd.Series) -> pd.DataFrame:
        results = []
        for d in dates:
            year = d.year
            doy  = d.dayofyear

            # Chiến lược mapping:
            # - 2023 → 2022 chính, fallback 2021
            # - 2024 → weighted 0.7 * 2022 + 0.3 * 2021 (KHÔNG dùng 2023 recursive)
            if year == 2023:
                src_years = [(2022, 0.75), (2021, 0.25)]
            elif year == 2024:
                src_years = [(2022, 0.6), (2021, 0.3), (2020, 0.1)]
            else:
                src_years = [(year - 1, 1.0)]

            # Trend scale
            scale_rev = recent_trend_scale(self.hist_, "Revenue", d)
            scale_cog = recent_trend_scale(self.hist_, "COGS",    d)

            # Weighted lookup
            rev_sum, cog_sum, w_sum = 0.0, 0.0, 0.0
            for yr, w in src_years:
                r = self._lookup(yr, doy, 0)
                c = self._lookup(yr, doy, 1)
                if r is not None and c is not None:
                    rev_sum += w * r
                    cog_sum += w * c
                    w_sum   += w

            if w_sum > 0:
                rev = rev_sum / w_sum
                cog = cog_sum / w_sum
            else:
                # Nothing found → fallback: overall median
                rev = self.hist_["Revenue"].median()
                cog = self.hist_["COGS"].median()

            results.append({
                "Date"   : d,
                "Revenue": rev * scale_rev,
                "COGS"   : cog * scale_cog,
            })

        return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL B — DAY-OF-YEAR ROBUST PROFILE
# ══════════════════════════════════════════════════════════════════════════════

class DayOfYearModel:
    """
    Median theo dayofyear (robust, giữ spikes) +
    nhẹ nhàng smooth bằng rolling window 3 để giảm noise,
    nhưng KHÔNG làm phẳng spike lớn.
    """

    def __init__(self, smooth_window: int = 3):
        self.smooth_window = smooth_window
        self.profile_      = None
        self.hist_         = None

    def fit(self, sales: pd.DataFrame):
        self.hist_ = sales.copy()
        df = sales.copy()
        df["doy"] = df["Date"].dt.dayofyear

        # Robust: median thay vì mean để giảm outlier
        profile = (
            df.groupby("doy")[["Revenue", "COGS"]]
            .median()
            .reset_index()
        )

        # Đảm bảo đủ 366 dòng (bù ngày lễ nhuận)
        full_doy = pd.DataFrame({"doy": np.arange(1, 367)})
        profile  = full_doy.merge(profile, on="doy", how="left")

        # Điền NaN bằng interpolation + fill với median
        profile["Revenue"] = profile["Revenue"].interpolate().bfill().ffill()
        profile["COGS"]    = profile["COGS"].interpolate().bfill().ffill()

        # Smooth NHẸ window=3 – chỉ giảm noise, KHÔNG flatten spikes
        if self.smooth_window and self.smooth_window > 1:
            rev_s = profile["Revenue"].rolling(self.smooth_window, min_periods=1, center=True).mean()
            cog_s = profile["COGS"].rolling(self.smooth_window, min_periods=1, center=True).mean()
            # Blend 50/50 – giữ lại biên độ spike
            profile["Revenue"] = 0.5 * rev_s + 0.5 * profile["Revenue"]
            profile["COGS"]    = 0.5 * cog_s + 0.5 * profile["COGS"]

        self.profile_ = profile.set_index("doy")
        return self

    def predict(self, dates: pd.Series) -> pd.DataFrame:
        results = []
        for d in dates:
            doy = d.dayofyear
            if doy not in self.profile_.index:
                doy = min(doy, 365)

            rev_base = float(self.profile_.loc[doy, "Revenue"])
            cog_base = float(self.profile_.loc[doy, "COGS"])

            # Scale trend
            scale_rev = recent_trend_scale(self.hist_, "Revenue", d)
            scale_cog = recent_trend_scale(self.hist_, "COGS",    d)

            results.append({
                "Date"   : d,
                "Revenue": rev_base * scale_rev,
                "COGS"   : cog_base * scale_cog,
            })
        return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL C — CALENDAR PROFILE (month×dow + weekofyear×dow)
# ══════════════════════════════════════════════════════════════════════════════

class CalendarProfileModel:
    """
    Kết hợp 2 seasonal groupings:
      - (month, dayofweek) — pattern tháng × thứ
      - (weekofyear, dayofweek) — pattern tuần trong năm × thứ

    Final = 0.5 * month_dow_profile + 0.5 * weekofyear_dow_profile
    """

    def fit(self, sales: pd.DataFrame):
        df = sales.copy()
        df["month"]  = df["Date"].dt.month
        df["dow"]    = df["Date"].dt.dayofweek
        df["woy"]    = df["Date"].dt.isocalendar().week.astype(int)

        self.md_ = df.groupby(["month", "dow"])[["Revenue", "COGS"]].median().reset_index()
        self.wd_ = df.groupby(["woy",   "dow"])[["Revenue", "COGS"]].median().reset_index()

        # Global medians làm fallback
        self.global_rev_ = df["Revenue"].median()
        self.global_cog_ = df["COGS"].median()

        self.hist_ = df
        return self

    def _lookup(self, table: pd.DataFrame, keys: tuple, col: str, fallback: float):
        k1, k2 = keys
        a, b   = table.columns[0], table.columns[1]
        row = table[(table[a] == k1) & (table[b] == k2)]
        if len(row) > 0:
            return float(row[col].values[0])
        return fallback

    def predict(self, dates: pd.Series) -> pd.DataFrame:
        results = []
        for d in dates:
            month = d.month
            dow   = d.dayofweek
            woy   = int(d.isocalendar().week)

            r_md = self._lookup(self.md_, (month, dow), "Revenue", self.global_rev_)
            c_md = self._lookup(self.md_, (month, dow), "COGS",    self.global_cog_)
            r_wd = self._lookup(self.wd_, (woy,   dow), "Revenue", self.global_rev_)
            c_wd = self._lookup(self.wd_, (woy,   dow), "COGS",    self.global_cog_)

            rev = 0.5 * r_md + 0.5 * r_wd
            cog = 0.5 * c_md + 0.5 * c_wd

            # Trend scale
            scale_rev = recent_trend_scale(self.hist_, "Revenue", d)
            scale_cog = recent_trend_scale(self.hist_, "COGS",    d)

            results.append({
                "Date"   : d,
                "Revenue": rev * scale_rev,
                "COGS"   : cog * scale_cog,
            })
        return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL D — RECENT-YEAR WEIGHTED (2019-2022 weighted average by doy)
# ══════════════════════════════════════════════════════════════════════════════

class RecentYearWeightedModel:
    """
    Với mỗi forecast date, lấy cùng dayofyear của các năm gần nhất.
    Weight giảm dần theo khoảng cách năm: 2022 > 2021 > 2020 > 2019.
    """

    def __init__(self, year_weights: dict = None):
        # Mặc định: ưu tiên năm gần nhất
        self.year_weights = year_weights or {
            2022: 0.45,
            2021: 0.30,
            2020: 0.15,
            2019: 0.10,
        }
        self._idx_ = {}

    def fit(self, sales: pd.DataFrame):
        self.hist_ = sales.copy()
        # Index (year, doy) → (rev, cog) – search ±3 ngày
        for _, row in sales.iterrows():
            key = (row["Date"].year, row["Date"].dayofyear)
            self._idx_[key] = (row["Revenue"], row["COGS"])
        return self

    def _get_val(self, year: int, doy: int):
        for delta in range(0, 4):
            for sign in ([0] if delta == 0 else [1, -1]):
                key = (year, doy + sign * delta)
                if key in self._idx_:
                    return self._idx_[key]
        return None

    def predict(self, dates: pd.Series) -> pd.DataFrame:
        results = []
        for d in dates:
            doy = d.dayofyear

            rev_sum, cog_sum, w_sum = 0.0, 0.0, 0.0
            for yr, w in self.year_weights.items():
                v = self._get_val(yr, doy)
                if v is not None:
                    rev_sum += w * v[0]
                    cog_sum += w * v[1]
                    w_sum   += w

            if w_sum > 0:
                rev = rev_sum / w_sum
                cog = cog_sum / w_sum
            else:
                rev = self.hist_["Revenue"].median()
                cog = self.hist_["COGS"].median()

            # Trend scale
            scale_rev = recent_trend_scale(self.hist_, "Revenue", d)
            scale_cog = recent_trend_scale(self.hist_, "COGS",    d)

            results.append({
                "Date"   : d,
                "Revenue": rev * scale_rev,
                "COGS"   : cog * scale_cog,
            })
        return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL E — LIGHTGBM CALENDAR-ONLY (KHÔNG LAG/ROLLING)
# ══════════════════════════════════════════════════════════════════════════════

def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    CHỈ feature dựa trên Date, KHÔNG dùng lag/rolling của target.
    → Có thể apply trực tiếp cho future mà không bị drift.
    """
    df = df.copy()
    df["year"]       = df["Date"].dt.year
    df["month"]      = df["Date"].dt.month
    df["day"]        = df["Date"].dt.day
    df["dayofweek"]  = df["Date"].dt.dayofweek
    df["dayofyear"]  = df["Date"].dt.dayofyear
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["quarter"]    = df["Date"].dt.quarter
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["is_month_end"]   = df["Date"].dt.is_month_end.astype(int)

    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["doy_sin"]    = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"]    = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    # Trend index (ngày kể từ start)
    df["trend"] = (df["Date"] - pd.Timestamp("2012-01-01")).dt.days

    return df


class LGBMCalendarModel:
    """
    LightGBM với CHỈ calendar features (no lag).
    - log1p target để stabilize
    - Early stopping với time-based validation
    - Retrain full data với best_iter
    """

    def __init__(self):
        self.model_ = None
        self.feat_  = None

    def _build(self) -> LGBMRegressor:
        return LGBMRegressor(
            n_estimators      = 3000,
            learning_rate     = 0.02,
            num_leaves        = 63,
            max_depth         = 7,
            min_child_samples = 30,
            subsample         = 0.8,
            subsample_freq    = 1,
            colsample_bytree  = 0.8,
            reg_alpha         = 0.1,
            reg_lambda        = 0.5,
            min_gain_to_split = 0.001,
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
            verbose           = -1,
        )

    def fit(self, sales: pd.DataFrame, target: str, valid_start=VALID_START, valid_end=VALID_END):
        df = _calendar_features(sales[["Date", target]].copy())
        self.feat_ = [c for c in df.columns if c not in ["Date", target]]

        train = df[df["Date"] <  valid_start]
        valid = df[(df["Date"] >= valid_start) & (df["Date"] <= valid_end)]

        m = self._build()
        m.fit(
            train[self.feat_], np.log1p(train[target]),
            eval_set  = [(valid[self.feat_], np.log1p(valid[target]))],
            callbacks = [early_stopping(100, verbose=False),
                         log_evaluation(-1)],
        )

        best_iter = m.best_iteration_ or 500
        # Retrain toàn bộ với best_iter để có model cuối
        full = self._build()
        full.set_params(n_estimators=int(best_iter * 1.1))
        full.fit(df[self.feat_], np.log1p(df[target]))

        self.model_ = full
        self._target_ = target
        return self

    def predict_val(self, sales: pd.DataFrame, valid_start=VALID_START, valid_end=VALID_END):
        """Predict trên validation (bằng model đã fit trên pre-valid data)."""
        df = _calendar_features(sales[["Date", self._target_]].copy())
        val = df[(df["Date"] >= valid_start) & (df["Date"] <= valid_end)]
        # Note: model hiện tại đã retrain toàn bộ → dùng model kia cho valid thì leak
        # → Tạo lại model CHỈ train trên pre-valid để predict_val công bằng
        train = df[df["Date"] < valid_start]
        m = self._build()
        m.fit(train[self.feat_], np.log1p(train[self._target_]),
              eval_set  = [(val[self.feat_], np.log1p(val[self._target_]))],
              callbacks = [early_stopping(100, verbose=False),
                           log_evaluation(-1)])
        preds = np.expm1(m.predict(val[self.feat_]))
        return np.maximum(preds, 0), val["Date"].values

    def predict(self, dates: pd.Series) -> pd.DataFrame:
        df = _calendar_features(pd.DataFrame({"Date": dates}))
        preds = np.expm1(self.model_.predict(df[self.feat_]))
        preds = np.maximum(preds, 0)
        return pd.DataFrame({"Date": dates.values, self._target_: preds})


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE WEIGHT SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def grid_search_weights(preds_dict: dict, y_true: np.ndarray, step: float = 0.1) -> tuple:
    """
    Grid search 5 weights sum=1 tối ưu RMSE.
    Step 0.1 → 1001 combinations (5-simplex, manageable).
    """
    keys  = list(preds_dict.keys())
    steps = np.arange(0, 1.0 + 1e-9, step)

    best_rmse = np.inf
    best_w    = {k: 1/len(keys) for k in keys}

    for combo in product(steps, repeat=len(keys)):
        s = sum(combo)
        if abs(s - 1.0) > 1e-6:
            continue
        pred = np.zeros_like(y_true, dtype=float)
        for k, w in zip(keys, combo):
            pred += w * preds_dict[k]
        pred = np.maximum(pred, 0)
        r = rmse(y_true, pred)
        if r < best_rmse:
            best_rmse = r
            best_w    = dict(zip(keys, combo))

    return best_w, best_rmse


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def apply_constraints(rev: np.ndarray, cog: np.ndarray):
    """Business rules: Revenue >= 0, COGS >= 0, COGS <= Revenue * 0.95"""
    rev = np.maximum(rev, 0)
    cog = np.maximum(cog, 0)

    # Chỉ cap khi THỰC SỰ vi phạm — không cap khi dưới threshold
    violation = cog > rev * COGS_CAP
    cog = np.where(violation, rev * COGS_CAP, cog)

    return rev, cog


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  DATATHON 2026 – Ensemble v4 (seasonal-heavy, no recursive ML)")
    print("=" * 70)

    # ── 1. Load ───────────────────────────────────────────────
    sales_path  = os.path.join(DATA_PATH, "sales.csv")
    sample_path = os.path.join(DATA_PATH, "sample_submission.csv")
    if not os.path.exists(sales_path):
        raise FileNotFoundError(sales_path)
    if not os.path.exists(sample_path):
        raise FileNotFoundError(sample_path)

    sales  = pd.read_csv(sales_path,  parse_dates=["Date"])
    sample = pd.read_csv(sample_path, parse_dates=["Date"])

    original_order = sample["Date"].copy()

    sales  = sales.sort_values("Date").reset_index(drop=True)
    sample = sample.sort_values("Date").reset_index(drop=True)

    # ── 2. Sanity check ──────────────────────────────────────
    print(f"\n  sales.csv  : {sales.shape}  {sales['Date'].min().date()} → {sales['Date'].max().date()}")
    print(f"  sample.csv : {sample.shape}  {sample['Date'].min().date()} → {sample['Date'].max().date()}")
    print(f"\n  Revenue — min: {sales['Revenue'].min():>12,.0f}  max: {sales['Revenue'].max():>12,.0f}  mean: {sales['Revenue'].mean():>12,.0f}")
    print(f"  COGS    — min: {sales['COGS'].min():>12,.0f}  max: {sales['COGS'].max():>12,.0f}  mean: {sales['COGS'].mean():>12,.0f}")

    nan_count = sales[["Revenue", "COGS"]].isnull().sum().sum()
    inf_count = np.isinf(sales[["Revenue", "COGS"]].values).sum()
    if nan_count or inf_count:
        print(f"  ⚠️  NaN={nan_count}  Inf={inf_count} in train — will be left as-is")

    viol = (sales["COGS"] > sales["Revenue"]).sum()
    if viol > 0:
        print(f"  ⚠️  {viol} ngày train có COGS > Revenue (giữ nguyên, không xoá)")

    # ── 3. Split ─────────────────────────────────────────────
    train_internal = sales[sales["Date"] <  VALID_START].copy()
    valid_df       = sales[(sales["Date"] >= VALID_START) & (sales["Date"] <= VALID_END)].copy()

    print(f"\n  Internal train : {len(train_internal):,} rows (< {VALID_START})")
    print(f"  Validation     : {len(valid_df):,} rows ({VALID_START} → {VALID_END})")

    val_dates = valid_df["Date"]
    y_rev     = valid_df["Revenue"].values
    y_cog     = valid_df["COGS"].values

    # ── 4. Fit 5 models trên TRAIN INTERNAL → predict VALIDATION ──
    print("\n  ═══ Fitting models on INTERNAL train (for weight search) ═══")

    print("  [A] LastYearModel …")
    a = LastYearModel().fit(train_internal)
    a_val = a.predict(val_dates)

    print("  [B] DayOfYearModel …")
    b = DayOfYearModel(smooth_window=3).fit(train_internal)
    b_val = b.predict(val_dates)

    print("  [C] CalendarProfileModel …")
    c = CalendarProfileModel().fit(train_internal)
    c_val = c.predict(val_dates)

    print("  [D] RecentYearWeightedModel …")
    d = RecentYearWeightedModel().fit(train_internal)
    d_val = d.predict(val_dates)

    print("  [E] LGBMCalendarModel Revenue …")
    e_rev = LGBMCalendarModel().fit(sales, "Revenue")   # fit cả val để có best_iter, nhưng predict_val dùng model train-only
    e_rev_val_preds, _ = e_rev.predict_val(sales)

    print("  [E] LGBMCalendarModel COGS …")
    e_cog = LGBMCalendarModel().fit(sales, "COGS")
    e_cog_val_preds, _ = e_cog.predict_val(sales)

    # ── 5. Validation metrics + weight search ────────────────
    print("\n  ═══ Validation Metrics – Revenue ═══")
    rev_val_preds = {
        "A": safe_array(a_val["Revenue"].values),
        "B": safe_array(b_val["Revenue"].values),
        "C": safe_array(c_val["Revenue"].values),
        "D": safe_array(d_val["Revenue"].values),
        "E": safe_array(e_rev_val_preds),
    }
    for k, v in rev_val_preds.items():
        print_metrics(f"[{k}] individual", y_rev, v)

    print("\n  ═══ Validation Metrics – COGS ═══")
    cog_val_preds = {
        "A": safe_array(a_val["COGS"].values),
        "B": safe_array(b_val["COGS"].values),
        "C": safe_array(c_val["COGS"].values),
        "D": safe_array(d_val["COGS"].values),
        "E": safe_array(e_cog_val_preds),
    }
    for k, v in cog_val_preds.items():
        print_metrics(f"[{k}] individual", y_cog, v)

    print("\n  ═══ Grid-searching ensemble weights …")
    w_rev, rmse_rev = grid_search_weights(rev_val_preds, y_rev, step=0.1)
    w_cog, rmse_cog = grid_search_weights(cog_val_preds, y_cog, step=0.1)

    def fmt_w(w):
        return "  ".join(f"{k}={w[k]:.2f}" for k in w)

    print(f"\n  Best Revenue weights : {fmt_w(w_rev)}   RMSE={rmse_rev:,.1f}")
    print(f"  Best COGS    weights : {fmt_w(w_cog)}   RMSE={rmse_cog:,.1f}")

    # Ensemble validation
    ens_rev = sum(w_rev[k] * rev_val_preds[k] for k in w_rev)
    ens_cog = sum(w_cog[k] * cog_val_preds[k] for k in w_cog)

    print("\n  ═══ Ensemble Validation ═══")
    print_metrics("Ensemble Revenue", y_rev, ens_rev)
    print_metrics("Ensemble COGS",    y_cog, ens_cog)

    # ── 6. Refit các seasonal model trên FULL sales, predict TEST ──
    print("\n  ═══ Refitting all models on FULL sales → predicting test ═══")

    print("  [A] LastYearModel full …")
    a_full = LastYearModel().fit(sales)

    print("  [B] DayOfYearModel full …")
    b_full = DayOfYearModel(smooth_window=3).fit(sales)

    print("  [C] CalendarProfileModel full …")
    c_full = CalendarProfileModel().fit(sales)

    print("  [D] RecentYearWeightedModel full …")
    d_full = RecentYearWeightedModel().fit(sales)

    # Model E đã fit trên full sales ở bước trên, dùng luôn
    test_dates = sample["Date"]

    a_test = a_full.predict(test_dates)
    b_test = b_full.predict(test_dates)
    c_test = c_full.predict(test_dates)
    d_test = d_full.predict(test_dates)
    e_rev_test = e_rev.predict(test_dates)
    e_cog_test = e_cog.predict(test_dates)

    rev_test_preds = {
        "A": safe_array(a_test["Revenue"].values),
        "B": safe_array(b_test["Revenue"].values),
        "C": safe_array(c_test["Revenue"].values),
        "D": safe_array(d_test["Revenue"].values),
        "E": safe_array(e_rev_test["Revenue"].values),
    }
    cog_test_preds = {
        "A": safe_array(a_test["COGS"].values),
        "B": safe_array(b_test["COGS"].values),
        "C": safe_array(c_test["COGS"].values),
        "D": safe_array(d_test["COGS"].values),
        "E": safe_array(e_cog_test["COGS"].values),
    }

    final_rev = sum(w_rev[k] * rev_test_preds[k] for k in w_rev)
    final_cog = sum(w_cog[k] * cog_test_preds[k] for k in w_cog)

    # ── 7. Business constraints ──────────────────────────────
    final_rev, final_cog = apply_constraints(final_rev, final_cog)

    violations = int(np.sum(final_cog > final_rev))
    print(f"\n  COGS > Revenue violations after constraint: {violations}")

    # ── 8. Sanity check: prediction không được quá phẳng ────
    recent_seasonal_max = sales[sales["Date"] >= sales["Date"].max() - pd.DateOffset(years=2)]["Revenue"].quantile(0.95)
    print(f"\n  📊 Prediction stats:")
    print(f"     Revenue  pred : min={final_rev.min():>12,.0f}  max={final_rev.max():>12,.0f}  mean={final_rev.mean():>12,.0f}")
    print(f"     COGS     pred : min={final_cog.min():>12,.0f}  max={final_cog.max():>12,.0f}  mean={final_cog.mean():>12,.0f}")
    print(f"     Train recent Revenue p95 (last 2y): {recent_seasonal_max:,.0f}")

    if final_rev.max() < recent_seasonal_max * 0.5:
        print(f"  ⚠️  WARNING: Revenue prediction max quá thấp so với seasonal max gần đây!")
        print(f"     Có thể vẫn bị flat. Xem lại weights.")
    else:
        print(f"  ✅ Prediction max có vẻ hợp lý với seasonal range lịch sử.")

    # ── 9. Xuất submission ───────────────────────────────────
    result = pd.DataFrame({
        "Date"   : sample["Date"].values,
        "Revenue": final_rev,
        "COGS"   : final_cog,
    })

    # Khôi phục thứ tự gốc của sample_submission.csv
    submission = pd.DataFrame({"Date": original_order.values}).merge(result, on="Date", how="left")
    submission = submission[["Date", "Revenue", "COGS"]]

    # Safety checks
    assert not submission.isnull().any().any(), "Submission có NaN!"
    assert not np.isinf(submission[["Revenue", "COGS"]].values).any(), "Submission có inf!"
    assert len(submission) == len(sample), "Số rows submission không khớp sample!"

    submission.to_csv(OUTPUT_FILE, index=False)

    print(f"\n  💾 Saved → {OUTPUT_FILE}  ({len(submission)} rows)")
    print("\n  First 5:")
    print(submission.head().to_string(index=False))
    print("\n  Last 5:")
    print(submission.tail().to_string(index=False))

    print(f"\n{'=' * 70}")
    print("  ✅  Done! Nộp file: submission.csv")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()