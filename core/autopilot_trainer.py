# autopilot_trainer.py
# Practical-scale AutoPilot for Dairy Cow OS
# - Reads dairy_cow_management.xlsx (cows/daily/events)
# - Builds labels from events (heuristics; improves as you log more)
# - Generates strong time-series features
# - Trains calibrated probability models: mastitis/ketosis/estrus
# - Evaluates (AUC/Brier/ECE) and only promotes if improved
# - Versions models, supports rollback
# - Scores today's herd and exports:
#     outputs/scored_today.csv (all cows)
#     outputs/alerts_today.csv (threshold exceed)
#     outputs/todo_today.csv (top K)
#     outputs/metrics_latest.json
# - Updates plugins/ai_recommend.py (UI tab for app_main.py)
#
# Run once:
#   python autopilot_trainer.py --run-once
#
# Keep running (power-on mode):
#   python autopilot_trainer.py --loop --minutes 30

import os, json, time, math, argparse, shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

from excel_io import read_sheet_df

# ---------------- CONFIG ----------------
@dataclass
class CFG:
    xlsx_path: str = "dairy_cow_management.xlsx"

    out_dir: str = "outputs"
    models_dir: str = "models"
    history_dir: str = "models/history"

    # output files
    scored_today_csv: str = "outputs/scored_today.csv"
    alerts_today_csv: str = "outputs/alerts_today.csv"
    todo_today_csv: str = "outputs/todo_today.csv"
    metrics_json: str = "outputs/metrics_latest.json"

    # model files
    current_model: str = "models/current.joblib"

    # thresholds (practical starting points; tune later)
    thr_mastitis: float = 0.70
    thr_ketosis: float = 0.65
    thr_estrus: float = 0.75

    # todo list size
    top_k: int = 30

    # minimum training data
    min_pos: int = 20
    min_total: int = 800

    # keep last N model versions
    keep_history: int = 12

    # training
    random_seed: int = 42

    # acceptance rule (must improve at least one metric without hurting others badly)
    # AUC higher better, Brier/ECE lower better
    auc_gain_min: float = 0.005
    brier_drop_min: float = 0.001
    ece_drop_min: float = 0.002

CFG0 = CFG()

RNG = np.random.default_rng(CFG0.random_seed)

TASKS = ("mastitis", "ketosis", "estrus")

# ---------------- metrics ----------------
def ece_score(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = p[mask].mean()
        ece += (mask.sum() / len(p)) * abs(acc - conf)
    return float(ece)

def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

# ---------------- label building (events -> y) ----------------
def _norm(s: str) -> str:
    return str(s).lower().strip()

def build_labels_from_events(daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Creates heuristic labels on daily rows.
    Improves as you log:
      - mastitis: treatment / mastitis_check with detail indicating mastitis
      - ketosis : ketone_test positive / medication detail including ketosis
      - estrus  : insemination event (estrus day proxy)
    This is practical: initial bootstrapping then you refine event types/details.
    """
    df = daily.copy()
    df["date"] = df["date"].astype(str)
    df["cow_id"] = df["cow_id"].astype(str)

    # default unknown -> 0, but we also track "label_source"
    df["mastitis_y"] = 0
    df["ketosis_y"] = 0
    df["estrus_y"] = 0

    if events is None or events.empty:
        return df

    ev = events.copy()
    ev["date"] = ev["date"].astype(str)
    ev["cow_id"] = ev["cow_id"].astype(str)
    ev["event_type"] = ev["event_type"].astype(str).map(_norm)
    ev["detail"] = ev["detail"].astype(str).map(_norm)

    # Mastitis: if event_type mentions mastitis or detail includes mastitis keywords
    mast_mask = (
        ev["event_type"].isin(["mastitis_check", "mastitis", "treatment", "medication", "vet_visit"]) &
        (ev["detail"].str.contains("mastitis") | ev["detail"].str.contains("乳房炎") | ev["detail"].str.contains("cmt") | ev["detail"].str.contains("udder"))
    )
    mast = ev[mast_mask][["date","cow_id"]].drop_duplicates()

    # Ketosis: ketone_test positive, or detail includes ketosis keywords
    keto_mask = (
        ev["event_type"].isin(["ketone_test", "ketosis", "treatment", "medication", "vet_visit"]) &
        (ev["detail"].str.contains("ketosis") | ev["detail"].str.contains("ケト") | ev["detail"].str.contains("bhb") | ev["detail"].str.contains("acetone"))
    )
    keto = ev[keto_mask][["date","cow_id"]].drop_duplicates()

    # Estrus: insemination is a strong proxy for estrus day
    estr = ev[ev["event_type"].isin(["insemination"])][["date","cow_id"]].drop_duplicates()

    # merge back
    df = df.merge(mast.assign(mastitis_y=1), on=["date","cow_id"], how="left")
    df["mastitis_y"] = df["mastitis_y_y"].fillna(df["mastitis_y_x"]).astype(int)
    df.drop(columns=[c for c in df.columns if c.endswith("_x") or c.endswith("_y")], inplace=True, errors="ignore")

    # after drop cleanup, remerge keto and estrus carefully
    # (simple: re-merge onto fresh copy)
    df2 = df.copy()
    df2 = df2.merge(keto.assign(ketosis_y=1), on=["date","cow_id"], how="left")
    df2["ketosis_y"] = df2["ketosis_y_y"].fillna(df2["ketosis_y_x"]).astype(int)
    df2.drop(columns=[c for c in df2.columns if c.endswith("_x") or c.endswith("_y")], inplace=True, errors="ignore")

    df3 = df2.copy()
    df3 = df3.merge(estr.assign(estrus_y=1), on=["date","cow_id"], how="left")
    df3["estrus_y"] = df3["estrus_y_y"].fillna(df3["estrus_y_x"]).astype(int)
    df3.drop(columns=[c for c in df3.columns if c.endswith("_x") or c.endswith("_y")], inplace=True, errors="ignore")

    # ensure cols exist
    for c in ["mastitis_y","ketosis_y","estrus_y"]:
        if c not in df3.columns:
            df3[c] = 0
        df3[c] = df3[c].fillna(0).astype(int)

    return df3

# ---------------- feature engineering (practical & strong) ----------------
def add_timeseries_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds robust features that scale to real farms:
    - rolling means, stds
    - deltas vs rolling baseline
    - short-term trend
    """
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"])
    x = x.sort_values(["cow_id","date"])

    base_cols = [
        "milk_yield_kg","milking_count","milking_minutes",
        "activity_neck","activity_leg","rumination_min","conductivity",
        "temp_c","rh","thi"
    ]
    for c in base_cols:
        if c not in x.columns:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")

    g = x.groupby("cow_id", group_keys=False)

    # rolling windows
    for c in ["milk_yield_kg","rumination_min","activity_neck","activity_leg","conductivity","thi"]:
        x[f"{c}_r3_mean"] = g[c].transform(lambda s: s.rolling(3, min_periods=1).mean())
        x[f"{c}_r7_mean"] = g[c].transform(lambda s: s.rolling(7, min_periods=1).mean())
        x[f"{c}_r14_mean"] = g[c].transform(lambda s: s.rolling(14, min_periods=1).mean())
        x[f"{c}_r7_std"]  = g[c].transform(lambda s: s.rolling(7, min_periods=2).std()).fillna(0)

        # deltas: today - rolling mean
        x[f"{c}_d_r7"] = x[c] - x[f"{c}_r7_mean"]
        x[f"{c}_d_r3"] = x[c] - x[f"{c}_r3_mean"]

        # 1-day change
        x[f"{c}_d1"] = g[c].transform(lambda s: s.diff(1)).fillna(0)

        # 3-day trend slope (simple)
        x[f"{c}_trend3"] = g[c].transform(lambda s: s.diff(3)).fillna(0)

    # combine activity
    x["activity_sum"] = x["activity_neck"].fillna(0) + x["activity_leg"].fillna(0)
    x["activity_sum_r7"] = g["activity_sum"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    x["activity_sum_d_r7"] = x["activity_sum"] - x["activity_sum_r7"]

    # missingness flags (helps practical data)
    for c in base_cols:
        x[f"{c}_isna"] = x[c].isna().astype(int)

    # fill numeric NaN with group median then global median
    for c in x.columns:
        if c in ["cow_id","date"]:
            continue
        if pd.api.types.is_numeric_dtype(x[c]):
            x[c] = g[c].transform(lambda s: s.fillna(s.median()))
            x[c] = x[c].fillna(x[c].median())

    # return with date string
    x["date"] = x["date"].dt.date.astype(str)
    return x

def select_feature_columns(df: pd.DataFrame) -> List[str]:
    # all numeric except labels and identifiers
    drop = {"cow_id","date","notes","mastitis_y","ketosis_y","estrus_y"}
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

# ---------------- training ----------------
def time_split(df: pd.DataFrame, test_days: int = 28) -> Tuple[pd.DataFrame,pd.DataFrame]:
    d = df.copy()
    d["date_dt"] = pd.to_datetime(d["date"])
    maxd = d["date_dt"].max()
    cutoff = maxd - pd.Timedelta(days=test_days)
    train = d[d["date_dt"] <= cutoff].copy()
    test  = d[d["date_dt"] > cutoff].copy()
    train.drop(columns=["date_dt"], inplace=True)
    test.drop(columns=["date_dt"], inplace=True)
    return train, test

def train_one_task(df: pd.DataFrame, task: str, feat_cols: List[str], seed: int) -> Tuple[object, Dict]:
    ycol = f"{task}_y"
    train_df, test_df = time_split(df, test_days=28)

    # guardrails
    y_train = train_df[ycol].values.astype(int)
    y_test  = test_df[ycol].values.astype(int)

    stats = {
        "task": task,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_pos": int(y_train.sum()),
        "test_pos": int(y_test.sum()),
    }

    if len(train_df) < CFG0.min_total or y_train.sum() < CFG0.min_pos or len(np.unique(y_train)) < 2:
        stats["status"] = "skipped_insufficient_data"
        return None, stats
    if len(test_df) < 100 or len(np.unique(y_test)) < 2:
        stats["status"] = "skipped_insufficient_test"
        return None, stats

    X_train = train_df[feat_cols].values
    X_test  = test_df[feat_cols].values

    base = HistGradientBoostingClassifier(
        random_state=seed,
        max_depth=6,
        learning_rate=0.07,
        max_iter=400
    )
    # calibration for trustworthy probabilities
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, y_train)

    p = model.predict_proba(X_test)[:, 1]

    stats["auc"] = float(roc_auc_score(y_test, p))
    stats["brier"] = float(brier_score_loss(y_test, p))
    stats["ece"] = float(ece_score(y_test, p))
    stats["status"] = "trained"
    return model, stats

# ---------------- model selection / versioning ----------------
def load_current_bundle(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

def should_promote(new_metrics: Dict, old_metrics: Optional[Dict]) -> bool:
    if old_metrics is None:
        return True

    # Compare per-task with conservative rule:
    # Promote if at least one task improves meaningfully and none catastrophically worse.
    improved = False
    for t in TASKS:
        nm = new_metrics.get(t, {})
        om = old_metrics.get(t, {})
        if nm.get("status") != "trained" or om.get("status") != "trained":
            # if old was missing, allow
            if nm.get("status") == "trained" and om.get("status") != "trained":
                improved = True
            continue

        auc_gain = nm["auc"] - om["auc"]
        brier_drop = om["brier"] - nm["brier"]
        ece_drop = om["ece"] - nm["ece"]

        if (auc_gain >= CFG0.auc_gain_min) or (brier_drop >= CFG0.brier_drop_min) or (ece_drop >= CFG0.ece_drop_min):
            improved = True

        # catastrophic guard: avoid big degradation
        if auc_gain < -0.02 or brier_drop < -0.01 or ece_drop < -0.02:
            return False

    return improved

def save_bundle(bundle: dict):
    safe_mkdir(CFG0.models_dir)
    safe_mkdir(CFG0.history_dir)
    joblib.dump(bundle, CFG0.current_model)

    # version copy
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    vpath = os.path.join(CFG0.history_dir, f"bundle_{ts}.joblib")
    joblib.dump(bundle, vpath)

    # cleanup old versions
    versions = sorted([p for p in os.listdir(CFG0.history_dir) if p.endswith(".joblib")])
    if len(versions) > CFG0.keep_history:
        for p in versions[:len(versions) - CFG0.keep_history]:
            try:
                os.remove(os.path.join(CFG0.history_dir, p))
            except Exception:
                pass

# ---------------- scoring & outputs ----------------
def score_today(bundle: dict, df_feat: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    latest_date = df_feat["date"].astype(str).max()
    today = df_feat[df_feat["date"].astype(str) == latest_date].copy()
    X = today[feat_cols].values

    for t in TASKS:
        model = bundle["models"].get(t)
        if model is None:
            today[f"p_{t}"] = 0.0
        else:
            today[f"p_{t}"] = model.predict_proba(X)[:, 1]

    # combined priority score (practical default; later profit-based)
    today["priority_score"] = (
        today["p_mastitis"] * 1.2 +
        today["p_ketosis"]  * 1.0 +
        today["p_estrus"]   * 0.9
    )

    # simple recommended action text
    def action_row(r):
        actions = []
        if r["p_mastitis"] >= CFG0.thr_mastitis:
            actions.append("乳房炎疑い：乳房確認→CMT/導電率→必要なら治療")
        if r["p_ketosis"] >= CFG0.thr_ketosis:
            actions.append("ケト疑い：採食/反芻確認→ケトン検査→補液/投薬検討")
        if r["p_estrus"] >= CFG0.thr_estrus:
            actions.append("発情候補：活動上昇確認→観察→授精判断")
        if not actions:
            actions.append("経過観察")
        return " / ".join(actions)

    today["recommended_action"] = today.apply(action_row, axis=1)

    # short reason (human-readable)
    reason_cols = []
    for c in ["milk_yield_kg_d_r7","rumination_min_d_r7","conductivity_d_r7","thi"]:
        if c in today.columns:
            reason_cols.append(c)

    def why_row(r):
        parts = []
        if "milk_yield_kg_d_r7" in r:
            if r["milk_yield_kg_d_r7"] < -2:
                parts.append(f"乳量↓({r['milk_yield_kg_d_r7']:.1f})")
        if "rumination_min_d_r7" in r:
            if r["rumination_min_d_r7"] < -30:
                parts.append(f"反芻↓({r['rumination_min_d_r7']:.0f})")
        if "conductivity_d_r7" in r:
            if r["conductivity_d_r7"] > 0.3:
                parts.append(f"導電率↑(+{r['conductivity_d_r7']:.2f})")
        if "thi" in r and r["thi"] >= 72:
            parts.append(f"暑熱(THI {r['thi']:.0f})")
        return " / ".join(parts) if parts else "-"

    today["why_short"] = today.apply(why_row, axis=1)

    return today

def export_outputs(scored: pd.DataFrame):
    safe_mkdir(CFG0.out_dir)

    scored_out = scored.copy()
    cols_keep = ["date","cow_id","p_mastitis","p_ketosis","p_estrus","priority_score","recommended_action","why_short"]
    base_show = [c for c in cols_keep if c in scored_out.columns]
    scored_out = scored_out[base_show].sort_values("priority_score", ascending=False)

    scored_out.to_csv(CFG0.scored_today_csv, index=False, encoding="utf-8")

    alerts = scored_out[
        (scored_out["p_mastitis"] >= CFG0.thr_mastitis) |
        (scored_out["p_ketosis"]  >= CFG0.thr_ketosis) |
        (scored_out["p_estrus"]   >= CFG0.thr_estrus)
    ].copy()
    alerts.to_csv(CFG0.alerts_today_csv, index=False, encoding="utf-8")

    todo = scored_out.head(CFG0.top_k).copy()
    todo.to_csv(CFG0.todo_today_csv, index=False, encoding="utf-8")

# ---------------- plugin writer ----------------
PLUGIN_PATH = os.path.join("plugins", "ai_recommend.py")

PLUGIN_TEMPLATE = r'''# plugins/ai_recommend.py
# Auto-generated by autopilot_trainer.py
# Shows today's prioritized cows + one-click actions (writes to Excel events sheet)

import pandas as pd
import streamlit as st
from datetime import datetime, date
from excel_io import append_row, read_sheet_df

NAME = "🤖 AIおすすめ（今日の優先牛）"

XLSX_PATH = "{xlsx_path}"
SCORED_PATH = "{scored_path}"
TODO_PATH = "{todo_path}"
ALERTS_PATH = "{alerts_path}"

def run(ctx):
  st.subheader("🤖 今日の優先牛（AI）")
  st.caption("AutoPilotが毎回スコアリングして更新します。ボタンでeventsに即記録できます。")

  # load scored/todo/alerts
  def _safe_read(p):
    try:
      return pd.read_csv(p)
    except Exception:
      return pd.DataFrame()

  scored = _safe_read(SCORED_PATH)
  todo = _safe_read(TODO_PATH)
  alerts = _safe_read(ALERTS_PATH)

  if scored.empty:
    st.warning("まだスコアがありません。先に python autopilot_trainer.py --run-once を実行してください。")
    return

  c1, c2, c3 = st.columns(3)
  with c1:
    st.metric("全頭スコア", int(len(scored)))
  with c2:
    st.metric("要対応（alerts）", int(len(alerts)) if not alerts.empty else 0)
  with c3:
    st.metric("優先（todo）", int(len(todo)) if not todo.empty else 0)

  st.markdown("### 🚨 要対応（alerts）")
  if alerts.empty:
    st.info("閾値を超えた牛はいません。")
  else:
    st.dataframe(alerts, use_container_width=True, height=260)

  st.markdown("### ✅ 今日の優先（todo）")
  st.dataframe(todo if not todo.empty else scored.head(30), use_container_width=True, height=320)

  st.divider()
  st.markdown("## 🧷 ワンクリック記録（eventsへ）")

  cows_df = read_sheet_df(XLSX_PATH, "cows")
  cow_ids = sorted([str(x) for x in cows_df["cow_id"].dropna().tolist()]) if not cows_df.empty else []
  default_cow = cow_ids[0] if cow_ids else ""

  colA, colB, colC = st.columns([2,2,3])
  with colA:
    cow_id = st.selectbox("牛ID", cow_ids if cow_ids else [default_cow], index=0)
    d = st.date_input("日付", value=date.today())
  with colB:
    operator = st.text_input("作業者（任意）", value="")
  with colC:
    note = st.text_input("メモ（任意）", value="")

  b1, b2, b3, b4 = st.columns(4)

  def log(event_type, detail):
    append_row(XLSX_PATH, "events", {{
      "timestamp": datetime.now().isoformat(timespec="seconds"),
      "date": d.isoformat(),
      "cow_id": cow_id,
      "event_type": event_type,
      "detail": (detail + (" | " + note if note else "")).strip(),
      "operator": operator
    }})

  if b1.button("✅ 授精（insemination）", use_container_width=True):
    log("insemination", "AI recommended insemination")
    st.success("授精を記録しました ✅")

  if b2.button("💉 点滴（iv）", use_container_width=True):
    log("iv", "AI recommended IV")
    st.success("点滴を記録しました ✅")

  if b3.button("🩺 治療（treatment）", use_container_width=True):
    log("treatment", "AI recommended treatment")
    st.success("治療を記録しました ✅")

  if b4.button("🧪 検査（ketone_test）", use_container_width=True):
    log("ketone_test", "AI recommended ketone test")
    st.success("ケトン検査を記録しました ✅")
'''

def write_plugin():
    safe_mkdir("plugins")
    s = PLUGIN_TEMPLATE.format(
        xlsx_path=CFG0.xlsx_path,
        scored_path=CFG0.scored_today_csv,
        todo_path=CFG0.todo_today_csv,
        alerts_path=CFG0.alerts_today_csv
    )
    with open(PLUGIN_PATH, "w", encoding="utf-8") as f:
        f.write(s)

# ---------------- main pipeline ----------------
def run_once() -> Dict:
    # dirs
    safe_mkdir(CFG0.out_dir)
    safe_mkdir(CFG0.models_dir)
    safe_mkdir(CFG0.history_dir)

    # load Excel
    cows = read_sheet_df(CFG0.xlsx_path, "cows")
    daily = read_sheet_df(CFG0.xlsx_path, "daily")
    events = read_sheet_df(CFG0.xlsx_path, "events")

    if daily.empty:
        raise RuntimeError("daily が空です。cow_manager_app で日次入力してください。")

    # standardize
    daily["cow_id"] = daily["cow_id"].astype(str)
    daily["date"] = daily["date"].astype(str)

    # labels from events
    labeled = build_labels_from_events(daily, events)

    # features
    feat = add_timeseries_features(labeled)
    feat_cols = select_feature_columns(feat)

    # train models
    models = {}
    metrics = {}
    for t in TASKS:
        model, stat = train_one_task(feat, t, feat_cols, seed=CFG0.random_seed)
        models[t] = model
        metrics[t] = stat

    new_bundle = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "xlsx_path": CFG0.xlsx_path,
        "feature_columns": feat_cols,
        "models": models,
        "metrics": metrics
    }

    # compare with current
    old_bundle = load_current_bundle(CFG0.current_model)
    old_metrics = old_bundle.get("metrics") if old_bundle else None

    promote = should_promote(metrics, old_metrics)

    if promote:
        save_bundle(new_bundle)
        status = "PROMOTED"
    else:
        status = "REJECTED_KEEP_OLD"

    # score using best available bundle (new if promoted else old if exists else new)
    active = new_bundle if (promote or old_bundle is None) else old_bundle
    scored_today = score_today(active, feat, active["feature_columns"])
    export_outputs(scored_today)

    # write latest metrics json
    out = {
        "status": status,
        "active_model_created_at": active.get("created_at"),
        "new_model_created_at": new_bundle.get("created_at"),
        "metrics_new": metrics,
        "metrics_old": old_metrics
    }
    with open(CFG0.metrics_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # update plugin (UI)
    write_plugin()

    return out

def loop(minutes: int):
    while True:
        try:
            out = run_once()
            print(f"[{datetime.now().isoformat(timespec='seconds')}] OK: {out['status']}")
        except Exception as e:
            print(f"[{datetime.now().isoformat(timespec='seconds')}] ERROR: {e}")
        time.sleep(max(60, minutes * 60))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--minutes", type=int, default=30)
    args = ap.parse_args()

    if args.loop:
        loop(args.minutes)
        return

    # default: run once
    out = run_once()
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
