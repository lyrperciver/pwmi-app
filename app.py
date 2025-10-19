# app.py — PWMI 预测原型（远程下载模型 + 缓存 + 多语言 + 批量导出）
# 依赖：streamlit pandas numpy scikit-learn joblib imbalanced-learn lightgbm catboost skops requests packaging
# 建议 Python 3.11（在根目录 runtime.txt: python-3.11.9）

import json, io, os, hashlib, tempfile, shutil
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from packaging.version import Version

# =========================
# 0) 多语言文本
# =========================
TEXT = {
    "lang_label": {"zh": "界面语言", "en": "Interface language"},
    "title": {"zh": "PWMI 风险预测 · 原型（研究性）", "en": "PWMI Risk Prediction · Prototype"},
    "caption": {
        "zh": "说明：本工具仅用于科研/质控，不构成临床诊断依据。",
        "en": "Note: Research/QC only. Not for clinical diagnosis.",
    },
    "meta": {"zh": "版本/来源信息", "en": "Release / Meta"},
    "input_section": {"zh": "输入特征", "en": "Input features"},
    "thresh_mode": {"zh": "阈值策略", "en": "Threshold mode"},
    "predict": {"zh": "预测", "en": "Predict"},
    "youd": {"zh": "Youden", "en": "Youden"},
    "hsens": {"zh": "高敏感", "en": "High sensitivity"},
    "probability": {"zh": "预测概率", "en": "Predicted probability"},
    "two_rules": {"zh": "两套阈值判定：", "en": "Decisions under two thresholds:"},
    "pos": {"zh": "阳性", "en": "Positive"},
    "neg": {"zh": "阴性", "en": "Negative"},
    "current_mode": {"zh": "当前策略", "en": "Current mode"},
    "result": {"zh": "结果", "en": "Result"},
    "download_html": {"zh": "下载 HTML 报告", "en": "Download HTML report"},
    "batch_title": {"zh": "批量 CSV 预测", "en": "Batch CSV prediction"},
    "batch_caption": {
        "zh": "上传包含同名列的 CSV（至少包含小特征集这几列；缺失将按训练时的 SimpleImputer 处理）",
        "en": "Upload CSV with matching columns (missing values handled by training-time imputer).",
    },
    "upload_csv": {"zh": "选择 CSV 文件", "en": "Select CSV file"},
    "download_csv": {"zh": "下载结果 CSV", "en": "Download result CSV"},
    "range_help": {"zh": "范围", "en": "Range"},
    "step_help": {"zh": "步进", "en": "Step"},
    "binary_help": {"zh": "二元变量：1=发生，0=未发生", "en": "Binary: 1=present, 0=absent"},
    "risk_low": {"zh": "低", "en": "low"},
    "risk_mid": {"zh": "中", "en": "medium"},
    "risk_high": {"zh": "高", "en": "high"},
    "risk_sentence": {
        "zh": "该患儿脑损伤的风险较{level}（概率 {p:.1%}）。",
        "en": "The infant's risk of brain injury is {level} (probability {p:.1%}).",
    },
    "read_fail": {"zh": "读取或预测失败：", "en": "Read / prediction failed: "},
    "done_n": {"zh": "预测完成：{n} 条", "en": "Done: {n} rows"},
    "model_missing": {
        "zh": "未找到本地模型，也未配置远程下载地址。请在 Secrets 中设置 MODEL_URL 或 MODEL_URLS。",
        "en": "No local model and no remote URL configured. Please set MODEL_URL or MODEL_URLS in Secrets.",
    },
    "downloading": {"zh": "下载模型文件…", "en": "Downloading model file…"},
    "verifying": {"zh": "校验文件完整性…", "en": "Verifying file integrity…"},
    "merging": {"zh": "合并分片…", "en": "Merging parts…"},
    "cached": {"zh": "已命中缓存，跳过下载。", "en": "Cache hit, skip download."},
    "skops_fail": {"zh": "读取 SKOPS 失败：", "en": "SKOPS load failed: "},
    "joblib_fail": {"zh": "加载 joblib 模型失败：", "en": "Load joblib failed: "},
}

# =============== 语言选择 ===============
st.set_page_config(page_title=TEXT["title"]["zh"], layout="centered")
lang_choice = st.sidebar.radio(
    f'{TEXT["lang_label"]["zh"]} / {TEXT["lang_label"]["en"]}',
    ["中文", "English"],
    index=0,
    horizontal=True,
)
LANG = "zh" if lang_choice == "中文" else "en"
st.title(TEXT["title"][LANG])
st.caption(TEXT["caption"][LANG])

# =============== 常量与路径 ===============
ROOT = Path(__file__).parent
CACHE_DIR = ROOT / ".model_cache"
CACHE_DIR.mkdir(exist_ok=True)

LOCAL_SKOPS = ROOT / "final_pipeline.skops"
LOCAL_JOBLIB = ROOT / "final_pipeline.joblib"

SCHEMA_PATH = ROOT / "feature_schema.json"
THR_PATH = ROOT / "thresholds.json"
META_PATH = ROOT / "release_meta.json"

DISPLAY = {
    "alb":  {"zh": "白蛋白 (ALB, g/L)",           "en": "Albumin (ALB, g/L)"},
    "ldh":  {"zh": "乳酸脱氢酶 (LDH, U/L)",        "en": "Lactate Dehydrogenase (LDH, U/L)"},
    "hbdh": {"zh": "α-羟丁酸脱氢酶 (HBDH, U/L)",  "en": "α-Hydroxybutyrate Dehydrogenase (HBDH, U/L)"},
    "apgar_1min": {"zh": "Apgar 1分钟评分（1-10）",        "en": "Apgar at 1 minute"},
    "aop":  {"zh": "早产儿贫血 (AOP, 0/1)",    "en": "Apnea of prematurity (AOP, 0/1)"},
    "seizure": {"zh": "惊厥 (0/1)",           "en": "Clinical seizure (0/1)"},
    "inv_vent_days": {"zh": "有创通气天数 (d)",    "en": "Invasive ventilation (days)"},
    "birth_weight_g": {"zh": "出生体重 (g)",       "en": "Birth weight (g)"},
}

# =============== 工具：下载 & 校验 ===============
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download_one(url: str, dst: Path, progress: Optional[st.delta_generator.DeltaGenerator]=None) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        with dst.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    got += len(chunk)
                    if progress and total:
                        progress.progress(min(1.0, got / total))

def merge_files(parts: List[Path], dst: Path) -> None:
    with dst.open("wb") as out:
        for p in parts:
            with p.open("rb") as f:
                shutil.copyfileobj(f, out)

def get_urls_from_secrets_or_env() -> Tuple[List[str], Optional[str]]:
    urls, sha = [], None
    try:
        if "MODEL_URLS" in st.secrets:
            urls = [u.strip() for u in str(st.secrets["MODEL_URLS"]).strip().splitlines() if u.strip()]
        elif "MODEL_URL" in st.secrets:
            urls = [str(st.secrets["MODEL_URL"]).strip()]
        if "MODEL_SHA256" in st.secrets:
            sha = str(st.secrets["MODEL_SHA256"]).strip().lower()
    except Exception:
        pass
    if not urls:
        env_urls = os.environ.get("MODEL_URLS") or os.environ.get("MODEL_URL") or ""
        if env_urls.strip():
            if "\n" in env_urls:
                urls = [u.strip() for u in env_urls.strip().splitlines() if u.strip()]
            else:
                urls = [u.strip() for u in env_urls.split(",") if u.strip()]
    if sha is None:
        sha = (os.environ.get("MODEL_SHA256") or "").strip().lower() or None
    return urls, sha

def ensure_model_file() -> Path:
    # 本地优先
    if LOCAL_SKOPS.exists():
        return LOCAL_SKOPS
    if LOCAL_JOBLIB.exists():
        return LOCAL_JOBLIB

    urls, sha = get_urls_from_secrets_or_env()
    if not urls:
        st.error(TEXT["model_missing"][LANG])
        st.stop()

    target = CACHE_DIR / "final_pipeline.skops"
    # 命中缓存
    if target.exists() and (not sha or sha256_file(target) == sha):
        st.info(TEXT["cached"][LANG])
        return target

    # 下载
    if len(urls) > 1:
        st.info(TEXT["downloading"][LANG])
        tmp_dir = Path(tempfile.mkdtemp())
        parts = []
        for i, url in enumerate(urls):
            st.write(f"Part {i+1}/{len(urls)}")
            bar = st.progress(0.0)
            p = tmp_dir / f"part_{i:03d}"
            download_one(url, p, progress=bar)
            parts.append(p)
        st.info(TEXT["merging"][LANG])
        merge_files(parts, target)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        st.info(TEXT["downloading"][LANG])
        bar = st.progress(0.0)
        tmp = target.with_suffix(".down")
        download_one(urls[0], tmp, progress=bar)
        tmp.replace(target)

    # 校验
    if sha:
        st.info(TEXT["verifying"][LANG])
        got = sha256_file(target)
        if got.lower() != sha.lower():
            target.unlink(missing_ok=True)
            st.error(TEXT["read_fail"][LANG] + f"SHA256 mismatch. expected={sha}, got={got}")
            st.stop()

    return target

# =============== SKOPS 安全加载（兼容 0.10+） ===============
def safe_skops_load(path: Path):
    import skops, skops.io as sio
    ver = Version(skops.__version__)
    # 允许用户通过 Secrets/ENV 自定义信任前缀（逗号分隔）
    default_prefixes = ("sklearn.", "numpy.", "scipy.", "lightgbm", "catboost", "xgboost", "skops.")
    allow_env = (st.secrets.get("SKOPS_ALLOWED_PREFIXES", None)
                 if hasattr(st, "secrets") else None) or os.environ.get("SKOPS_ALLOWED_PREFIXES", "")
    if allow_env.strip():
        prefixes = tuple([p.strip() for p in allow_env.split(",") if p.strip()])
    else:
        prefixes = default_prefixes

    trust_all = (str(st.secrets.get("SKOPS_TRUST_ALL", ""))
                 if hasattr(st, "secrets") else os.environ.get("SKOPS_TRUST_ALL", "")) \
                 .strip().lower() in {"1", "true", "yes"}

    # 0.10 之前：仍接受 bool
    if ver < Version("0.10"):
        return sio.load(path, trusted=True)

    # 0.10 及以上：必须传字符串列表
    untrusted = sio.get_untrusted_types(path)

    if trust_all:
        trusted = untrusted
    else:
        trusted = [t for t in untrusted if any(t.startswith(pref) for pref in prefixes)]

    with st.expander("SKOPS 安全审计 / Trusted types", expanded=False):
        st.write("发现的类型（untrusted types）:", untrusted)
        st.write("将被信任并加载的类型（trusted）:", trusted)
        if not trust_all and len(trusted) != len(untrusted):
            st.info("提示：如需一次全信任，可在 Secrets/ENV 设 SKOPS_TRUST_ALL=1；或扩大 SKOPS_ALLOWED_PREFIXES。")

    return sio.load(path, trusted=trusted)

# =============== 载入资产（模型 + schema/阈值/元信息） ===============
@st.cache_resource
def load_assets():
    model_path = ensure_model_file()

    pipe = None
    suffix = model_path.suffix.lower()

    if suffix == ".skops":
        try:
            pipe = safe_skops_load(model_path)
        except Exception as e:
            st.error(TEXT["skops_fail"][LANG] + str(e))
            st.stop()

    elif suffix in {".joblib", ".pkl", ".pickle"}:
        try:
            pipe = joblib.load(model_path)
        except Exception as e:
            st.error(TEXT["joblib_fail"][LANG] + str(e))
            st.stop()
    else:
        st.error(TEXT["read_fail"][LANG] + f"Unsupported model file: {model_path.name}")
        st.stop()

    # 读取 schema / 阈值 / 元信息
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    order = schema.get("order") or [d["name"] for d in schema["features"]]
    feat_defs = schema.get("features") or [
        {"name": n, "dtype": "float", "allowed_range": [None, None]} for n in order
    ]
    defs_by_name = {d["name"]: d for d in feat_defs}

    thr = json.loads(THR_PATH.read_text(encoding="utf-8"))

    def pick(obj, *keys):
        for key in keys:
            if key in obj:
                x = obj.get(key)
                return float(x["thr"]) if isinstance(x, dict) else float(x)
        return None

    youden = pick(thr, "youden", "Youden")
    highs = pick(thr, "high_sensitivity", "highSensitivity", "HighSens", "highsens")

    meta = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    return pipe, order, defs_by_name, {"youden": youden, "highs": highs}, meta

pipe, order, featdefs, thrs, meta = load_assets()

with st.expander(TEXT["meta"][LANG], expanded=False):
    st.json(meta or {"note": "N/A"})

# =============== 工具：二元识别 / 标签与提示 ===============
DISPLAY_CI = {k.lower(): v for k, v in DISPLAY.items()}
BINARY_FEATURES = {
    "seizure", "aop", "pda", "bpd", "rds",
    "ivh", "sepsis", "asphyxia", "hypoglycemia",
    "hypocalcemia", "phototherapy", "steroids",
    "ventilation", "invasive_vent", "inv_vent", "cpap",
}

def is_binary_name(name: str) -> bool:
    n = str(name).lower()
    if n in BINARY_FEATURES:
        return True
    for suf in ("_flag", "_bin", "_binary"):
        if n.endswith(suf):
            return True
    return False

def is_binary_def(defn: dict, name: str) -> bool:
    dtype = str(defn.get("dtype", "")).lower()
    if dtype == "binary":
        return True
    rng = defn.get("allowed_range") or [None, None]
    if rng[0] == 0 and rng[1] == 1:
        step = defn.get("step", 1)
        try:
            if int(step) == 1:
                return True
        except Exception:
            pass
    return is_binary_name(name)

def label_for(name: str, defn: dict) -> str:
    unit = defn.get("unit") or ""
    mapped = DISPLAY_CI.get(str(name).lower(), {}).get(LANG)
    if mapped:
        return mapped
    return f"{name} ({unit})" if unit else str(name)

def help_for(defn: dict, is_bin: bool) -> str:
    lo, hi = (defn.get("allowed_range") or [None, None])
    step = defn.get("step", 1 if is_bin else 0.1)
    bits = []
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        bits.append(f'{TEXT["range_help"][LANG]}: {lo}–{hi}')
    bits.append(f'{TEXT["step_help"][LANG]}: {step}')
    if is_bin:
        bits.append(TEXT["binary_help"][LANG])
    return " · ".join(bits)

# =============== 表单输入 ===============
st.markdown(f"### {TEXT['input_section'][LANG]}")
cols = st.columns(2)
values = {}

for i, name in enumerate(order):
    d = featdefs.get(name, {})
    is_bin = is_binary_def(d, name)
    lbl = label_for(name, d)
    help_text = help_for(d, is_bin)
    lo, hi = (d.get("allowed_range") or [None, None])

    with cols[i % 2]:
        if is_bin:
            values[name] = st.selectbox(
                lbl, options=[0, 1], index=0, help=help_text, key=f"bin_{name}"
            )
            st.caption(TEXT["binary_help"][LANG])
        else:
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                default = (float(lo) + float(hi)) / 2.0
                step = float(d.get("step", 0.1))
                values[name] = st.number_input(
                    lbl,
                    value=float(default),
                    min_value=float(lo),
                    max_value=float(hi),
                    step=step,
                    format="%.3f",
                    help=help_text,
                )
            else:
                values[name] = st.number_input(
                    lbl,
                    value=0.0,
                    step=float(d.get("step", 0.1)),
                    format="%.3f",
                    help=help_text,
                )

st.divider()
mode = st.radio(
    TEXT["thresh_mode"][LANG],
    [TEXT["youd"][LANG], TEXT["hsens"][LANG]],
    horizontal=True,
)
btn_predict = st.button(TEXT["predict"][LANG], type="primary")

# =============== 风险分级规则（两阈值） ===============
def pick_thresholds(thrs_dict):
    t_youden = thrs_dict.get("youden")
    t_hsens = thrs_dict.get("highs")
    if t_youden is None:
        t_youden = 0.5
    if t_hsens is None:
        t_hsens = 0.2
    lo, hi = sorted([float(t_youden), float(t_hsens)])
    return lo, hi, float(t_youden), float(t_hsens)

low_thr, high_thr, t_youden, t_hsens = pick_thresholds(thrs)

def risk_bucket(p: float, lang="zh") -> str:
    if p < low_thr:
        return TEXT["risk_low"][lang]
    elif p < high_thr:
        return TEXT["risk_mid"][lang]
    else:
        return TEXT["risk_high"][lang]

# =============== 单例预测 ===============
if btn_predict:
    x = pd.DataFrame([values])[order]
    p = float(pipe.predict_proba(x)[:, 1][0])
    label_youden = int(p >= t_youden)
    label_highs = int(p >= t_hsens)

    st.metric(TEXT["probability"][LANG], f"{p:.4f}")
    st.write(f"**{TEXT['two_rules'][LANG]}**")
    st.write(f"- Youden @ {t_youden:.6f} → **{TEXT['pos'][LANG] if label_youden else TEXT['neg'][LANG]}**")
    st.write(f"- {TEXT['hsens'][LANG]} @ {t_hsens:.6f} → **{TEXT['pos'][LANG] if label_highs else TEXT['neg'][LANG]}**")

    final_label = label_youden if mode == TEXT["youd"][LANG] else label_highs
    st.success(f"{TEXT['current_mode'][LANG]}：**{mode}** → {TEXT['result'][LANG]}：**{TEXT['pos'][LANG] if final_label else TEXT['neg'][LANG]}**")

    risk_sent = TEXT["risk_sentence"][LANG].format(level=risk_bucket(p, LANG), p=p)
    st.write(risk_sent)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>PWMI Report</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;max-width:900px;margin:24px auto;}}
h1{{font-size:22px;}} table{{border-collapse:collapse;width:100%;}}
td,th{{border:1px solid #ddd;padding:6px 8px;}} .ok{{color:#0b8a00;}} .bad{{color:#b80000;}}
.small{{color:#666;font-size:12px;}}
</style></head>
<body>
<h1>{TEXT['title'][LANG]}</h1>
<div class="small">version: {meta.get('version','N/A')} · build: {meta.get('build_time','N/A')}</div>
<h2>{TEXT['input_section'][LANG]}</h2>
<table>
<tr><th>Feature</th><th>Value</th></tr>
{''.join([f'<tr><td>{name}</td><td>{values[name]}</td></tr>' for name in order])}
</table>
<h2>{TEXT['result'][LANG]}</h2>
<p>{TEXT['probability'][LANG]}：<b>{p:.4f}</b></p>
<ul>
<li>Youden@{t_youden:.6f} → <b class="{'bad' if label_youden else 'ok'}">{TEXT['pos'][LANG] if label_youden else TEXT['neg'][LANG]}</b></li>
<li>{TEXT['hsens'][LANG]}@{t_hsens:.6f} → <b class="{'bad' if label_highs else 'ok'}">{TEXT['pos'][LANG] if label_highs else TEXT['neg'][LANG]}</b></li>
</ul>
<p><b>{TEXT['current_mode'][LANG]}：</b>{mode} → <b class="{ 'bad' if final_label else 'ok'}">{TEXT['pos'][LANG] if final_label else TEXT['neg'][LANG]}</b></p>
<p style="font-weight:600">{risk_sent}</p>
<hr>
<p class="small">{TEXT['caption'][LANG]}</p>
</body></html>"""
    st.download_button(TEXT["download_html"][LANG], data=html.encode("utf-8"),
                       file_name="pwmi_report.html", mime="text/html")

# =============== 批量 CSV 预测 + 风险句子 ===============
st.markdown(f"### {TEXT['batch_title'][LANG]}")
st.caption(TEXT["batch_caption"][LANG]")
up = st.file_uploader(TEXT["upload_csv"][LANG], type=["csv"], accept_multiple_files=False)

if up is not None:
    try:
        df_in = pd.read_csv(up)
        X = df_in.reindex(columns=order)
        p = pipe.predict_proba(X)[:, 1]
        out = df_in.copy()
        out["prob"] = p
        out["label_youden"] = (p >= t_youden).astype(int)
        out["label_highsens"] = (p >= t_hsens).astype(int)
        out["risk_level_zh"] = [risk_bucket(float(x), "zh") for x in p]
        out["risk_level_en"] = [risk_bucket(float(x), "en") for x in p]
        out["summary_zh"] = [TEXT["risk_sentence"]["zh"].format(level=risk_bucket(float(x), "zh"), p=float(x)) for x in p]
        out["summary_en"] = [TEXT["risk_sentence"]["en"].format(level=risk_bucket(float(x), "en"), p=float(x)) for x in p]

        st.success(TEXT["done_n"][LANG].format(n=len(out)))
        st.dataframe(out.head(20))

        buf = io.StringIO()
        out.to_csv(buf, index=False, encoding="utf-8-sig")
        st.download_button(TEXT["download_csv"][LANG],
                           data=buf.getvalue().encode("utf-8-sig"),
                           file_name="pwmi_predictions.csv",
                           mime="text/csv")
    except Exception as e:
        st.error(TEXT["read_fail"][LANG] + str(e))

st.divider()
st.caption("Roadmap: SHAP/EBM explain, stricter schema validation, PDF export, FastAPI.")
