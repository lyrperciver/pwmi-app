# app.py — PWMI 预测原型（多语言 + 字段提示 + 风险等级/句子 + 批量导出）
# 依赖：pip install streamlit pandas joblib numpy
import json, io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# 1) 多语言文本（可自行扩展）
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
}

# ===============
# 2) 语言选择
# ===============
st.set_page_config(page_title=TEXT["title"]["zh"], layout="centered")
lang_choice = st.sidebar.radio(
    f'{TEXT["lang_label"]["zh"]} / {TEXT["lang_label"]["en"]}',
    ["中文", "English"],
    index=0,
    horizontal=True,
)
LANG = "zh" if lang_choice == "中文" else "en"
# 更新标题
st.title(TEXT["title"][LANG])
st.caption(TEXT["caption"][LANG])

# =============================
# 3) 路径与发布物（与原版一致）
# =============================
ROOT = Path(__file__).parent
PIPE_PATH = ROOT / "final_pipeline.joblib"
SCHEMA_PATH = ROOT / "feature_schema.json"
THR_PATH = ROOT / "thresholds.json"
META_PATH = ROOT / "release_meta.json"

# 可选：字段显示映射（键=特征列名）
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

# ===========================
# 4) 载入资产（缓存）
# ===========================
@st.cache_resource
def load_assets():
    # 1) 先确定文件存在
    if not PIPE_PATH.exists():
        st.error(f"未找到模型文件：{PIPE_PATH.name}。请将模型文件放到应用根目录。")
        st.stop()

    # 2) 更稳健的加载器：
    #    - 若同名 .skops 文件存在，优先用 skops（避免 joblib 反序列化依赖本地自定义模块报错）
    #    - 否则回退到 joblib.load
    pipe = None
    skops_path = PIPE_PATH.with_suffix(".skops")

    # 2.1 先尝试 skops
    if skops_path.exists():
        try:
            import skops.io as sio  # 仅在需要时导入，requirements 里建议加入 `skops`
            pipe = sio.load(skops_path, trusted=True)
        except Exception as e:
            st.warning(f"读取 SKOPS 模型失败（将回退到 joblib）：{e}")

    # 2.2 若还没有成功，回退到 joblib
    if pipe is None:
        try:
            pipe = joblib.load(PIPE_PATH)
        except ModuleNotFoundError as e:
            st.error(
                "无法通过 joblib 加载模型，通常是因为训练时的自定义模块/路径在云端不存在。\n"
                "解决方案之一：在本地把模型导出为 SKOPS 格式（如 final_pipeline.skops）并上传到仓库根目录；"
                "或将模型中用到的自定义类/函数随应用一起打包。"
                f"\n原始错误：{e}"
            )
            st.stop()
        except Exception as e:
            st.error(f"加载 joblib 模型失败：{e}")
            st.stop()

    # 3) 读取 schema / 阈值 / 元信息（保持你原先逻辑）
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    order = schema.get("order") or [d["name"] for d in schema["features"]]
    feat_defs = schema.get("features") or [{"name": n, "dtype": "float", "allowed_range": [None, None]} for n in order]
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

# =====================================
# 5) 工具函数：二元识别 / 标签与提示（大小写无关）
# =====================================

# ① 大小写无关的显示映射：把 DISPLAY 的 key 全部转成小写做查找
DISPLAY_CI = {k.lower(): v for k, v in DISPLAY.items()}

# ② 常见的二元变量名称（全小写）；可按需增删
BINARY_FEATURES = {
    "seizure", "aop", "pda", "bpd", "rds",
    "ivh", "sepsis", "asphyxia", "hypoglycemia",
    "hypocalcemia", "phototherapy", "steroids",
    "ventilation", "invasive_vent", "inv_vent", "cpap",
    # 以 _flag/_bin 结尾的也按二元处理（见 is_binary_name）
}
# 明确是“天数/小时数”的不要放进来（如 inv_vent_days）

def is_binary_name(name: str) -> bool:
    n = str(name).lower()
    if n in BINARY_FEATURES:
        return True
    # 变量名以这些后缀结尾，也视为二元
    for suf in ("_flag", "_bin", "_binary"):
        if n.endswith(suf):
            return True
    return False

def is_binary_def(defn: dict, name: str) -> bool:
    """多策略判断是否二元：dtype/allowed_range/变量名启发式"""
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
    # 最后用名称启发式
    return is_binary_name(name)

def label_for(name: str, defn: dict) -> str:
    """按 DISPLAY（大小写无关） → 否则用 列名(+单位)"""
    unit = defn.get("unit") or ""
    mapped = DISPLAY_CI.get(str(name).lower(), {}).get(LANG)
    if mapped:
        return mapped  # 已含单位
    return f"{name} ({unit})" if unit else str(name)

def help_for(defn: dict, is_bin: bool) -> str:
    """生成 tooltip：范围/步进 + 二元说明"""
    lo, hi = (defn.get("allowed_range") or [None, None])
    step = defn.get("step", 1 if is_bin else 0.1)
    bits = []
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        bits.append(f'{TEXT["range_help"][LANG]}: {lo}–{hi}')
    bits.append(f'{TEXT["step_help"][LANG]}: {step}')
    if is_bin:
        bits.append(TEXT["binary_help"][LANG])
    return " · ".join(bits)

# ===========================
# 6) 表单输入（带提示/二元下拉）
# ===========================
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
            # 二元变量：用下拉 0/1，既有 tooltip，也在控件下方再写一遍文字提示
            values[name] = st.selectbox(
                lbl, options=[0, 1], index=0, help=help_text, key=f"bin_{name}"
            )
            # 在控件下方**再明确**写一次，避免用户没注意到问号提示
            st.caption(TEXT["binary_help"][LANG])
        else:
            # 数值变量：默认值 = 范围中点，否则 0.0
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
# —— 6.5) 阈值策略 + 预测按钮（必须先定义，后面要用） ——
st.divider()
mode = st.radio(
    TEXT["thresh_mode"][LANG],
    [TEXT["youd"][LANG], TEXT["hsens"][LANG]],
    horizontal=True,
)
btn_predict = st.button(TEXT["predict"][LANG], type="primary")

# ===========================
# 7) 风险分级规则（基于两阈值）
# ===========================
def pick_thresholds(thrs_dict):
    t_youden = thrs_dict.get("youden")
    t_hsens = thrs_dict.get("highs")
    # 缺失兜底
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

# ===========================
# 8) 单例预测
# ===========================
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

    # 风险等级句子（显示 + 报告写入）
    risk_sent = TEXT["risk_sentence"][LANG].format(level=risk_bucket(p, LANG), p=p)
    st.write(risk_sent)

    # HTML 报告
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

# ===========================
# 9) 批量 CSV 预测 + 风险句子
# ===========================
st.markdown(f"### {TEXT['batch_title'][LANG]}")
st.caption(TEXT["batch_caption"][LANG])
up = st.file_uploader(TEXT["upload_csv"][LANG], type=["csv"], accept_multiple_files=False)

if up is not None:
    try:
        df_in = pd.read_csv(up)
        X = df_in.reindex(columns=order)  # 缺列→全缺失列，交由管道处理
        p = pipe.predict_proba(X)[:, 1]
        out = df_in.copy()
        out["prob"] = p
        out["label_youden"] = (p >= t_youden).astype(int)
        out["label_highsens"] = (p >= t_hsens).astype(int)
        # 风险等级 + 句子（中英文）
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

