"""
Multimodal Lifestyle Disorder Risk & Explainability Engine
KSCST demo dashboard - Dual-Tower Gated Multimodal Unit (GMU)
    Clinical tower : MIMIC-IV EHR  (glucose, BMI, systolic BP)
    Wearable tower : WESAD 4 Hz    (heart rate, EDA, BVP amplitude)

Run:  streamlit run app.py
Deps: pip install streamlit plotly torch numpy

Design notes (read before the demo)
-----------------------------------
* The risk/state logic is a transparent multivariate INTERACTION engine
  (no feature is scored in isolation). The PyTorch GMU runs alongside it; if a
  trained checkpoint `gmu_weights.pt` sits next to this file it is loaded and
  blended into the state logits, otherwise the GMU runs with untrained demo
  weights and is shown for gate/architecture visualisation only.
* SHAP values are EXACT Shapley values (all 2^n coalitions) of the fused risk
  function against a healthy reference patient, so interaction effects
  (glucose x BMI, EDA x BMI, ...) are shared correctly between features.
* Research prototype. Not a medical device. Not for clinical use.
"""

import time
from math import factorial
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from plotly.subplots import make_subplots

st.set_page_config(page_title="Multimodal Lifestyle Risk Engine", page_icon="🫀", layout="wide")

# =============================================================================
# ===== ENGINE =====
# =============================================================================
FEATURES = ["glucose", "bmi", "sbp", "hr", "eda", "bvp"]
LABELS = {
    "glucose": "Glucose", "bmi": "BMI", "sbp": "Systolic BP",
    "hr": "Heart Rate", "eda": "EDA", "bvp": "BVP Amplitude",
}
UNITS = {"glucose": "mg/dL", "bmi": "kg/m²", "sbp": "mmHg", "hr": "bpm", "eda": "µS", "bvp": "a.u."}
TOWER = {"glucose": "Clinical", "bmi": "Clinical", "sbp": "Clinical",
         "hr": "Wearable", "eda": "Wearable", "bvp": "Wearable"}

# Healthy reference patient (SHAP baseline and neutral imputation value)
BASELINE = {"glucose": 90.0, "bmi": 22.0, "sbp": 115.0, "hr": 65.0, "eda": 2.0, "bvp": 1.0}

# Physiological plausibility windows. <= 0.0 is always a dropout / lead-off.
VALID = {"glucose": (40, 500), "bmi": (12, 70), "sbp": (60, 260),
         "hr": (30, 220), "eda": (0.05, 30), "bvp": (0.05, 6.0)}

# Slider ranges deliberately include 0.0 and out-of-range values so judges can
# trigger the artifact handling.
SLIDER = {"glucose": (0.0, 600.0, 1.0), "bmi": (0.0, 80.0, 0.1), "sbp": (0.0, 300.0, 1.0),
          "hr": (0.0, 250.0, 1.0), "eda": (0.0, 40.0, 0.1), "bvp": (0.0, 8.0, 0.05)}

STATES = {
    0: dict(name="Optimal Baseline (Homeostasis)", color="#2e9e5b",
            summary="Autonomic, cardiovascular and metabolic indices all sit within healthy resting ranges.",
            detail=("Sympathetic and parasympathetic tone are balanced, glucose handling is efficient and "
                    "body composition is not contributing to cardiometabolic load. Wearable streams show "
                    "resting-level EDA and pulse-wave amplitude concordant with normal clinical baselines. "
                    "No intervention is indicated; routine monitoring only.")),
    1: dict(name="Mild Autonomic Stress", color="#8bc34a",
            summary="Acute sympathetic arousal without metabolic compromise.",
            detail=("Electrodermal activity and pulse-wave amplitude are elevated while clinical baselines "
                    "(BMI, glucose, blood pressure) remain normal. This pattern is characteristic of "
                    "transient, situational stress (cognitive load, anxiety, caffeine) rather than chronic "
                    "disease. It is expected to resolve on recovery; it becomes relevant only if it "
                    "persists or later co-occurs with metabolic markers.")),
    2: dict(name="Early Metabolic Drift", color="#cddc39",
            summary="Subclinical insulin-resistance indicators emerging.",
            detail=("Glucose and/or BMI have moved above optimal but below diagnostic thresholds. Pancreatic "
                    "compensation is likely still masking overt dysglycaemia (pre-diabetic trajectory). "
                    "Autonomic and cardiovascular signals are largely unremarkable. This is the most "
                    "modifiable stage: lifestyle intervention has the highest yield here.")),
    3: dict(name="Elevated Cardiovascular Load", color="#ffc107",
            summary="Sustained heart rate and pulse-wave elevation.",
            detail=("Resting heart rate, BVP pulse amplitude and/or systolic pressure are persistently raised "
                    "without a matching electrodermal (stress) response, pointing to a haemodynamic rather "
                    "than psychological driver (deconditioning, exertion, volume/pressure load). Sustained "
                    "load of this kind increases myocardial work and long-term vascular risk.")),
    4: dict(name="Moderate Lifestyle Risk", color="#ff9800",
            summary="Combined biometric and clinical warning markers.",
            detail=("Several systems show moderate, concurrent deviations - for example borderline glucose "
                    "with raised BMI and mildly elevated heart rate. No single marker is alarming, but the "
                    "combination indicates a cumulative lifestyle-related risk profile that warrants a "
                    "structured review of diet, activity, sleep and stress.")),
    5: dict(name="High Glycemic Variability", color="#f4511e",
            summary="Significant glucose and BMI threshold breaches.",
            detail=("Glucose and BMI both exceed clinical thresholds and compound each other "
                    "(insulin resistance amplified by adiposity). The metabolic tower dominates the fused "
                    "representation. Risk of progression to type 2 diabetes and metabolic syndrome is high; "
                    "clinical evaluation (HbA1c, lipid panel) is recommended.")),
    6: dict(name="Acute Sympathetic Surge", color="#e53935",
            summary="High EDA/BVP paired with cardiovascular strain.",
            detail=("Strong electrodermal and pulse-wave activation coincide with elevated heart rate and/or "
                    "blood pressure: a fight-or-flight response with real haemodynamic cost. Clinical "
                    "metabolic markers need not be abnormal. Persistent or recurrent episodes raise "
                    "arrhythmic and hypertensive risk; check for triggers (pain, panic, stimulants).")),
    7: dict(name="Critical Multimodal Risk", color="#b71c1c",
            summary="Severe multi-system physiological breakdown across both towers.",
            detail=("Metabolic, autonomic and cardiovascular indices are simultaneously and substantially "
                    "abnormal, and both the clinical (MIMIC-IV) and wearable (WESAD) towers agree. Stress "
                    "physiology is acting on a compromised metabolic baseline, producing compounding "
                    "risk. Prompt clinical assessment is warranted.")),
}


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def validate(x):
    """Return {feature: message} for values that are sensor artifacts / non-physiological."""
    bad = {}
    for k in FEATURES:
        lo, hi = VALID[k]
        v = x[k]
        if v <= 0.0:
            bad[k] = (f"{LABELS[k]} = {v:g} → sensor dropout / lead-off / missing record "
                      f"(not physiologically possible)")
        elif v < lo or v > hi:
            bad[k] = (f"{LABELS[k]} = {v:g} {UNITS[k]} → outside plausible range "
                      f"({lo:g}-{hi:g} {UNITS[k]}); treated as artifact")
    return bad


def components(x, rel=1.0):
    """
    Multivariate interaction model. Returns normalised risk components in [0,1]
    plus the fused risk R. `rel` (<1) down-weights the wearable tower when the
    signal is motion-corrupted.
    """
    g = clip01((x["glucose"] - 100) / 100)
    b = clip01((x["bmi"] - 25) / 15)
    s = clip01((x["sbp"] - 120) / 60)
    h = clip01((x["hr"] - 75) / 55) * rel
    e = clip01((x["eda"] - 3) / 9) * rel
    v = clip01((x["bvp"] - 1.2) / 1.3) * rel

    # Metabolic axis: glucose x BMI compounding multiplier
    mult = 1.0 + 1.5 * g * b
    M = min(1.0, (0.55 * g + 0.45 * b) * mult)
    # Autonomic axis (EDA/BVP) and cardiovascular axis (HR/BVP/SBP)
    A = 0.6 * e + 0.4 * v
    C = 0.45 * h + 0.30 * v + 0.25 * s

    # Context gate: stress on top of a normal chronic baseline is "situational"
    # and is damped; stress on a compromised baseline is amplified (cross term).
    ctx = min(1.0, max(M, 0.5 * C) / 0.4)
    damp = 0.4 + 0.6 * ctx
    A_eff = A * damp
    cross = 0.25 * A * M
    R = 1.0 - (1 - 0.7 * M) * (1 - 0.5 * C) * (1 - 0.35 * A_eff)
    R = clip01(R + cross)
    situational = bool(A > 0.35 and M < 0.15 and s < 0.25)
    return dict(g=g, b=b, s=s, h=h, e=e, v=v, mult=mult, M=M, A=A, C=C,
                damp=damp, A_eff=A_eff, cross=cross, R=R, situational=situational)


def state_scores(c):
    """Interaction-based evidence for each of the 8 states (pre-softmax)."""
    A, M, C, g, b = c["A"], c["M"], c["C"], c["g"], c["b"]
    mx = max(A, M, C)
    mean3 = (A + M + C) / 3
    n_sys = sum(z > 0.3 for z in (A, M, C))
    gm = (A * M * C) ** (1 / 3)
    sc = np.zeros(8)
    sc[0] = 3.2 * (1 - mx) ** 1.5
    sc[1] = 4.0 * A * (1 - M) * (1 - C)
    sc[2] = 3.2 * np.exp(-(((M - 0.3) / 0.18) ** 2)) * (1 - C) * (1 - 0.5 * A)
    sc[3] = 3.6 * C * (1 - A) * (1 - M)
    sc[4] = 3.0 * np.exp(-(((mean3 - 0.4) / 0.2) ** 2)) * (1.0 if n_sys >= 2 else 0.4)
    sc[5] = 5.2 * max(0.0, M - 0.35) * (1.0 if (g > 0.3 and b > 0.3) else 0.5)
    sc[6] = 4.2 * A * C * (1 - 0.5 * M)
    sc[7] = (5.0 if min(A, M, C) > 0.45 else 1.6) * gm
    return sc


def softmax(z):
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def shapley(x, usable, rel):
    """Exact Shapley values of fused risk R vs. the healthy baseline (interaction-aware)."""
    n = len(usable)
    phi = {k: 0.0 for k in FEATURES}
    if n == 0:
        return phi
    cache = {}

    def val(mask):
        if mask not in cache:
            full = dict(BASELINE)
            for i, k in enumerate(usable):
                if (mask >> i) & 1:
                    full[k] = x[k]
            cache[mask] = components(full, rel)["R"]
        return cache[mask]

    for i, k in enumerate(usable):
        for mask in range(1 << n):
            if (mask >> i) & 1:
                continue
            sz = bin(mask).count("1")
            w = factorial(sz) * factorial(n - sz - 1) / factorial(n)
            phi[k] += w * (val(mask | (1 << i)) - val(mask))
    return phi


def explain_feature(k, x, c):
    """Context-aware reason WHY a feature contributes, given the other variables."""
    g, b, s, h, e, v = c["g"], c["b"], c["s"], c["h"], c["e"], c["v"]
    if k == "eda":
        if c["situational"]:
            return (f"EDA {x['eda']:.1f} µS is high while BMI ({x['bmi']:.1f}) and glucose ({x['glucose']:.0f}) are "
                    f"normal → acute situational stress: sympathetic arousal without a metabolic substrate, "
                    f"so its contribution is damped.")
        if e > 0.3 and b > 0.3:
            return ("Elevated EDA coupled with elevated BMI amplifies metabolic strain by compounding "
                    "autonomic stress with chronic baseline risk.")
        if e > 0.3 and g > 0.3:
            return ("Elevated EDA with elevated glucose: the stress response (catecholamines/cortisol) raises "
                    "hepatic glucose output on an already impaired glycaemic baseline.")
        if e > 0.3 and h > 0.3:
            return "High EDA together with raised heart rate indicates a genuine sympathetic surge, not isolated arousal."
        if e > 0.3:
            return "Isolated electrodermal arousal; no corroborating clinical or cardiovascular marker."
        return "EDA is near resting level; little autonomic contribution."
    if k == "bvp":
        if v > 0.3 and h > 0.3:
            return "Raised pulse-wave amplitude with elevated heart rate points to sustained vascular/cardiovascular load."
        if v > 0.3 and e > 0.3:
            return "Pulse-wave amplitude and EDA rise together, consistent with sympathetic vasomotor activation."
        if v > 0.3:
            return "Pulse-wave amplitude is raised without heart-rate or EDA support; weak evidence on its own."
        return "BVP amplitude is within resting range."
    if k == "hr":
        if h > 0.3 and e > 0.3:
            return "Elevated heart rate with high EDA reflects autonomically driven tachycardia (sympathetic surge)."
        if h > 0.3 and v > 0.3:
            return "Heart rate and pulse-wave amplitude are both elevated → sustained cardiovascular load."
        if h > 0.3:
            return ("Elevated heart rate without electrodermal arousal is more consistent with cardiovascular "
                    "or exertional load than psychological stress.")
        return "Heart rate is within the resting range."
    if k == "glucose":
        if g > 0.3 and b > 0.3:
            return (f"Glucose {x['glucose']:.0f} mg/dL together with BMI {x['bmi']:.1f} triggers a compounding "
                    f"metabolic multiplier (×{c['mult']:.2f}): hyperglycaemia on top of excess adiposity "
                    f"signals insulin resistance rather than an isolated lab excursion.")
        if g > 0.3 and e > 0.3:
            return "Elevated glucose with high EDA: stress physiology is adding glycaemic load on a raised baseline."
        if g > 0.3:
            return "Glucose is raised but BMI is normal → scored as early glycaemic drift, without adiposity compounding."
        return "Glucose is within the normal fasting range."
    if k == "bmi":
        if b > 0.3 and g > 0.3:
            return (f"BMI {x['bmi']:.1f} is amplified by elevated glucose (×{c['mult']:.2f}): adiposity is "
                    f"acting on a dysglycaemic baseline.")
        if b > 0.3 and e > 0.3:
            return ("Elevated BMI coupled with elevated EDA amplifies metabolic strain by compounding chronic "
                    "baseline risk with autonomic stress.")
        if b > 0.3:
            return "Elevated BMI with normal glucose: chronic baseline risk that is currently metabolically compensated."
        return "BMI is in the healthy range."
    if k == "sbp":
        if s > 0.3 and b > 0.3:
            return "Raised systolic pressure on top of elevated BMI adds haemodynamic load to an adiposity-driven baseline."
        if s > 0.3 and h > 0.3:
            return "Raised blood pressure with elevated heart rate indicates combined cardiovascular strain."
        if s > 0.3:
            return "Isolated systolic elevation; contributes to cardiovascular load."
        return "Systolic pressure is within the normal range."
    return ""


# =============================================================================
# ===== GMU (PyTorch) =====
# =============================================================================
class DualTowerGMU(nn.Module):
    """Two modality towers fused by a Gated Multimodal Unit: h = z*h_clin + (1-z)*h_wear."""

    def __init__(self, clin_dim=3, wear_dim=3, hidden=16, n_states=8):
        super().__init__()
        self.clin = nn.Sequential(nn.Linear(clin_dim, hidden), nn.Tanh())
        self.wear = nn.Sequential(nn.Linear(wear_dim, hidden), nn.Tanh())
        self.gate = nn.Linear(2 * hidden, hidden)
        self.head = nn.Linear(hidden, n_states)

    def forward(self, xc, xw):
        hc, hw = self.clin(xc), self.wear(xw)
        z = torch.sigmoid(self.gate(torch.cat([hc, hw], dim=-1)))
        h = z * hc + (1 - z) * hw
        return self.head(h), z


@st.cache_resource
def load_gmu():
    torch.manual_seed(7)
    model = DualTowerGMU()
    ckpt = Path(__file__).with_name("gmu_weights.pt")
    trained = False
    if ckpt.exists():
        try:
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            trained = True
        except Exception:
            trained = False
    model.eval()
    return model, trained


# =============================================================================
# ===== UI HELPERS =====
# =============================================================================
PRESETS = {
    "Healthy baseline": dict(glucose=90.0, bmi=22.0, sbp=115.0, hr=65.0, eda=2.0, bvp=1.0),
    "Acute situational stress (normal baselines)": dict(glucose=92.0, bmi=22.5, sbp=118.0, hr=88.0, eda=9.5, bvp=1.9),
    "Glucose + BMI compounding": dict(glucose=165.0, bmi=34.0, sbp=128.0, hr=72.0, eda=2.5, bvp=1.1),
    "Stress on metabolic baseline": dict(glucose=150.0, bmi=33.0, sbp=135.0, hr=90.0, eda=9.0, bvp=1.8),
    "Cardiovascular load": dict(glucose=95.0, bmi=26.0, sbp=150.0, hr=112.0, eda=2.8, bvp=2.0),
    "Critical multimodal": dict(glucose=210.0, bmi=38.0, sbp=165.0, hr=122.0, eda=13.0, bvp=2.6),
    "Sensor dropout (EDA = 0, HR = 0)": dict(glucose=130.0, bmi=29.0, sbp=125.0, hr=0.0, eda=0.0, bvp=1.2),
}


def apply_preset():
    for k, v in PRESETS[st.session_state["preset"]].items():
        st.session_state[k] = v


def show(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, key=key)


def pnoise(idx):
    """Deterministic pseudo-noise in [-0.5, 0.5] from absolute sample index (stable as the window scrolls)."""
    return (np.sin(idx * 12.9898) * 43758.5453) % 1.0 - 0.5


def build_stream(x, motion, n=240):
    """Synthetic 4 Hz window (60 s) clock-synchronised to wall time; driven by the current inputs."""
    end = int(time.time() * 4)
    idx = np.arange(end - n + 1, end + 1)
    t = idx / 4.0
    burst = np.clip(np.sin(2 * np.pi * t / 9.0) * 1.4, 0, 1) if motion else np.zeros(n)
    mnoise = burst * pnoise(idx * 3.1) * 2.0

    def chan(val, tonic_amp, phasic, scale):
        if val <= 0.0:
            return np.zeros(n)  # flat-line = dropout
        y = val * (1 + tonic_amp * np.sin(2 * np.pi * t / 40.0)) + phasic
        y = y + 0.02 * val * pnoise(idx)
        return np.maximum(y + mnoise * scale * (val + 1), 0)

    scr = 0.15 * x["eda"] * np.maximum(0, np.sin(2 * np.pi * t / 13.0)) ** 8
    eda = chan(x["eda"], 0.04, scr, 0.35)
    bvp = chan(x["bvp"], 0.10, 0.0, 0.25)
    hr = chan(x["hr"], 0.02, 2.0 * np.sin(2 * np.pi * t / 20.0), 0.20)
    return t - t[-1], eda, bvp, hr


def render_stream(x, motion):
    t, eda, bvp, hr = build_stream(x, motion)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("EDA (µS) - native 4 Hz", "BVP pulse-amplitude envelope (a.u.)",
                                        "Heart rate (bpm)"))
    for r, (y, col) in enumerate([(eda, "#1e88e5"), (bvp, "#8e24aa"), (hr, "#e53935")], start=1):
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color=col, width=1.6), showlegend=False),
                      row=r, col=1)
        if np.allclose(y, 0):
            fig.add_annotation(text="FLATLINE - sensor dropout", x=-30, y=0, showarrow=False,
                               font=dict(color="#e53935"), row=r, col=1)
    fig.update_xaxes(title_text="seconds (0 = now)", row=3, col=1)
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10))
    show(fig)
    sqi = 0.58 if motion else 0.96
    st.caption(f"Simulated signal quality index: **{sqi:.2f}**"
               f"{' - motion artifact injected; wearable tower down-weighted' if motion else ''}. "
               "WESAD BVP is recorded at 64 Hz; at the 4 Hz synchronised rate only the pulse-amplitude "
               "envelope is shown (the pulse waveform exceeds the 2 Hz Nyquist limit).")


# =============================================================================
# ===== APP =====
# =============================================================================
st.session_state.setdefault("preset", "Healthy baseline")
if "glucose" not in st.session_state:
    apply_preset()

with st.sidebar:
    st.header("Patient inputs")
    st.selectbox("Scenario preset", list(PRESETS), key="preset", on_change=apply_preset)
    st.subheader("Clinical tower (MIMIC-IV)")
    for k in ("glucose", "bmi", "sbp"):
        lo, hi, stp = SLIDER[k]
        st.slider(f"{LABELS[k]} ({UNITS[k]})", lo, hi, step=stp, key=k)
    st.subheader("Wearable tower (WESAD, 4 Hz)")
    for k in ("hr", "eda", "bvp"):
        lo, hi, stp = SLIDER[k]
        st.slider(f"{LABELS[k]} ({UNITS[k]})", lo, hi, step=stp, key=k)
    st.subheader("Judge controls")
    motion = st.toggle("Inject motion-artifact noise", key="motion",
                       help="Corrupts the wearable stream; the engine lowers wearable reliability and widens uncertainty.")
    live = st.toggle("Live 4 Hz stream refresh", value=True)
    st.caption("Tip: set any sensor to 0.0 to see anomaly handling.")

raw = {k: float(st.session_state[k]) for k in FEATURES}
rel = 0.55 if motion else 1.0

# --- anomaly / artifact handling: invalid readings are NOT scored as zero risk ---
bad = validate(raw)
usable = [k for k in FEATURES if k not in bad]
x = {k: (raw[k] if k in usable else BASELINE[k]) for k in FEATURES}  # neutral imputation, excluded from SHAP

c = components(x, rel)
phi = shapley(x, usable, rel)

# --- GMU forward pass (untrained demo weights unless gmu_weights.pt exists) ---
gmu, gmu_trained = load_gmu()
with torch.no_grad():
    xc = torch.tensor([[c["g"], c["b"], c["s"]]], dtype=torch.float32)
    xw = torch.tensor([[c["e"], c["v"], c["h"]]], dtype=torch.float32)
    gmu_logits, gate = gmu(xc, xw)
clin_share = float(gate.mean())

# --- state distribution ---
temp = 0.6 + (0.25 if motion else 0.0)
logits = state_scores(c) / temp
if gmu_trained:
    logits = logits + 0.5 * gmu_logits[0].numpy()
probs = softmax(logits)
uncert = min(0.85, 0.9 * len(bad) / len(FEATURES) + (0.2 if motion else 0.0))
probs = (1 - uncert) * probs + uncert * np.ones(8) / 8
confidence = 1 - uncert
top = int(np.argmax(probs))
risk = c["R"] * 100

# ------------------------------- header --------------------------------------
st.title("🫀 Multimodal Lifestyle Disorder Risk & Explainability Engine")
st.caption("Dual-Tower Gated Multimodal Unit · MIMIC-IV EHR ⊕ WESAD 4 Hz wearable · interaction-aware risk and exact Shapley explanations. "
           "Research prototype - not for clinical use.")

if bad:
    msg = "\n".join(f"- {m}" for m in bad.values())
    st.error(
        "**Physiological anomaly / sensor artifact detected.** Invalid channels are excluded and imputed "
        "neutrally - a 0.0 reading is treated as sensor failure, **not** as zero risk. "
        "The risk score is provisional and the distribution is widened accordingly.\n\n" + msg)
if motion:
    st.warning("Motion artifact injected: wearable tower reliability reduced to 55 %, uncertainty increased.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Predicted state", f"State {top}")
m2.metric("Fused risk score", f"{risk:.0f} / 100")
m3.metric("Data confidence", f"{confidence * 100:.0f} %")
profile = ("Acute situational stress" if c["situational"] else
           "Chronic metabolic" if c["M"] > 0.3 else
           "Cardiovascular load" if c["C"] > 0.3 else
           "Low risk")
m4.metric("Risk profile", profile)

# ------------------------------- 3 columns -----------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("GMU fused risk")
    gcol = "#2e9e5b" if risk < 25 else "#ffc107" if risk < 50 else "#ff9800" if risk < 75 else "#e53935"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk,
        title=dict(text="Provisional" if bad else "Composite risk"),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=gcol),
                   steps=[dict(range=[0, 25], color="#e8f5e9"), dict(range=[25, 50], color="#fff8e1"),
                          dict(range=[50, 75], color="#ffe0b2"), dict(range=[75, 100], color="#ffcdd2")])))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=10))
    show(fig, key="gauge")
    st.progress(clin_share, text=f"GMU gate: clinical tower {clin_share * 100:.0f} % | wearable {100 - clin_share * 100:.0f} %")
    st.caption("Gate from a " + ("trained checkpoint." if gmu_trained else
               "demo (untrained) GMU - architecture visualisation only; add gmu_weights.pt for learned gating."))

with col2:
    st.subheader("8-state distribution")
    fig = go.Figure(go.Bar(
        x=probs * 100, y=[f"S{i}: {STATES[i]['name'].split(' (')[0]}" for i in range(8)], orientation="h",
        marker_color=[STATES[i]["color"] if i == top else "#b0bec5" for i in range(8)],
        text=[f"{p * 100:.1f}%" for p in probs], textposition="outside"))
    fig.update_layout(height=370, margin=dict(l=10, r=40, t=10, b=10), xaxis_title="Probability (%)",
                      yaxis=dict(autorange="reversed"), xaxis_range=[0, max(100 * probs.max() * 1.25, 10)])
    show(fig, key="states")

with col3:
    st.subheader("Live SHAP attribution")
    order = sorted(FEATURES, key=lambda k: abs(phi[k]))
    vals = [phi[k] * 100 for k in order]
    fig = go.Figure(go.Bar(
        x=vals, y=[f"{LABELS[k]} ({TOWER[k][0]})" + (" ⚠" if k in bad else "") for k in order],
        orientation="h", marker_color=["#e53935" if v >= 0 else "#1e88e5" for v in vals],
        text=[f"{v:+.1f}" for v in vals], textposition="outside"))
    fig.update_layout(height=370, margin=dict(l=10, r=40, t=10, b=10), xaxis_title="Risk points vs healthy reference")
    show(fig, key="shap")
    st.caption("Exact Shapley values of the interaction-aware risk function; they sum to the fused risk. "
               "⚠ = excluded artifact channel.")

# ------------------------------- narrative -----------------------------------
st.divider()
left, right = st.columns([1, 1.3])
with left:
    st.subheader(f"State {top}: {STATES[top]['name']}")
    st.markdown(f"**{STATES[top]['summary']}**")
    st.write(STATES[top]["detail"])
with right:
    st.subheader("Why this prediction")
    with st.container(border=True):
        if c["situational"]:
            st.markdown("**Acute situational stress** - wearable arousal is present, but resting clinical baselines "
                        "(BMI, glucose) are normal, so this is not scored as chronic metabolic risk.")
        elif c["M"] > 0.3 and c["A"] > 0.3:
            st.markdown(f"**Stress on a compromised baseline** - autonomic arousal is amplifying a chronic metabolic "
                        f"profile (interaction adds ≈{c['cross'] * 100:.0f} risk points).")
        elif c["M"] > 0.3 and c["g"] > 0.3 and c["b"] > 0.3:
            st.markdown(f"**Compounding metabolic risk** - glucose × BMI multiplier ×{c['mult']:.2f}.")
        elif risk < 15:
            st.markdown("**Low-risk profile** - no meaningful interaction between markers.")
        else:
            st.markdown(f"Risk driven mainly by the **{profile.lower()}** axis.")
        ranked = [k for k in sorted(usable, key=lambda k: -abs(phi[k])) if abs(phi[k]) * 100 >= 1.0][:4]
        for k in ranked:
            st.markdown(f"- **{LABELS[k]}** ({phi[k] * 100:+.1f} pts): {explain_feature(k, x, c)}")
        if not ranked:
            st.markdown("- No feature contributes materially relative to the healthy reference.")
        for k in bad:
            st.markdown(f"- **{LABELS[k]}**: excluded - sensor artifact, not interpreted as normal or zero risk.")

# ------------------------------- expanders -----------------------------------
with st.expander("📡 Live 4 Hz synchronised wearable stream", expanded=False):
    if hasattr(st, "fragment"):
        st.fragment(run_every=1.0 if live else None)(render_stream)(dict(raw), motion)
    else:
        render_stream(dict(raw), motion)

with st.expander("📖 All 8 patient states - clinical definitions"):
    for i in range(8):
        st.markdown(f"**State {i} - {STATES[i]['name']}**  \n*{STATES[i]['summary']}*  \n{STATES[i]['detail']}")

with st.expander("🔬 Engine internals (interaction terms)"):
    st.write({
        "Metabolic axis M (glucose×BMI)": round(c["M"], 3),
        "Glucose×BMI multiplier": round(c["mult"], 3),
        "Autonomic axis A (EDA/BVP)": round(c["A"], 3),
        "Cardiovascular axis C (HR/BVP/SBP)": round(c["C"], 3),
        "Stress context damping factor": round(c["damp"], 3),
        "Stress×metabolic cross term (risk pts)": round(c["cross"] * 100, 2),
        "Situational-stress flag": c["situational"],
        "Wearable reliability": rel,
        "Excluded channels": list(bad) or "none",
    })
