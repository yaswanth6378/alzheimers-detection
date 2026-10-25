"""Full Streamlit interface for the Alzheimer MRI classification academic project.

Features: patient signup/login, password hashing, MRI prediction, confidence scores,
per-user scan history, PDF report download, doctor dashboard, admin user management.

NOTE: Local SQLite storage on Streamlit Community Cloud is ephemeral. Do not use
this prototype to store real patient/clinical information.
"""
from __future__ import annotations

import json
import os
import html
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = Path(tempfile.gettempdir())
DB_PATH = Path(os.environ.get("ALZHEIMER_DB_PATH", str(TEMP_DIR / "alzheimer_streamlit.db")))

CLASS_NAMES = [
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment",
]
MODEL_FILES = {
    "ResNet50": BASE_DIR / "resnet50_alzheimer.keras",
    "Custom CNN": BASE_DIR / "custom_cnn_alzheimer.keras",
}
RESNET50_GDRIVE_ID = "133sc-4AOEn79U1QTL_SNgK6iGvmXVAkg"
ALLOWED_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]
ROLE_OPTIONS = ["patient", "doctor", "admin"]

st.set_page_config(
    page_title="Alzheimer MRI Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Match the original Flask/Bootstrap UI: navy sidebar, pale-grey canvas, white panels, blue section bars.
THEME_ACCENTS = {"Blue": "#2c85b8", "Dark": "#35485c", "Green": "#229b62"}
active_theme = st.session_state.get("theme_choice", "Blue")
accent = THEME_ACCENTS.get(active_theme, THEME_ACCENTS["Blue"])
style_block = """
<style>
  :root { --ink:#273746; --muted:#718096; --accent:ACCENT_PLACEHOLDER; --navy:#1d3e5d; --canvas:#edf2f7; --line:#dfe6ed; --green:#22ae68; }
  .stApp { background:var(--canvas); color:var(--ink); }
  .main .block-container { max-width:100%; padding-top:1.15rem; padding-bottom:2rem; padding-left:1.4rem; padding-right:1.4rem; }
  html, body, [class*="css"] { font-family:Arial, Helvetica, sans-serif; }
  h1,h2,h3,h4 { color:var(--ink); font-family:Arial, Helvetica, sans-serif; }
  h2 { font-size:1.22rem !important; }
  h3 { font-size:1.04rem !important; }
  .hero { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.75rem 1rem; background:#fff; color:var(--ink); border:1px solid var(--line); border-top:4px solid var(--accent); border-radius:8px; margin-bottom:1rem; box-shadow:0 2px 8px rgba(30,55,80,.06); }
  .hero-copy { min-width:0; }
  .hero h1 { color:var(--ink); margin:0 0 .25rem 0; font-size:1.45rem; font-weight:700; }
  .hero p { color:var(--muted); margin:0; font-size:.91rem; }
  .eyebrow { color:var(--accent); font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin-bottom:.15rem; }
  .hero-user { text-align:right; color:#5a6978; font-size:.83rem; min-width:100px; }
  .hero-user strong { color:#2d3b4b; }
  .section-card { background:#fff; border:1px solid var(--line); border-radius:8px; padding:.9rem 1rem; box-shadow:0 2px 8px rgba(30,55,80,.05); margin-bottom:.45rem; }
  .panel-header { background:var(--accent); color:#fff; padding:.58rem .85rem; font-weight:700; border-radius:8px 8px 0 0; margin-top:.25rem; margin-bottom:0; font-size:.96rem; }
  .panel-body { background:#fff; border:1px solid var(--line); border-top:0; border-radius:0 0 8px 8px; padding:.8rem .9rem; }
  .muted { color:var(--muted); }
  .result-pill { display:inline-block; padding:.4rem .9rem; background:#e6f7ed; color:#149653; border:1px solid #a6e7c1; border-radius:999px; font-weight:700; }
  .small-note { color:var(--muted); font-size:.84rem; }
  [data-testid="stSidebar"] { background:var(--navy) !important; border-right:1px solid #17334c; }
  [data-testid="stSidebar"] > div:first-child { background:var(--navy) !important; }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"], [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span { color:#e5edf5; }
  .sidebar-brand { text-align:center; color:#fff; padding:.6rem .2rem .8rem; font-weight:700; font-size:1rem; line-height:1.35; }
  .sidebar-brand .brain { font-size:1.9rem; line-height:1.25; }
  [data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.18); }
  [data-testid="stSidebar"] .stButton > button { background:transparent; color:#dce7f1; border:0; border-radius:7px; text-align:left; justify-content:flex-start; padding:.55rem .7rem; font-size:.9rem; }
  [data-testid="stSidebar"] .stButton > button:hover { background:#294f73; color:#fff; border:0; }
  [data-testid="stSidebar"] .stButton > button[kind="primary"] { background:#2c9dd8; color:#fff; border:0; box-shadow:none; }
  [data-testid="stSidebar"] .stRadio > div { gap:.45rem; }
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#b9cad9 !important; }
  .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button { border-radius:6px; font-weight:600; border-color:#cdd6df; }
  .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] { background:var(--accent); border-color:var(--accent); color:#fff; }
  div.st-key-pdf_action [data-testid="stDownloadButton"] > button { background:#c52e45 !important; border-color:#c52e45 !important; color:#fff !important; }
  div.st-key-new_analysis_action .stButton > button { background:#2d9bd7 !important; border-color:#2d9bd7 !important; color:#fff !important; }
  div.st-key-records_action .stButton > button { background:#f7f9fb !important; border-color:#b9c4cf !important; color:#415263 !important; }
  div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-top:3px solid var(--accent); border-radius:7px; padding:.65rem .8rem; box-shadow:0 2px 8px rgba(30,55,80,.04); }
  div[data-testid="stForm"] { background:#fff; padding:1rem; border:1px solid var(--line); border-radius:8px; }
  [data-testid="stFileUploader"] section { background:#fff; border-color:#cdd7e1; }
  .stat-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.65rem; margin:.25rem 0 1.1rem; }
  .stat-card { border-radius:8px; padding:.85rem 1rem; color:#fff; min-height:82px; box-shadow:0 2px 8px rgba(30,55,80,.10); }
  .stat-number { font-size:1.7rem; line-height:1.2; font-weight:800; }
  .stat-label { font-size:.83rem; opacity:.96; margin-top:.28rem; }
  .stat-blue { background:#318fc0; } .stat-green { background:#28b96b; } .stat-orange { background:#e98b18; } .stat-red { background:#c93a37; }
  @media (max-width: 850px) { .stat-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .hero { align-items:flex-start; } }
</style>
""".replace("ACCENT_PLACEHOLDER", accent)
st.markdown(style_block, unsafe_allow_html=True)


@contextmanager
def db_connect():
    """Yield a SQLite connection and always close it after each operation."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH), timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    with db_connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'patient',
                created_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                image_name TEXT NOT NULL,
                image_bytes BLOB,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities_json TEXT NOT NULL,
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )"""
        )
        conn.commit()

    # Optional secure first-admin bootstrap. Configure in Streamlit Cloud > App settings > Secrets.
    admin_email = _secret("ADMIN_EMAIL")
    admin_password = _secret("ADMIN_PASSWORD")
    admin_name = _secret("ADMIN_NAME") or "Project Administrator"
    if admin_email and admin_password:
        with db_connect() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE email = ?", (admin_email.strip().lower(),)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO users (name,email,password_hash,role,created_at) VALUES (?,?,?,?,?)",
                    (admin_name.strip(), admin_email.strip().lower(), generate_password_hash(admin_password), "admin", now_iso()),
                )
                conn.commit()


def _secret(key: str) -> str | None:
    try:
        value = st.secrets.get(key)
        if value is not None and str(value).strip():
            return str(value)
    except Exception:
        pass
    value = os.getenv(key)
    return value if value and value.strip() else None


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def get_user_by_email(email: str) -> sqlite3.Row | None:
    with db_connect() as conn:
        return conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()


def get_user_by_id(user_id: int) -> sqlite3.Row | None:
    with db_connect() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()


def create_user(name: str, email: str, password: str) -> tuple[bool, str]:
    name, email = name.strip(), email.strip().lower()
    if len(name) < 2:
        return False, "Please enter your full name."
    if "@" not in email or "." not in email.split("@")[-1]:
        return False, "Enter a valid email address."
    if len(password) < 8:
        return False, "Use a password with at least 8 characters."
    try:
        with db_connect() as conn:
            conn.execute(
                "INSERT INTO users (name,email,password_hash,role,created_at) VALUES (?,?,?,?,?)",
                (name, email, generate_password_hash(password), "patient", now_iso()),
            )
            conn.commit()
        return True, "Account created. Please sign in."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists. Please sign in instead."


def list_user_scans(user_id: int | None = None, limit: int | None = None) -> list[sqlite3.Row]:
    query = """SELECT s.*, u.name AS user_name, u.email AS user_email
               FROM scans s JOIN users u ON u.id = s.user_id"""
    args: list[Any] = []
    if user_id is not None:
        query += " WHERE s.user_id = ?"
        args.append(int(user_id))
    query += " ORDER BY s.created_at DESC"
    if limit is not None:
        query += " LIMIT ?"
        args.append(int(limit))
    with db_connect() as conn:
        return conn.execute(query, args).fetchall()


def save_scan(user_id: int, image_name: str, image_bytes: bytes, prediction: str,
              confidence: float, probabilities: dict[str, float], notes: str) -> int:
    with db_connect() as conn:
        cursor = conn.execute(
            """INSERT INTO scans
               (user_id,image_name,image_bytes,prediction,confidence,probabilities_json,notes,created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (int(user_id), image_name, sqlite3.Binary(image_bytes), prediction,
             float(confidence), json.dumps(probabilities), notes.strip(), now_iso()),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_scan(scan_id: int, allowed_user_id: int | None = None) -> sqlite3.Row | None:
    query = """SELECT s.*, u.name AS user_name, u.email AS user_email, u.role AS user_role
               FROM scans s JOIN users u ON u.id=s.user_id WHERE s.id=?"""
    args: list[Any] = [int(scan_id)]
    if allowed_user_id is not None:
        query += " AND s.user_id=?"
        args.append(int(allowed_user_id))
    with db_connect() as conn:
        return conn.execute(query, args).fetchone()


def model_file_is_real(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < 1000:
            return False
        with path.open("rb") as f:
            return b"version https://git-lfs.github.com/spec/v1" not in f.read(160)
    except OSError:
        return False


def resnet_download_path() -> Path:
    path = TEMP_DIR / "alzheimer_model_cache" / "resnet50_alzheimer.keras"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@st.cache_resource(max_entries=1, show_spinner=False)
def load_model(model_name: str):
    path = MODEL_FILES[model_name]
    if not model_file_is_real(path):
        if model_name == "Custom CNN":
            raise FileNotFoundError("custom_cnn_alzheimer.keras is missing or is only a Git LFS pointer. Upload the complete model file to GitHub.")
        path = resnet_download_path()
        if not model_file_is_real(path):
            import gdown
            downloaded = gdown.download(
                f"https://drive.google.com/uc?id={RESNET50_GDRIVE_ID}", str(path), quiet=False
            )
            if not downloaded or not model_file_is_real(path):
                raise RuntimeError("ResNet50 model could not be downloaded. Check that its Google Drive file is accessible by link.")
    from tensorflow.keras.models import load_model as keras_load_model
    return keras_load_model(str(path), compile=False)


def prepare_image(image: Image.Image) -> np.ndarray:
    resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    return np.expand_dims(np.asarray(resized, dtype=np.float32) / 255.0, axis=0)


def normalize_probabilities(output: np.ndarray) -> np.ndarray:
    scores = np.asarray(output, dtype=np.float64).reshape(-1)
    if len(scores) != len(CLASS_NAMES):
        raise ValueError(f"Model returned {len(scores)} scores, but {len(CLASS_NAMES)} classes are expected.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Model output contains invalid numbers.")
    if np.all(scores >= 0) and np.isclose(scores.sum(), 1.0, atol=0.05):
        return scores
    scores = scores - scores.max()
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum()


def generate_pdf(scan: sqlite3.Row) -> bytes:
    from fpdf import FPDF

    probs = json.loads(scan["probabilities_json"] or "{}")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_fill_color(30, 66, 125)
    pdf.rect(0, 0, 210, 34, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(12, 8)
    pdf.cell(186, 9, "Alzheimer MRI Classification Report", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(12, 20)
    pdf.cell(186, 6, "Academic research prototype - AI-generated image classification", align="C")

    pdf.set_text_color(30, 40, 55)
    pdf.set_xy(15, 43)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(180, 8, "User and Scan Details", ln=1)
    pdf.set_font("Helvetica", "", 11)
    for label, value in [
        ("Name", scan["user_name"]),
        ("Email", scan["user_email"]),
        ("Record ID", f"#{scan['id']:04d}"),
        ("Date", scan["created_at"]),
        ("Uploaded image", scan["image_name"]),
    ]:
        pdf.set_x(15)
        safe_value = str(value or "").encode("cp1252", "replace").decode("cp1252")
        pdf.cell(45, 7, f"{label}:")
        pdf.multi_cell(135, 7, safe_value)

    pdf.ln(4)
    pdf.set_draw_color(35, 101, 209)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_x(15)
    pdf.cell(180, 8, "Model Output", ln=1)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_x(15)
    pdf.set_text_color(35, 101, 209)
    pdf.cell(180, 10, str(scan["prediction"]), ln=1)
    pdf.set_text_color(30, 40, 55)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_x(15)
    pdf.cell(180, 8, f"Model confidence score: {float(scan['confidence']):.2f}%", ln=1)

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(180, 8, "Scores across the four categories", ln=1)
    pdf.set_font("Helvetica", "", 11)
    for category in CLASS_NAMES:
        pdf.cell(125, 7, category)
        pdf.cell(50, 7, f"{float(probs.get(category, 0.0)):.2f}%", align="R", ln=1)

    notes = str(scan["notes"] or "").strip()
    if notes:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(180, 8, "Notes", ln=1)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(180, 6, notes.encode("cp1252", "replace").decode("cp1252"))

    pdf.ln(6)
    pdf.set_fill_color(242, 246, 253)
    pdf.set_font("Helvetica", "I", 9)
    disclaimer = (
        "Important: This output is produced by an academic AI prototype. The confidence score is not the model's overall accuracy, "
        "and the output is not a diagnosis or treatment recommendation. Consult a qualified medical professional for clinical interpretation."
    )
    pdf.multi_cell(180, 5, disclaimer, border=0, fill=True)
    out = pdf.output()
    return bytes(out) if not isinstance(out, str) else out.encode("cp1252", "replace")


def show_hero(title: str, subtitle: str) -> None:
    safe_title = html.escape(str(title))
    safe_subtitle = html.escape(str(subtitle))
    username = html.escape(str(st.session_state.get("display_name", "")))
    role = html.escape(str(st.session_state.get("display_role", "")))
    user_block = (
        f'<div class="hero-user">👤 <strong>{username}</strong><br><span>{role}</span></div>'
        if username else ""
    )
    st.markdown(
        f'<div class="hero"><div class="hero-copy"><div class="eyebrow">Alzheimer’s Detection</div>'
        f'<h1>{safe_title}</h1><p>{safe_subtitle}</p></div>{user_block}</div>',
        unsafe_allow_html=True,
    )


def panel_header(title: str) -> None:
    st.markdown(f'<div class="panel-header">{html.escape(str(title))}</div>', unsafe_allow_html=True)


def show_auth() -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">Deep Learning Research Project</div><h1>Alzheimer’s MRI Classification</h1><p>Sign in to upload MRI images, view model outputs, and download reports.</p></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.subheader("Welcome")
        st.write("This web application demonstrates a custom CNN and ResNet50 model trained for four MRI dataset categories.")
        st.markdown("- Secure password hashing\n- MRI upload and prediction\n- Scan history and PDF reports\n- Role-based Patient, Doctor and Admin views")
        st.warning("Academic prototype only. Not for medical diagnosis or real patient-care decisions.")
    with right:
        login_tab, register_tab = st.tabs(["Sign in", "Create account"])
        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email address", key="login_email").strip().lower()
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if submitted:
                user = get_user_by_email(email)
                if user and check_password_hash(user["password_hash"], password):
                    st.session_state["user_id"] = int(user["id"])
                    st.session_state.pop("active_page", None)
                    st.rerun()
                st.error("Email or password is incorrect.")
        with register_tab:
            with st.form("register_form"):
                name = st.text_input("Full name", key="register_name")
                email = st.text_input("Email address", key="register_email").strip().lower()
                password = st.text_input("Password (at least 8 characters)", type="password", key="register_password")
                confirm = st.text_input("Confirm password", type="password", key="register_confirm")
                submitted = st.form_submit_button("Create patient account", type="primary", use_container_width=True)
            if submitted:
                if password != confirm:
                    st.error("The passwords do not match.")
                else:
                    ok, message = create_user(name, email, password)
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)
            st.caption("New registrations receive Patient access. Doctor/Admin access must be assigned by an administrator.")


def show_dashboard(user: sqlite3.Row) -> None:
    show_hero(f"Welcome, {user['name'].split()[0]}", "Your personal workspace for MRI classification records and reports.")
    records = list_user_scans(int(user["id"]))
    total = len(records)
    counts = {c: sum(1 for r in records if r["prediction"] == c) for c in CLASS_NAMES}
    mild_family = counts["Mild Impairment"] + counts["Very Mild Impairment"]
    st.markdown(
        '<div class="stat-grid">'
        f'<div class="stat-card stat-blue"><div class="stat-number">{total}</div><div class="stat-label">Total Scans</div></div>'
        f'<div class="stat-card stat-green"><div class="stat-number">{counts["No Impairment"]}</div><div class="stat-label">No Impairment</div></div>'
        f'<div class="stat-card stat-orange"><div class="stat-number">{mild_family}</div><div class="stat-label">Mild / Very Mild</div></div>'
        f'<div class="stat-card stat-red"><div class="stat-number">{counts["Moderate Impairment"]}</div><div class="stat-label">Moderate</div></div>'
        '</div>', unsafe_allow_html=True,
    )
    st.subheader("Quick Actions")
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="section-card"><h3>🧠 Analyse MRI</h3><p>Upload a brain MRI image and generate a model output.</p></div>', unsafe_allow_html=True)
        if st.button("Analyse MRI", key="dashboard_upload", type="primary", use_container_width=True):
            st.session_state["active_page"] = "Analyse MRI"
            st.rerun()
    with b:
        st.markdown('<div class="section-card"><h3>🕘 My Records</h3><p>Review saved scans and download PDF reports.</p></div>', unsafe_allow_html=True)
        if st.button("View My Records", key="dashboard_history", use_container_width=True):
            st.session_state["active_page"] = "My Records"
            st.rerun()
    with c:
        st.markdown('<div class="section-card"><h3>👨‍⚕️ Doctor View</h3><p>Authorized users can review recent scan records.</p></div>', unsafe_allow_html=True)
        if user["role"] in ("doctor", "admin"):
            if st.button("Open Doctor View", key="dashboard_doctor", use_container_width=True):
                st.session_state["active_page"] = "Doctor View"
                st.rerun()
        else:
            st.caption("Doctor access is role-restricted.")
    st.caption("Uppalapalli Yaswanth Reddy · 23SE02ML212 · MLAI 7C")
    if records:
        st.subheader("Recent scans")
        for scan in records[:5]:
            with st.container(border=True):
                x, y, z = st.columns([2, 1, 1])
                x.write(f"**{scan['prediction']}**")
                x.caption(scan["created_at"])
                y.metric("Confidence", f"{float(scan['confidence']):.1f}%")
                if z.button("View report", key=f"dashboard_view_{scan['id']}"):
                    st.session_state["view_scan_id"] = int(scan["id"])
                    st.session_state["active_page"] = "My Records"
                    st.rerun()


def show_upload(user: sqlite3.Row) -> None:
    show_hero("MRI upload & prediction", "Upload an image, choose a trained model, and review the output.")
    info, caution = st.columns([1, 1])
    with info:
        st.markdown('<div class="section-card"><b>Input requirements</b><p>Supported: PNG, JPG, JPEG, BMP, TIFF, WEBP</p><p>Image is resized to 224 × 224 pixels for model inference.</p></div>', unsafe_allow_html=True)
    with caution:
        st.markdown('<div class="section-card"><b>Important note</b><p>The output is for academic demonstration only and is not a clinical diagnosis.</p></div>', unsafe_allow_html=True)

    available = [name for name, path in MODEL_FILES.items() if model_file_is_real(path)]
    if not available:
        available = ["ResNet50"]  # model fallback may retrieve the ResNet model from Drive
    model_name = st.selectbox("Choose model", available)
    uploaded = st.file_uploader("Upload a brain MRI image", type=ALLOWED_TYPES, key="mri_uploader")
    notes = st.text_area("Optional notes", max_chars=1500, help="Avoid entering sensitive health information in this academic demo.")

    if uploaded is not None:
        try:
            image_bytes = uploaded.getvalue()
            image = Image.open(uploaded).convert("RGB")
            img_col, details_col = st.columns([1, 1])
            with img_col:
                st.image(image, caption="Selected MRI image", use_container_width=True)
            with details_col:
                st.write(f"**File:** {uploaded.name}")
                st.write(f"**Dimensions:** {image.width} × {image.height}")
                st.write(f"**Model:** {model_name}")
                st.write("**Output classes:**")
                for category in CLASS_NAMES:
                    st.write(f"• {category}")
            if st.button("Classify MRI", type="primary", use_container_width=True):
                try:
                    with st.spinner("Loading model and generating prediction…"):
                        model = load_model(model_name)
                        raw = model.predict(prepare_image(image), verbose=0)
                        probs_array = normalize_probabilities(raw)
                    idx = int(np.argmax(probs_array))
                    probs = {CLASS_NAMES[i]: float(probs_array[i] * 100.0) for i in range(len(CLASS_NAMES))}
                    scan_id = save_scan(
                        int(user["id"]), uploaded.name, image_bytes,
                        CLASS_NAMES[idx], probs[CLASS_NAMES[idx]], probs, notes,
                    )
                    st.session_state["view_scan_id"] = scan_id
                    st.session_state["active_page"] = "My Records"
                    st.rerun()
                except Exception as exc:
                    st.error("Prediction could not be completed. Check that the model files are complete and the hosting service has enough memory.")
                    st.code(f"{type(exc).__name__}: {exc}")
        except (UnidentifiedImageError, OSError):
            st.error("The selected file is not a valid readable image. Please upload a supported MRI image.")


def show_scan_result(scan: sqlite3.Row) -> None:
    # Layout follows the original Flask page: image on the left, diagnosis and
    # stage probabilities on the right, and the three original actions below.
    show_hero("Analysis Result", "MRI classification result and confidence scores.")
    left, right = st.columns([0.94, 1.46], gap="medium")
    probs = json.loads(scan["probabilities_json"] or "{}")
    with left:
        panel_header("🖼  Uploaded MRI")
        with st.container(border=True):
            if scan["image_bytes"]:
                try:
                    st.image(Image.open(sqlite3_blob_file(scan["image_bytes"])), caption=scan["image_name"], use_container_width=True)
                except Exception:
                    st.caption(f"Uploaded file: {scan['image_name']}")
            else:
                st.info("No image preview is saved for this record.")
            st.caption(f"Analysed on {scan['created_at']}")
    with right:
        panel_header("♟  Diagnosis")
        with st.container(border=True):
            st.markdown(f'<div style="text-align:center;padding:12px"><span class="result-pill">● {html.escape(str(scan["prediction"]))}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="text-align:center;color:#5c6b7a;padding:2px 0 8px">Confidence: <strong style="color:#22ae68">{float(scan["confidence"]):.2f}%</strong></div>', unsafe_allow_html=True)
            st.progress(max(0.0, min(1.0, float(scan["confidence"]) / 100.0)))
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        panel_header("▤  Stage Probabilities")
        with st.container(border=True):
            for category, color in zip(CLASS_NAMES, ["#df7a18", "#c84343", "#24aa68", "#d39b09"]):
                score = float(probs.get(category, 0.0))
                col_label, col_score = st.columns([4, 1])
                col_label.markdown(f"**{category}**")
                col_score.markdown(f"<div style='text-align:right;color:{color};font-weight:700'>{score:.1f}%</div>", unsafe_allow_html=True)
                st.progress(max(0.0, min(1.0, score / 100.0)))
    st.info("A model confidence score is not the same as overall model accuracy. This academic prototype has not been clinically validated and must not be used for diagnosis or treatment decisions.")
    pdf = generate_pdf(scan)
    pdf_col, new_col, records_col = st.columns([1.25, 1, 1])
    with pdf_col:
        with st.container(key="pdf_action"):
            st.download_button(
                "📄 Download PDF Report",
                data=pdf,
                file_name=f"Alzheimer_MRI_Report_{int(scan['id'])}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    with new_col:
        with st.container(key="new_analysis_action"):
            if st.button("＋ New Analysis", key=f"new_analysis_{scan['id']}", use_container_width=True):
                st.session_state["active_page"] = "Analyse MRI"
                st.rerun()
    with records_col:
        with st.container(key="records_action"):
            if st.button("⟲ All Records", key=f"all_records_{scan['id']}", use_container_width=True):
                st.session_state["active_page"] = "My Records"
                st.rerun()

def sqlite_blob_to_image_bytes(blob: bytes) -> bytes:
    return bytes(blob)


def sqlite3_blob_file(blob: bytes):
    # PIL can read a BytesIO object without writing private image data to disk.
    from io import BytesIO
    return BytesIO(sqlite_blob_to_image_bytes(blob))


def show_history(user: sqlite3.Row) -> None:
    show_hero("Scan history", "Review previous predictions and download individual PDF reports.")
    records = list_user_scans(int(user["id"]))
    if not records:
        st.info("You have no saved scans yet. Open Upload & Predict to classify an MRI image.")
        if st.button("Upload an MRI now", type="primary"):
            st.session_state["active_page"] = "Analyse MRI"
            st.rerun()
        return
    selected_id = st.session_state.get("view_scan_id")
    record_ids = [int(r["id"]) for r in records]
    default_idx = record_ids.index(selected_id) if selected_id in record_ids else 0
    options = {f"#{int(r['id']):04d} · {r['created_at']} · {r['prediction']}": int(r["id"]) for r in records}
    labels = list(options.keys())
    picked_label = st.selectbox("Select a scan", labels, index=default_idx)
    scan = get_scan(options[picked_label], allowed_user_id=int(user["id"]))
    if not scan:
        st.error("This scan record is not available for your account.")
        return
    show_scan_result(scan)
    with st.expander("Delete this scan record"):
        st.warning("Deleting a scan cannot be undone.")
        if st.button("Delete selected record", key=f"delete_scan_{scan['id']}"):
            with db_connect() as conn:
                conn.execute("DELETE FROM scans WHERE id=? AND user_id=?", (int(scan["id"]), int(user["id"])))
                conn.commit()
            st.session_state.pop("view_scan_id", None)
            st.success("Scan record deleted.")
            st.rerun()


def show_doctor_dashboard() -> None:
    show_hero("Doctor View", "View patient accounts, scan counts, and recent model outputs.")
    with db_connect() as conn:
        patient_count = conn.execute("SELECT COUNT(*) FROM users WHERE role='patient'").fetchone()[0]
        total_scans = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    a, b = st.columns(2)
    a.metric("Registered patients", int(patient_count))
    b.metric("Total saved scans", int(total_scans))
    scans = list_user_scans()
    if scans:
        st.subheader("Recent patient scans")
        for scan in scans[:100]:
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                col1.write(f"**{scan['user_name']}**")
                col1.caption(scan["user_email"])
                col2.write(scan["prediction"])
                col2.caption(scan["created_at"])
                col3.metric("Confidence", f"{float(scan['confidence']):.1f}%")
                col4.download_button(
                    "PDF",
                    data=generate_pdf(scan),
                    file_name=f"Alzheimer_Report_{int(scan['id'])}.pdf",
                    mime="application/pdf",
                    key=f"doctor_pdf_{scan['id']}",
                )
    else:
        st.info("No scan records have been saved yet.")
    st.caption("Only use sample/demo records. Do not store real patient data in this prototype.")


def show_admin_panel(current_user: sqlite3.Row) -> None:
    show_hero("Admin panel", "Manage registered accounts and assign authorized roles.")
    with db_connect() as conn:
        users = conn.execute("SELECT id,name,email,role,created_at FROM users ORDER BY created_at DESC").fetchall()
    if not users:
        st.info("No registered users found.")
        return
    st.write("New registrations are assigned Patient role by default. Grant elevated access only to trusted demo accounts.")
    for account in users:
        with st.container(border=True):
            c1, c2, c3 = st.columns([2.2, 1, 1])
            c1.write(f"**{account['name']}**")
            c1.caption(f"{account['email']} · joined {account['created_at']}")
            role_options = ROLE_OPTIONS
            role_idx = role_options.index(account["role"]) if account["role"] in role_options else 0
            new_role = c2.selectbox("Role", role_options, index=role_idx, key=f"role_select_{account['id']}")
            if c3.button("Save role", key=f"save_role_{account['id']}", disabled=int(account["id"]) == int(current_user["id"])):
                with db_connect() as conn:
                    conn.execute("UPDATE users SET role=? WHERE id=?", (new_role, int(account["id"])))
                    conn.commit()
                st.success(f"Updated role for {account['name']}.")
                st.rerun()
            if int(account["id"]) != int(current_user["id"]) and account["role"] != "admin":
                if c3.button("Delete user", key=f"delete_user_{account['id']}"):
                    with db_connect() as conn:
                        conn.execute("DELETE FROM users WHERE id=?", (int(account["id"]),))
                        conn.commit()
                    st.success("User and associated scan records deleted.")
                    st.rerun()


def show_about() -> None:
    show_hero("About this project", "Alzheimer’s Disease Stage Classification from Brain MRI Images using CNN")
    st.markdown("""
    <div class="section-card">
      <h3>Project details</h3>
      <p><b>Student:</b> Uppalapalli Yaswanth Reddy</p>
      <p><b>Enrollment:</b> 23SE02ML212</p>
      <p><b>Program:</b> MLAI 7C</p>
      <p><b>Institution:</b> P P Savani University, School of Engineering</p>
      <p><b>Models:</b> Custom CNN and ResNet50 transfer learning</p>
      <p><b>Stack:</b> Python, TensorFlow/Keras, Streamlit, SQLite, fpdf2</p>
      <p><b>Source code:</b> <a href="https://github.com/yaswanth6378/alzheimers-detection" target="_blank">GitHub repository</a></p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Classification categories")
    for name in CLASS_NAMES:
        st.write(f"- {name}")
    st.warning("Research/educational prototype only. It has not been clinically validated and is not a substitute for a qualified clinician.")


init_db()

if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

current_user = get_user_by_id(st.session_state["user_id"]) if st.session_state.get("user_id") else None
if current_user is None:
    st.session_state["user_id"] = None
    show_auth()
    st.stop()

# Sidebar navigation follows the original Flask application's menu structure.
st.session_state["display_name"] = current_user["name"]
st.session_state["display_role"] = str(current_user["role"]).title()
with st.sidebar:
    st.markdown("<div class='sidebar-brand'><div class='brain'>🧠</div>Alzheimer's<br>Detection</div>", unsafe_allow_html=True)
    st.divider()
    st.caption(f"{current_user['name']} · {str(current_user['role']).title()}")
    pages = ["Dashboard", "Analyse MRI", "My Records"]
    if current_user["role"] in ("doctor", "admin"):
        pages.append("Doctor View")
    if current_user["role"] == "admin":
        pages.append("Admin Panel")
    if st.session_state.get("active_page") not in pages:
        st.session_state["active_page"] = "Dashboard"
    nav_labels = {
        "Dashboard": "⌂  Dashboard",
        "Analyse MRI": "♧  Analyse MRI",
        "My Records": "◴  My Records",
        "Doctor View": "♟  Doctor View",
        "Admin Panel": "⚙  Admin Panel",
    }
    for page in pages:
        if st.button(
            nav_labels.get(page, page),
            key=f"nav_button_{page.replace(' ', '_').replace('&', 'and')}",
            type="primary" if st.session_state["active_page"] == page else "secondary",
            use_container_width=True,
        ):
            st.session_state["active_page"] = page
            st.rerun()
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.caption("Theme")
    st.radio("Theme", list(THEME_ACCENTS.keys()), horizontal=True, key="theme_choice", label_visibility="collapsed")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("↪ Logout", key="sign_out_button", use_container_width=True):
        st.session_state["user_id"] = None
        st.session_state.pop("view_scan_id", None)
        st.session_state.pop("display_name", None)
        st.session_state.pop("display_role", None)
        st.session_state["active_page"] = "Dashboard"
        st.rerun()
    st.caption("Academic demo · Not for clinical use")

active_page = st.session_state.get("active_page", "Dashboard")
if active_page == "Dashboard":
    show_dashboard(current_user)
elif active_page == "Analyse MRI":
    show_upload(current_user)
elif active_page == "My Records":
    show_history(current_user)
elif active_page == "Doctor View":
    show_doctor_dashboard()
elif active_page == "Admin Panel":
    show_admin_panel(current_user)

st.divider()
st.caption("Alzheimer’s MRI Classification · CNN & ResNet50 · Educational research prototype · Not a clinical diagnostic tool")
