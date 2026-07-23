# =============================================================
#   Alzheimer's Detection Web App
#   Features: Login/Register, Doctor & Patient Dashboard,
#             MRI Upload & Prediction, Patient History,
#             PDF Report Download, Admin Panel
# =============================================================

from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, send_file)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from fpdf import FPDF
import numpy as np
import cv2
import os
import uuid
import gdown

# ─────────────────────────────────────────────────────────────
# APP CONFIGURATION & PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'alzheimer_secret_key_2024')
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'alzheimer.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'static', 'reports'), exist_ok=True)

db = SQLAlchemy(app)

# ─────────────────────────────────────────────────────────────
# DATABASE MODELS
# ─────────────────────────────────────────────────────────────


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='patient')  # patient/doctor/admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    records = db.relationship('Record', backref='user', lazy=True)


class Record(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(200))
    prediction = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    mild_prob = db.Column(db.Float)
    moderate_prob = db.Column(db.Float)
    no_prob = db.Column(db.Float)
    verymild_prob = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text, default='')


# ─────────────────────────────────────────────────────────────
# LOAD MODEL & AUTO-DOWNLOAD
# ─────────────────────────────────────────────────────────────
model = None
CLASS_NAMES = ['Mild Impairment', 'Moderate Impairment',
               'No Impairment',   'Very Mild Impairment']

STAGE_INFO = {
    'No Impairment': ('No Impairment',       '#27ae60', '🟢'),
    'Very Mild Impairment': ('Very Mild Impairment', '#f1c40f', '🟡'),
    'Mild Impairment': ('Mild Impairment',      '#e67e22', '🟠'),
    'Moderate Impairment': ('Moderate Impairment',  '#e74c3c', '🔴'),
}

# Google Drive File ID for resnet50_alzheimer.keras
GDRIVE_FILE_ID = "133sc-4AOEn79U1QTL_SNgK6iGvmXVAkg"
MODEL_PATH = os.path.join(BASE_DIR, 'resnet50_alzheimer.keras')


def load_model():
    global model
    try:
        from tensorflow.keras.models import load_model as keras_load

        # Download model from Google Drive if missing or if it's an un-pulled Git LFS text pointer
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
            print("⏳ Downloading model weights from Google Drive...")
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            print("✅ Model download complete!")

        if os.path.exists(MODEL_PATH):
            model = keras_load(MODEL_PATH, compile=False)
            print("✅ ResNet50 model loaded successfully!")
        else:
            print("⚠️  No model file found.")
    except Exception as e:
        print(f"Model load error: {e}")


def predict(image_path):
    if model is None:
        return None, 0, {}
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    arr = np.expand_dims(img / 255.0, axis=0).astype(np.float32)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    probs = {CLASS_NAMES[i]: float(preds[i] * 100)
             for i in range(len(CLASS_NAMES))}
    return CLASS_NAMES[idx], float(preds[idx] * 100), probs


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
ALLOWED = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED


def login_required(f):
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def role_required(*roles):
    from functools import wraps

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            user = User.query.get(session['user_id'])
            if user.role not in roles:
                flash('Access denied.', 'danger')
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated
    return decorator

# ─────────────────────────────────────────────────────────────
# ROUTES — AUTH
# ─────────────────────────────────────────────────────────────


@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['name'] = user.name
            session['role'] = user.role
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        role = request.form.get('role', 'patient')
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return render_template('register.html')
        user = User(
            name=name,
            email=email,
            password=generate_password_hash(password),
            role=role)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

# ─────────────────────────────────────────────────────────────
# ROUTES — DASHBOARD
# ─────────────────────────────────────────────────────────────


@app.route('/dashboard')
@login_required
def dashboard():
    user = User.query.get(session['user_id'])
    records = Record.query.filter_by(
        user_id=user.id).order_by(
        Record.created_at.desc()).limit(5).all()
    total = Record.query.filter_by(user_id=user.id).count()

    stats = {
        'total': total,
        'no': Record.query.filter_by(
            user_id=user.id,
            prediction='No Impairment').count(),
        'mild': Record.query.filter_by(
            user_id=user.id,
            prediction='Mild Impairment').count(),
        'moderate': Record.query.filter_by(
            user_id=user.id,
            prediction='Moderate Impairment').count(),
        'verymild': Record.query.filter_by(
            user_id=user.id,
            prediction='Very Mild Impairment').count(),
    }
    return render_template('dashboard.html',
                           user=user,
                           records=records,
                           stats=stats,
                           stage_info=STAGE_INFO)

# ─────────────────────────────────────────────────────────────
# ROUTES — PREDICT
# ─────────────────────────────────────────────────────────────


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected.', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = str(uuid.uuid4()) + '_' + \
                secure_filename(file.filename)
            filepath = os.path.join(
                app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            pred, conf, probs = predict(filepath)

            if pred is None:
                flash('Model not loaded. Check model weights.', 'danger')
                return redirect(request.url)

            notes = request.form.get('notes', '')
            record = Record(
                user_id=session['user_id'],
                image_path=filename,
                prediction=pred,
                confidence=conf,
                mild_prob=probs.get('Mild Impairment', 0),
                moderate_prob=probs.get('Moderate Impairment', 0),
                no_prob=probs.get('No Impairment', 0),
                verymild_prob=probs.get('Very Mild Impairment', 0),
                notes=notes)
            db.session.add(record)
            db.session.commit()

            return redirect(url_for('result', record_id=record.id))
        flash('Invalid file type. Use JPG/PNG.', 'danger')
    return render_template('predict.html')


@app.route('/result/<int:record_id>')
@login_required
def result(record_id):
    record = Record.query.get_or_404(record_id)
    user = User.query.get(session['user_id'])
    probs = {
        'Mild Impairment': record.mild_prob,
        'Moderate Impairment': record.moderate_prob,
        'No Impairment': record.no_prob,
        'Very Mild Impairment': record.verymild_prob,
    }
    info = STAGE_INFO.get(record.prediction,
                          ('Unknown', '#888', '❓'))
    return render_template('result.html',
                           record=record,
                           user=user,
                           probs=probs,
                           info=info)

# ─────────────────────────────────────────────────────────────
# ROUTES — HISTORY
# ─────────────────────────────────────────────────────────────


@app.route('/history')
@login_required
def history():
    user = User.query.get(session['user_id'])
    records = Record.query.filter_by(
        user_id=user.id).order_by(
        Record.created_at.desc()).all()
    return render_template('history.html',
                           records=records,
                           user=user,
                           stage_info=STAGE_INFO)


@app.route('/delete_record/<int:record_id>')
@login_required
def delete_record(record_id):
    record = Record.query.get_or_404(record_id)
    if record.user_id == session['user_id']:
        db.session.delete(record)
        db.session.commit()
        flash('Record deleted.', 'info')
    return redirect(url_for('history'))

# ─────────────────────────────────────────────────────────────
# ROUTES — PDF REPORT
# ─────────────────────────────────────────────────────────────


@app.route('/report/<int:record_id>')
@login_required
def download_report(record_id):
    record = Record.query.get_or_404(record_id)
    user = User.query.get(session['user_id'])

    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(41, 128, 185)
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 20)
    pdf.set_xy(0, 8)
    pdf.cell(210, 10, "Alzheimer's Detection Report",
             align='C', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.cell(210, 8,
             'AI-Powered MRI Analysis System',
             align='C', ln=True)

    # Patient Info
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(15, 45)
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 8, 'Patient Information', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.set_x(15)
    pdf.cell(0, 7, f'Name    : {user.name}', ln=True)
    pdf.set_x(15)
    pdf.cell(0, 7, f'Email   : {user.email}', ln=True)
    pdf.set_x(15)
    pdf.cell(0, 7,
             f'Date    : {record.created_at.strftime("%d %B %Y %I:%M %p")}',
             ln=True)
    pdf.set_x(15)
    pdf.cell(0, 7, f'Report ID: #{record.id:04d}', ln=True)

    # Divider
    pdf.set_draw_color(41, 128, 185)
    pdf.set_line_width(0.5)
    pdf.line(15, pdf.get_y() + 3, 195, pdf.get_y() + 3)
    pdf.ln(8)

    # Result
    pdf.set_x(15)
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 8, 'Diagnosis Result', ln=True)
    pdf.set_x(15)
    pdf.set_font('Arial', 'B', 16)

    color_map = {
        'No Impairment': (39, 174, 96),
        'Very Mild Impairment': (241, 196, 15),
        'Mild Impairment': (230, 126, 34),
        'Moderate Impairment': (231, 76, 60),
    }
    r, g, b = color_map.get(record.prediction, (0, 0, 0))
    pdf.set_text_color(r, g, b)
    pdf.cell(0, 10, f'Stage: {record.prediction}', ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_x(15)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8,
             f'Confidence: {record.confidence:.1f}%', ln=True)

    # Probabilities
    pdf.ln(3)
    pdf.set_x(15)
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 8, 'Stage Probabilities', ln=True)
    pdf.set_font('Arial', '', 11)

    probs = [
        ('No Impairment',        record.no_prob),
        ('Very Mild Impairment', record.verymild_prob),
        ('Mild Impairment',      record.mild_prob),
        ('Moderate Impairment',  record.moderate_prob),
    ]
    for stage, prob in probs:
        pdf.set_x(15)
        pdf.cell(80, 7, stage)
        pdf.cell(0, 7, f'{prob:.1f}%', ln=True)

    # Notes
    if record.notes:
        pdf.ln(3)
        pdf.set_x(15)
        pdf.set_font('Arial', 'B', 13)
        pdf.cell(0, 8, 'Doctor Notes', ln=True)
        pdf.set_x(15)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(180, 7, record.notes)

    # Disclaimer
    pdf.ln(5)
    pdf.set_fill_color(236, 240, 241)
    pdf.set_x(15)
    pdf.set_font('Arial', 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(
        180, 6,
        'DISCLAIMER: This report is generated by an AI system for '
        'screening purposes only. It is not a substitute for professional '
        'medical diagnosis. Please consult a qualified neurologist.',
        fill=True)

    # Footer
    pdf.set_y(-20)
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10,
             f"Generated on {datetime.now().strftime('%d/%m/%Y')} | "
             f"Alzheimer's Detection System",
             align='C')

    report_name = f'report_{record.id}_{user.id}.pdf'
    report_path = os.path.join(BASE_DIR, 'static', 'reports', report_name)
    pdf.output(report_path)

    return send_file(report_path,
                     as_attachment=True,
                     download_name=f'Alzheimer_Report_{record.id}.pdf')

# ─────────────────────────────────────────────────────────────
# ROUTES — DOCTOR DASHBOARD & ADMIN
# ─────────────────────────────────────────────────────────────


@app.route('/doctor')
@login_required
@role_required('doctor', 'admin')
def doctor_dashboard():
    patients = User.query.filter_by(role='patient').all()
    total_records = Record.query.count()
    stage_counts = {
        'No Impairment': Record.query.filter_by(
            prediction='No Impairment').count(),
        'Very Mild Impairment': Record.query.filter_by(
            prediction='Very Mild Impairment').count(),
        'Mild Impairment': Record.query.filter_by(
            prediction='Mild Impairment').count(),
        'Moderate Impairment': Record.query.filter_by(
            prediction='Moderate Impairment').count(),
    }
    recent = Record.query.order_by(
        Record.created_at.desc()).limit(10).all()
    return render_template('doctor.html',
                           patients=patients,
                           total=total_records,
                           stage_counts=stage_counts,
                           recent=recent,
                           stage_info=STAGE_INFO)


@app.route('/patient/<int:patient_id>')
@login_required
@role_required('doctor', 'admin')
def patient_records(patient_id):
    patient = User.query.get_or_404(patient_id)
    records = Record.query.filter_by(
        user_id=patient_id).order_by(
        Record.created_at.desc()).all()
    return render_template('patient_records.html',
                           patient=patient,
                           records=records,
                           stage_info=STAGE_INFO)


@app.route('/admin')
@login_required
@role_required('admin')
def admin_panel():
    users = User.query.all()
    records = Record.query.count()
    return render_template('admin.html',
                           users=users,
                           total_records=records)


@app.route('/admin/delete_user/<int:user_id>')
@login_required
@role_required('admin')
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.role == 'admin':
        flash('Cannot delete admin.', 'danger')
        return redirect(url_for('admin_panel'))
    Record.query.filter_by(user_id=user_id).delete()
    db.session.delete(user)
    db.session.commit()
    flash(f'User {user.name} deleted.', 'info')
    return redirect(url_for('admin_panel'))


@app.route('/admin/change_role/<int:user_id>/<role>')
@login_required
@role_required('admin')
def change_role(user_id, role):
    user = User.query.get_or_404(user_id)
    if role in ['patient', 'doctor', 'admin']:
        user.role = role
        db.session.commit()
        flash(f'{user.name} role changed to {role}.', 'success')
    return redirect(url_for('admin_panel'))


# ─────────────────────────────────────────────────────────────
# INITIALIZE & RUN
# ─────────────────────────────────────────────────────────────
with app.app_context():
    db.create_all()
    if not User.query.filter_by(email='admin@alzheimer.com').first():
        admin = User(
            name='Admin',
            email='admin@alzheimer.com',
            password=generate_password_hash('admin123'),
            role='admin')
        db.session.add(admin)
        db.session.commit()
        print("✅ Default admin created: admin@alzheimer.com / admin123")

# Load model weights on startup
load_model()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
