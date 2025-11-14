# ======================================================
# KrushiSetu (TFLite Version) – Login + Signup + ML Model
# Deploy-Ready for Streamlit Cloud
# ======================================================

import os
# ensure working dir is repo root (safe guard)
try:
    os.chdir(os.path.dirname(__file__))
except Exception:
    pass

import json
import streamlit as st
from PIL import Image
import numpy as np
import time
from datetime import datetime

# tflite runtime
try:
    import tflite_runtime.interpreter as tflite
except Exception:
    tflite = None

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="KrushiSetu", page_icon="🌿", layout="wide")

# ======================================================
# SESSION SETUP
# ======================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = {}

if "history" not in st.session_state:
    st.session_state.history = []

# ======================================================
# FILE PATHS
# ======================================================
USERS_FILE = "users.json"
CLASS_JSON = "class_indices.json"
DISEASE_JSON = "disease_info.json"
TFLITE_MODEL = "model.tflite"

# ======================================================
# JSON HELPERS
# ======================================================
def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

users_db = load_json(USERS_FILE)
class_map = load_json(CLASS_JSON)      # expected format: {"3": "Apple___healthy", ...}
disease_info = load_json(DISEASE_JSON) # expected format: {"0": {"name":..., "description":..., "solution":...}, ...}

# ======================================================
# AUTH FUNCTIONS
# ======================================================
def login_user(email, password):
    if email in users_db and users_db[email].get("password") == password:
        return True, users_db[email]
    return False, None

def register_user(name, email, password):
    if email in users_db:
        return False, "Email already registered."
    users_db[email] = {"name": name, "email": email, "password": password}
    save_json(USERS_FILE, users_db)
    return True, "Registration successful."

# ======================================================
# LOAD TFLITE MODEL
# ======================================================
interpreter = None
input_details = None
output_details = None

if tflite is not None and os.path.exists(TFLITE_MODEL):
    try:
        interpreter = tflite.Interpreter(model_path=TFLITE_MODEL)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        st.sidebar.success("✅ TFLite model loaded")
    except Exception as e:
        st.sidebar.error(f"Model load failed: {e}")
        interpreter = None
else:
    if tflite is None:
        st.sidebar.warning("tflite-runtime not installed in environment.")
    else:
        st.sidebar.warning("model.tflite not found — using dummy predictions.")

# ======================================================
# CSS — full UI theme (paste of your original CSS)
# ======================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

:root {
    --primary-bg: #1A1D1A;
    --secondary-bg: #2B2F2B;
    --text-color: #E6E6E6;
    --accent-color: #6A994E;
    --accent-color-hover: #8ABF69;
    --border-color: #3C413C;
    --font-family: 'Inter', sans-serif;
}

body {
    color: var(--text-color);
    background-color: var(--primary-bg);
    font-family: var(--font-family);
}

.main {
    background-color: var(--primary-bg);
    padding: 2rem;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--accent-color);
    font-weight: 600;
}

h1 {
    font-size: 2.5rem;
    border-bottom: 2px solid var(--border-color);
    padding-bottom: 0.5rem;
}

.stButton>button {
    background-color: var(--accent-color);
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background-color: var(--accent-color-hover);
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.stTextInput>div>div>input {
    background-color: var(--secondary-bg);
    color: var(--text-color);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 0.75rem;
}

.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--secondary-bg);
    padding: 15px 60px;
    border-radius: 16px;
    box-shadow: 0 4px 10px rgba(0,0,0,.35);
    margin-bottom: 30px;
    border: 1px solid var(--border-color);
}

.logo {
    font-size: 26px;
    font-weight: 700;
    color: var(--accent-color-hover);
}

.nav-links {
    display: flex;
    gap: 35px;
}

.nav-links a {
    text-decoration: none;
    color: var(--text-color);
    font-weight: 500;
    transition: .2s;
    padding: 8px 15px;
    border-radius: 8px;
}

.nav-links a:hover {
    color: var(--accent-color-hover);
    background-color: var(--border-color);
}

.hero {
    background: linear-gradient(135deg, var(--secondary-bg), var(--primary-bg));
    border: 1px solid var(--border-color);
    border-radius: 25px;
    padding: 60px 80px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 6px 22px rgba(0,0,0,.45);
}

.hero-text {
    max-width: 50%;
}

.hero-text h1 {
    font-size: 42px;
    font-weight: 800;
    color: var(--text-color);
    border: none;
}

.hero-text p {
    font-size: 18px;
    color: #bfe8c8;
    margin-top: 10px;
}

.hero-buttons {
    margin-top: 25px;
}

.hero-buttons button {
    background: var(--accent-color);
    color: #fff;
    border: none;
    padding: 12px 28px;
    border-radius: 8px;
    font-weight: 600;
    margin-right: 15px;
    cursor: pointer;
    transition: .3s;
}

.hero-buttons button:hover {
    background: var(--accent-color-hover);
    transform: translateY(-2px);
}

.hero-img {
    width: 38%;
    text-align: right;
}

.hero-img img {
    width: 90%;
    border-radius: 20px;
}

.prediction-section {
    background: var(--secondary-bg);
    border: 1px solid var(--border-color);
    border-radius: 25px;
    padding: 50px 70px;
    margin-top: 50px;
    text-align: center;
    box-shadow: 0 6px 22px rgba(0,0,0,.35);
}

.prediction-section h2 {
    color: var(--accent-color);
    font-size: 32px;
    margin-bottom: 25px;
    border: none;
}

.result-card {
    background: var(--primary-bg);
    border: 2px solid var(--accent-color);
    border-radius: 20px;
    padding: 24px;
    margin: 25px auto 0;
    width: 90%;
    text-align: left;
    box-shadow: 0 6px 18px rgba(0,0,0,.45);
}

.result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
}

.badge {
    display: inline-block;
    background: var(--accent-color);
    color: #fff;
    padding: 6px 12px;
    border-radius: 16px;
    font-size: 13px;
    font-weight: 700;
}

.conf {
    display: inline-block;
    background: var(--secondary-bg);
    color: var(--text-color);
    padding: 6px 12px;
    border-radius: 16px;
    font-size: 13px;
    border: 1px solid var(--border-color);
    font-weight: 700;
}

.result-title {
    margin: 8px 0 2px 0;
    font-size: 22px;
    font-weight: 800;
    color: var(--text-color);
}

.result-sub {
    margin: 0 0 14px 0;
    color: #9fd7ac;
    font-size: 13px;
}

.result-block {
    background: var(--secondary-bg);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 14px;
    margin-top: 14px;
}

.result-block h5 {
    margin: 0 0 8px 0;
    font-size: 15px;
    color: var(--accent-color-hover);
}

.result-block div {
    color: var(--text-color);
}

.features {
    display: flex;
    justify-content: space-around;
    margin-top: 50px;
    gap: 20px;
}

.feature-card {
    background: var(--secondary-bg);
    border: 1px solid var(--border-color);
    border-radius: 15px;
    box-shadow: 0 4px 14px rgba(0,0,0,.35);
    padding: 25px;
    width: 28%;
    text-align: center;
    transition: transform .3s, box-shadow .3s;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(106, 153, 78, 0.2);
}

.feature-card img {
    width: 60px;
    margin-bottom: 15px;
}

.feature-card h4 {
    color: var(--accent-color);
    margin-bottom: 10px;
    font-weight: 700;
}

.feature-card p {
    color: var(--text-color);
    font-size: 15px;
}

.footer {
    text-align: center;
    color: var(--accent-color);
    margin-top: 40px;
    font-size: 14px;
}

header { visibility: hidden; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# PREDICTION (TFLITE) FUNCTION
# ======================================================
INPUT_SIZE = (128, 128)

def predict_tflite(image):
    # if interpreter absent => dummy
    if interpreter is None:
        dummy = ["Healthy Leaf", "Powdery Mildew", "Leaf Spot", "Rust Disease"]
        p = np.random.choice(dummy)
        c = np.random.uniform(0.70, 0.95)
        return p, c, "Dummy model — remedy not available"

    try:
        img = image.convert("RGB").resize(INPUT_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])

        # handle output shapes
        if preds.ndim == 2:
            pred_vec = preds[0]
        else:
            pred_vec = preds

        idx = int(np.argmax(pred_vec))
        conf = float(np.max(pred_vec))

        disease_name = class_map.get(str(idx), f"Class_{idx}")
        remedy = disease_info.get(str(idx), {}).get("solution", "No remedy available")

        return disease_name, conf, remedy
    except Exception as e:
        return "Error", 0.0, f"Prediction failed: {e}"

# ======================================================
# ROUTING HELPERS
# ======================================================
def go(page):
    st.query_params["page"] = page

page = st.query_params.get("page", "login")

# ======================================================
# LOGIN PAGE (if not logged in)
# ======================================================
if page == "login" and not st.session_state.logged_in:
    st.markdown("""
    <div style="max-width:700px;margin:auto;">
      <h1 style="color:var(--accent-color);">🌿 KrushiSetu</h1>
      <p style="color:#bfe8c8;">Login to access crop disease detection</p>
    </div>
    """, unsafe_allow_html=True)

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Login"):
            ok, user = login_user(email, password)
            if ok:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.success("Login successful!")
                time.sleep(0.8)
                go("home")
            else:
                st.error("Invalid email or password")

    with col2:
        if st.button("New user? Register"):
            go("signup")

    st.stop()

# ======================================================
# SIGNUP PAGE
# ======================================================
if page == "signup" and not st.session_state.logged_in:
    st.markdown("<h2 style='color:var(--accent-color)'>Create Account</h2>", unsafe_allow_html=True)
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Sign Up"):
        ok, msg = register_user(name, email, password)
        if ok:
            st.success(msg)
            time.sleep(1)
            go("login")
        else:
            st.error(msg)

    if st.button("Back to Login"):
        go("login")

    st.stop()

# ======================================================
# LOGOUT
# ======================================================
if page == "logout":
    st.session_state.logged_in = False
    st.session_state.user = {}
    go("login")
    st.stop()

# ======================================================
# Protect inner pages
# ======================================================
if not st.session_state.logged_in:
    go("login")
    st.stop()

# ======================================================
# NAVBAR (after login)
# ======================================================
username = st.session_state.user.get("name", "User")
avatar_letter = username[:1].upper() if username else "U"

st.markdown(f"""
    <div class="navbar">
      <div class="logo">🌿 KrushiSetu</div>
      <div class="nav-links">
        <a href="?page=home">Home</a>
        <a href="?page=detect">Crop Detection</a>
        <a href="?page=about">About</a>
        <a href="?page=profile" style="background:var(--accent-color);color:white;">{avatar_letter}</a>
        <a href="?page=logout" style="color:#ff7b7b;">Logout</a>
      </div>
    </div>
""", unsafe_allow_html=True)

# ======================================================
# PAGE: HOME
# ======================================================
if page == "home":
    st.markdown("""
    <div class="hero">
      <div class="hero-text">
        <h1>Empowering Farmers with AI-Powered Crop Health Insights</h1>
        <p>Detect diseases, improve yield, and make smarter farming decisions with KrushiSetu.</p>
        <div class="hero-buttons">
          <button onclick="window.location.href='?page=detect'">Upload Crop Image</button>
          <button onclick="window.location.href='?page=detect'">Predict Disease</button>
        </div>
      </div>
      <div class="hero-img">
        <img src="https://cdn-icons-png.flaticon.com/512/3663/3663197.png" alt="Farmer Illustration">
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="features">
      <div class="feature-card">
        <img src="https://cdn-icons-png.flaticon.com/512/2907/2907432.png">
        <h4>Crop Disease Detection</h4>
        <p>Get instant AI predictions for your crops.</p>
      </div>
      <div class="feature-card">
        <img src="https://cdn-icons-png.flaticon.com/512/869/869869.png">
        <h4>Weather Updates</h4>
        <p>Plan your farming schedule with accurate forecasts.</p>
      </div>
      <div class="feature-card">
        <img src="https://cdn-icons-png.flaticon.com/512/616/616408.png">
        <h4>Fertilizer Suggestions</h4>
        <p>Improve soil health and increase yield with eco-friendly inputs.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# PAGE: DETECT
# ======================================================
elif page == "detect":
    st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
    st.markdown('<h2>AI Crop Disease Prediction</h2>', unsafe_allow_html=True)

    uploaded = st.file_uploader("📂 Upload a Crop Image", type=["jpg","jpeg","png"])
    camera = st.camera_input("📸 Or Take a Picture")

    if uploaded or camera:
        img = Image.open(uploaded if uploaded else camera)
        st.image(img, caption="Captured Crop Image", use_column_width=True)

        with st.spinner("Analyzing crop health..."):
            time.sleep(0.5)
            prediction, confidence, remedy = predict_tflite(img)

        st.markdown(f"""
        <div class="result-card">
          <div class="result-header">
            <span class="badge">Detected</span>
            <span class="conf">Confidence: {confidence*100:.2f}%</span>
          </div>
          <div class="result-title">{prediction}</div>
          <div class="result-block">
            <h5>💊 Suggested Remedy</h5>
            <div>{remedy}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Save history in session
        st.session_state.history.append({
            "disease": prediction,
            "confidence": f"{confidence*100:.2f}%",
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "remedy": remedy
        })
    else:
        st.info("📸 Capture or upload a crop image to start prediction.")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# PAGE: ABOUT
# ======================================================
elif page == "about":
    st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
    st.markdown('<h2>🌍 About KrushiSetu</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:18px;text-align:center;'>
      <b>KrushiSetu</b> is an AI-powered digital agriculture assistant designed to help farmers diagnose crop diseases instantly and receive reliable organic remedies.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# PAGE: PROFILE
# ======================================================
elif page == "profile":
    st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
    st.markdown('<h2>👤 User Profile</h2>', unsafe_allow_html=True)

    profile = st.session_state.user
    with st.form("profile_form"):
        name = st.text_input("👤 Name", profile.get("name",""))
        email = st.text_input("📧 Email", profile.get("email",""))
        mobile = st.text_input("📱 Mobile", profile.get("mobile",""))
        farm = st.text_input("🌾 Farm/Organization", profile.get("farm_name",""))
        if st.form_submit_button("💾 Save Profile"):
            # update local session and also update users_db (if email exists)
            st.session_state.user.update({"name": name, "email": email, "mobile": mobile, "farm_name": farm})
            if profile.get("email") in users_db:
                users_db[profile.get("email")].update({"name": name, "email": email})
                save_json(USERS_FILE, users_db)
            st.success("✅ Profile updated successfully!")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""<div class="footer">©2025 KrushiSetu | Empowering Farmers with AI 🌾</div>""", unsafe_allow_html=True)
