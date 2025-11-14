# ============================================
# KrushiSetu + Login System + ML Model
# ============================================

import os
os.chdir(os.path.dirname(__file__))

import json
import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import time
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="KrushiSetu", page_icon="🌿", layout="wide")

# =====================================================
# SESSION SETUP
# =====================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = {}

if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# LOAD JSON HELPERS
# =====================================================
def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

USERS_FILE = "users.json"
users_db = load_json(USERS_FILE)

# =====================================================
# AUTH FUNCTIONS
# =====================================================

def login_user(email, password):
    if email in users_db:
        if password == users_db[email]["password"]:
            return True, users_db[email]
    return False, None

def register_user(name, email, password):
    if email in users_db:
        return False, "Email already registered"
    users_db[email] = {"name": name, "email": email, "password": password}
    save_json(USERS_FILE, users_db)
    return True, "Registered successfully"

# =====================================================
# LOAD MODEL
# =====================================================
MODEL_PATH = "model.keras"
CLASS_JSON = "class_indices.json"
DISEASE_JSON = "disease_info.json"

model = None
class_map = load_json(CLASS_JSON)  # e.g. {"3": "Apple___healthy"}
disease_info = load_json(DISEASE_JSON)  # e.g. {"0": {name, desc, solution}}

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        st.sidebar.success("Model Loaded")
    else:
        st.sidebar.warning("model.keras missing — dummy predictions used")
except:
    st.sidebar.error("Model load error — dummy prediction used")
    model = None

# =====================================================
# CSS (UNCHANGED — your original)
# =====================================================
st.markdown("""<style>  

/* Full CSS Removed for Shortness — PASTE YOUR FULL CSS HERE */

</style>""", unsafe_allow_html=True)

# =====================================================
# ML PREDICTION FUNCTION
# =====================================================
INPUT_SIZE = (128, 128)

def predict_disease(image):

    if model is None:    # Dummy fallback
        dummy = ["Healthy Leaf","Powdery Mildew","Leaf Spot","Rust Disease"]
        p = np.random.choice(dummy)
        c = np.random.uniform(0.80,0.97)
        return p, c, "Dummy model — no remedy available"

    try:
        img = image.convert("RGB").resize(INPUT_SIZE)
        arr = np.array(img)/255.0
        arr = np.expand_dims(arr,0)

        preds = model.predict(arr)
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))

        disease_name = class_map.get(str(idx), f"Class_{idx}")

        remedy = disease_info.get(str(idx), {}).get("solution","No solution available")

        return disease_name, conf, remedy

    except Exception as e:
        return "Error", 0.0, f"Prediction failed: {e}"

# =====================================================
# PAGE ROUTING LOGIC
# =====================================================

def go(page):
    st.query_params["page"] = page

page = st.query_params.get("page","login")

# =====================================================
# LOGIN PAGE
# =====================================================
if page == "login" and not st.session_state.logged_in:

    st.title("🌿 KrushiSetu Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        ok, usr = login_user(email, password)
        if ok:
            st.session_state.logged_in = True
            st.session_state.user = usr
            go("home")
        else:
            st.error("Invalid Credentials")

    if st.button("New User? Register"):
        go("signup")

    st.stop()   # Prevent seeing other pages

# =====================================================
# SIGNUP PAGE
# =====================================================
if page == "signup" and not st.session_state.logged_in:

    st.title("🌿 Create New Account")

    name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Create Account"):
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

# =====================================================
# LOGOUT
# =====================================================
if page == "logout":
    st.session_state.logged_in = False
    st.session_state.user = {}
    go("login")
    st.stop()

# =====================================================
# AUTH PROTECTION FOR INNER PAGES
# =====================================================
if not st.session_state.logged_in:
    go("login")
    st.stop()

# =====================================================
# NAVBAR (after login)
# =====================================================
st.markdown(f"""
<div class="navbar">
  <div class="logo">🌿 KrushiSetu</div>
  <div class="nav-links">
    <a href="?page=home">Home</a>
    <a href="?page=detect">Detect</a>
    <a href="?page=about">About</a>
    <a href="?page=profile">{st.session_state.user.get('name','U')[0:].upper()}</a>
    <a href="?page=logout" style="color:red;">Logout</a>
  </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# HOME PAGE
# =====================================================
if page == "home":
    st.markdown("""<div class="hero"> 
        <h1>Welcome, """+st.session_state.user["name"]+""" 👋</h1>
        <p>AI-powered Crop Health Assistant</p>
    </div>""", unsafe_allow_html=True)

# =====================================================
# DETECT PAGE
# =====================================================
elif page == "detect":
    st.header("📸 AI Crop Disease Detection")

    uploaded = st.file_uploader("Upload Crop Image", type=["jpg","jpeg","png"])
    camera = st.camera_input("Or Take Picture")

    if uploaded or camera:
        img = Image.open(uploaded if uploaded else camera)
        st.image(img, caption="Uploaded", use_column_width=True)

        with st.spinner("Analyzing..."):
            pred, conf, remedy = predict_disease(img)

        st.success(f"Prediction: {pred}")
        st.info(f"Confidence: {conf*100:.2f}%")
        st.warning(f"Suggested Remedy: {remedy}")

# =====================================================
# ABOUT PAGE
# =====================================================
elif page == "about":
    st.header("🌿 About KrushiSetu")
    st.write("AI-based plant disease detection system for farmers.")

# =====================================================
# PROFILE PAGE
# =====================================================
elif page == "profile":
    st.header("👤 User Profile")

    st.write(f"Name: {st.session_state.user['name']}")
    st.write(f"Email: {st.session_state.user['email']}")

    st.write("Password hidden for safety")

# =====================================================
# FOOTER
# =====================================================
st.markdown("<div class='footer'>© 2025 KrushiSetu</div>", unsafe_allow_html=True)
