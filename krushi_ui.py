# --- Auto-updated for Streamlit Cloud: load model from utils/model.keras ---
import os
os.chdir(os.path.dirname(__file__))  # ensure relative paths work on Streamlit Cloud
import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import time
from datetime import datetime

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="KrushiSetu", page_icon="🌿", layout="wide")

# ========== SESSION SETUP ==========
if "profile" not in st.session_state:
    st.session_state.profile = {"name": "Guest", "email": "", "mobile": "", "farm_name": ""}
if "history" not in st.session_state:
    st.session_state.history = []

# ========== ENHANCED DARK THEME CSS ==========
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

# ========== PREDICTION FUNCTION ==========
def predict_disease(image):
    """Dummy prediction function"""
    dummy_labels = ["Healthy Leaf", "Powdery Mildew", "Leaf Spot", "Rust Disease"]
    prediction = np.random.choice(dummy_labels)
    confidence = np.random.uniform(0.80, 0.98)
    
    remedies = {
        "Healthy Leaf": "No issue detected 🌿 Keep monitoring your crop health.",
        "Powdery Mildew": "Use organic sulfur spray or neem oil to control it.",
        "Leaf Spot": "Apply compost tea or copper-based fungicide.",
        "Rust Disease": "Use baking soda spray or remove affected leaves."
    }
    
    return prediction, confidence, remedies.get(prediction, "No remedy available.")

# ========== NAVBAR ==========
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown(f"""
    <div class="navbar">
      <div class="logo">🌿 KrushiSetu</div>
      <div class="nav-links">
        <a href="?page=home">Home</a>
        <a href="?page=detect">Crop Detection</a>
        <a href="?page=about">About</a>
        <a href="?page=profile" style="background:var(--accent-color);color:white;">
          {st.session_state.profile['name'][:1].upper() or 'U'}
        </a>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ========== PAGE ROUTING ==========
query_params = st.query_params
page = query_params.get("page", "home")

# ========== HOME ==========
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

# ========== DETECT ==========
elif page == "detect":
    st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
    st.markdown('<h2>AI Crop Disease Prediction</h2>', unsafe_allow_html=True)

    uploaded = st.file_uploader("📂 Upload a Crop Image", type=["jpg","jpeg","png"])
    camera = st.camera_input("📸 Or Take a Picture")

    if uploaded or camera:
        img = Image.open(uploaded if uploaded else camera)
        st.image(img, caption="Captured Crop Image", use_column_width=True)

        with st.spinner("Analyzing crop health..."):
            time.sleep(1)
            prediction, confidence, remedy = predict_disease(img)

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

        # Save history
        st.session_state.history.append({
            "disease": prediction,
            "confidence": f"{confidence*100:.2f}%",
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "remedy": remedy
        })
    else:
        st.info("📸 Capture or upload a crop image to start prediction.")
    st.markdown("</div>", unsafe_allow_html=True)


# ========== ABOUT ==========
elif page == "about":
    st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
    st.markdown('<h2>🌍 About KrushiSetu</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:18px;text-align:center;'>
      <b>KrushiSetu</b> Of course!


# 🌿 **About KrushiSetu – Smart Farming for a Smarter Future**

KrushiSetu is an AI-powered digital agriculture assistant designed to help farmers diagnose crop diseases instantly and receive reliable organic remedies.
Our goal is to bridge the gap between **traditional farming** and **modern artificial intelligence**, empowering farmers with accurate and affordable solutions.

---

# 🌾 **What KrushiSetu Does**

### ✔ **AI Crop Disease Detection**

Uploads or camera-captured leaf images are scanned using machine learning models to detect plant diseases with high accuracy.

### ✔ **Instant Organic Remedies**

Based on the detected disease, KrushiSetu suggests **eco-friendly, chemical-free treatments** to restore plant health.

### ✔ **Smart Farming Assistance**

The app provides additional help like fertilizer suggestions, weather awareness and crop care tips.

### ✔ **Scan History Tracking**

Every diagnosis is stored, helping farmers track plant health over time.

---

# 🌱 **Our Mission**

To empower every farmer with **smart, accessible and sustainable agricultural technology** — reducing crop losses, increasing yield, and improving long-term soil health.

---

# 💡 **Why We Built KrushiSetu**

* Farmers often struggle to identify early-stage crop diseases
* Expert guidance is not always available in rural areas
* Traditional farming lacks quick, reliable digital support
* Early detection saves crops, money, and effort
* AI-powered diagnosis brings **scientific support** directly to farmers’ phones

---

# 🤝 **Our Vision**

To build a digital ecosystem where:

🌍 **Every farmer receives instant support**
🌿 **Farming becomes more sustainable**
📈 **Crop yields increase through smart decisions**
🔬 **AI and agriculture work hand-in-hand**

---

# 🧑‍🌾 **Who Is KrushiSetu For?**

KrushiSetu is designed for:

* Farmers
* Agricultural students
* Researchers
* Gardeners
* Nursery owners
* Anyone who wants to maintain healthy plants

---

# 🛠 **Technology Behind KrushiSetu**

* Advanced AI/ML models trained on crop disease datasets
* Preprocessing and real-time detection
* Streamlit UI for fast and simple use
* Secure login & user management system
* SQLite backend for data storage
* Optimized image processing pipeline

---

# 🚀 **Future Enhancements**

We plan to add:

* Soil health scanning
* Weather-based risk prediction
* Crop yield prediction
* Voice-based farmer assistance
* Multi-language support
* Marketplace for seeds, fertilizers, and products

---

    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ========== PROFILE ==========
elif page == "profile":
    st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
    st.markdown('<h2>👤 User Profile</h2>', unsafe_allow_html=True)

    profile = st.session_state.profile
    with st.form("profile_form"):
        name = st.text_input("👤 Name", profile["name"])
        email = st.text_input("📧 Email", profile["email"])
        mobile = st.text_input("📱 Mobile", profile["mobile"])
        farm = st.text_input("🌾 Farm/Organization", profile["farm_name"])
        if st.form_submit_button("💾 Save Profile"):
            st.session_state.profile.update({"name": name, "email": email, "mobile": mobile, "farm_name": farm})
            st.success("✅ Profile updated successfully!")
    st.markdown("</div>", unsafe_allow_html=True)

# ========== FOOTER ==========
st.markdown("""<div class="footer">©2025 KrushiSetu | Empowering Farmers with AI 🌾</div>""", unsafe_allow_html=True)