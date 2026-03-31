import os
import time
import sqlite3
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    redirect,
    url_for,
    flash,
)
import time
from collections import defaultdict
import numpy as np
import joblib
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from openai import OpenAI
import gdown

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

def download_file(url, filename):
    """Download model files if they don't exist"""
    if not os.path.exists(filename):
        print(f"📥 Downloading {filename}...")
        try:
            gdown.download(url, filename, quiet=False)
            print(f"✅ Downloaded {filename}")
            return True
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            return False
    else:
        print(f"✅ {filename} already exists")
        return True

# Load environment variables from .env file
load_dotenv()

# =========================
# 🤖 HUGGING FACE API SETUP
# =========================
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")
)

# =========================
# 📥 AUTO MODEL DOWNLOAD
# =========================
print("🔄 Checking model files...")

# Download model files if they don't exist
model_files = [
    ("https://drive.google.com/uc?id=1keHlBAsOuV0LtJJFv5Lp1IRyc1ARL2C4", "model/improved_energy_model.pkl"),
    ("https://drive.google.com/uc?id=1p9kx-Jo8f2hXgupaes4qnoWKmSJh2g-7", "model/scaler.pkl"),
    ("https://drive.google.com/uc?id=1sa2pzfudlisNiF-_8DlhXKwigtpq8whq", "model/feature_selector.pkl"),
    ("https://drive.google.com/uc?id=1lZeFy4ChRMEl5G2tqP3aQJPQVWachvTU", "model/selected_features.pkl")
]

success = True
for url, filename in model_files:
    if not download_file(url, filename):
        success = False

if success:
    print("✅ All model files ready!")
else:
    print("⚠️ Warning: Some model files failed to download")

# =========================
# 🤖 LOAD IMPROVED ML MODEL
# =========================
try:
    ml_model = joblib.load(os.path.join(os.path.dirname(__file__), "model/improved_energy_model.pkl"))
    scaler = joblib.load(os.path.join(os.path.dirname(__file__), "model/scaler.pkl"))
    selector = joblib.load(os.path.join(os.path.dirname(__file__), "model/feature_selector.pkl"))
    selected_features = joblib.load(os.path.join(os.path.dirname(__file__), "model/selected_features.pkl"))
    print("✅ ML models loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not load ML models: {e}")
    # Fallback to basic calculation if models fail
    ml_model = None
    scaler = None
    selector = None
    selected_features = None

def query_huggingface(prompt):
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:novita",
            messages=[
                {
                    "role": "system",
                    "content": "You are EcoBot, an AI assistant that helps users with energy saving tips and general questions.Give short answers in 2-3 lines only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=60,
            temperature=0.7,
            timeout=30
        )

        reply = completion.choices[0].message.content
        return {"generated_text": reply}

    except Exception as e:
        print("HF ERROR:", str(e))
        return {"error": str(e)}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "ecopredict-dev-secret-change-me")
CORS(app)

# =========================
# 🔐 AUTH DATABASE (SQLite)
# =========================
def _auth_db_path():
    instance_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance")
    os.makedirs(instance_dir, exist_ok=True)
    return os.path.join(instance_dir, "users.db")


def init_auth_db():
    conn = sqlite3.connect(_auth_db_path())
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def get_user_by_email(email):
    conn = sqlite3.connect(_auth_db_path())
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT id, name, email, password_hash FROM users WHERE email = ?",
            (email.lower().strip(),),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_user(name, email, password_hash):
    conn = sqlite3.connect(_auth_db_path())
    try:
        conn.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name.strip(), email.lower().strip(), password_hash),
        )
        conn.commit()
    finally:
        conn.close()


init_auth_db()

def chatbot_response(user_message):
    msg = user_message.lower()

    if "predict" in msg:
        return "Please provide details like AC usage, fans, lights etc. to predict energy and CO2."

    try:
        response = query_huggingface(user_message)

        if "error" in response:
            return "⚡ EcoBot is busy. Try again."

        return response["generated_text"]

    except Exception as e:
        print("Chatbot Error:", str(e))
        return "⚡ EcoBot is busy."


# =========================
# 🔐 AUTH: session + guards
# =========================
@app.context_processor
def inject_auth():
    return {
        "auth_user_email": session.get("user"),
        "auth_user_name": session.get("user_name"),
        "auth_logged_in": bool(session.get("user")),
    }


@app.before_request
def require_login():
    if request.endpoint is None:
        return None
    if request.endpoint == "static":
        return None
    if request.path.startswith("/static"):
        return None
    if request.endpoint in ("login", "signup", "logout"):
        return None
    # Keep prediction JSON API working without session (unchanged contract for clients)
    if request.path == "/predict" and request.method == "POST":
        return None
    if not session.get("user"):
        return redirect(url_for("login"))


# =========================
# 🌐 ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict-page")
def predict_page():
    return render_template("predict.html")

@app.route("/result")
def result_page():
    energy = request.args.get("energy")
    co2 = request.args.get("co2")
    return render_template("result.html", energy=energy, carbon=co2)

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user"):
        return redirect(url_for("home"))
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        user = get_user_by_email(email) if email else None
        if user and check_password_hash(user["password_hash"], password):
            session["user"] = user["email"]
            session["user_name"] = user["name"]
            flash("Logged in successfully.", "success")
            return redirect(url_for("home"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("user"):
        return redirect(url_for("home"))
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""
        if not name or not email or not password:
            flash("Please fill in all fields.", "error")
            return render_template("signup.html")
        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("signup.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("signup.html")
        if get_user_by_email(email):
            flash("An account with this email already exists.", "error")
            return render_template("signup.html")
        pw_hash = generate_password_hash(password)
        try:
            create_user(name, email, pw_hash)
        except sqlite3.IntegrityError:
            flash("An account with this email already exists.", "error")
            return render_template("signup.html")
        session["user"] = email.lower().strip()
        session["user_name"] = name.strip()
        flash("Account created. Welcome!", "success")
        return redirect(url_for("home"))
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

@app.route("/profile")
def profile():
    return render_template(
        "profile.html",
        user_name=session.get("user_name") or "",
        user_email=session.get("user") or "",
    )

@app.route("/ecobot")
def ecobot_page():
    return render_template("ecobot.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# Rate limiting to prevent API abuse (TEMPORARILY DISABLED FOR TESTING)
# request_counts = defaultdict(list)  
# RATE_LIMIT = 30  # requests per minute
# RATE_WINDOW = 60  # seconds

# =========================
# CHATBOT API ROUTE
# =========================
@app.route("/chat", methods=["POST"])
def chat():
    # Temporarily disabled rate limiting for testing
    user_msg = request.json.get("message")
    reply = chatbot_response(user_msg)
    return jsonify({"reply": reply})

# =========================
# ENERGY + CO2 PREDICTION API
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    AC_POWER = 1.5
    FAN_POWER = 0.075
    TV_POWER = 0.1
    LED_POWER = 0.008
    NORMAL_POWER = 0.06
    FRIDGE_POWER = 0.15
    WASH_POWER = 0.5
    ELECTRICITY_RATE = 8
    DAYS = 30

    from datetime import datetime
    month = datetime.now().month

    if month in [4,5,6]:
        season = "Summer"
        factor = 1.3
    elif month in [7,8,9]:
        season = "Rainy"
        factor = 1.1
    elif month in [10,11]:
        season = "Autumn"
        factor = 1.0
    elif month in [12,1]:
        season = "Winter"
        factor = 0.8
    else:
        season = "Spring"
        factor = 0.9

    ac_energy = data["ac_units"] * data["ac_hours"] * AC_POWER * DAYS
    fan_energy = data["fans"] * data["fan_hours"] * FAN_POWER * DAYS
    tv_energy = data["tvs"] * data["tv_hours"] * TV_POWER * DAYS

    light_energy = (
        data["led_bulbs"] * data["lighting_hours"] * LED_POWER * DAYS +
        data["normal_bulbs"] * data["lighting_hours"] * NORMAL_POWER * DAYS
    )

    fridge_energy = data["refrigerators"] * data["fridge_hours"] * FRIDGE_POWER * DAYS
    washing_energy = data["washing_units"] * data["washing_per_week"] * WASH_POWER * 4

    total_energy = (
        ac_energy + fan_energy + tv_energy +
        light_energy + fridge_energy + washing_energy
    )

    # =========================
    # 🚀 IMPROVED ML PREDICTION
    # =========================
    from datetime import datetime
    now = datetime.now()
    hour = now.hour
    day_of_week = now.weekday()
    month = now.month
    
    # Advanced feature engineering (same as training)
    lights = (data["led_bulbs"] + data["normal_bulbs"]) * data["lighting_hours"]
    
    # Environmental estimates based on user input
    T1 = 22 + (data["ac_hours"] * 0.2)
    RH_1 = 40 + (data["family_members"] * 2)
    T2 = T1 - 1
    RH_2 = RH_1 - 2
    T_out = 30
    Press_mm_hg = 760
    Windspeed = 3
    
    # Advanced features
    temp_range = T1 - T2
    temp_out_diff = T1 - T_out
    temp_avg_all = (T1 + T2 + 20 + 21 + 19 + 18 + 17 + 20 + 19) / 9  # Simulated room temps
    humidity_range = RH_1 - RH_2
    humidity_avg_all = (RH_1 + RH_2 + 45 + 50 + 48 + 52 + 46 + 49 + 47) / 9  # Simulated humidity
    
    # Interaction features
    temp_humidity_interaction = temp_avg_all * humidity_avg_all
    lights_temp_interaction = lights * T_out
    pressure_temp_interaction = Press_mm_hg * temp_avg_all
    wind_temp_interaction = Windspeed * temp_avg_all
    
    # Time-based features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Usage patterns
    is_peak_hour = 1 if (18 <= hour <= 22) else 0
    is_night = 1 if (0 <= hour <= 6) else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Comfort index
    comfort_index = (temp_avg_all - 20) / 10 + (humidity_avg_all - 50) / 20
    
    # Efficiency score
    lights_per_temp = lights / (temp_avg_all + 1)
    
    # Create feature array with all 26 features
    all_features = np.array([[
        lights, T1, RH_1, T2, RH_2, T_out, Press_mm_hg, Windspeed,
        temp_range, temp_out_diff, temp_avg_all, humidity_range, humidity_avg_all,
        temp_humidity_interaction, lights_temp_interaction, pressure_temp_interaction,
        wind_temp_interaction, hour_sin, hour_cos, month_sin, month_cos,
        is_peak_hour, is_night, is_weekend, comfort_index, lights_per_temp
    ]])
    
    # Select top 20 features (same as training)
    if ml_model and scaler and selector:
        selected_features_array = selector.transform(all_features)
        scaled_features = scaler.transform(selected_features_array)
        ml_prediction = float(ml_model.predict(scaled_features)[0])
    else:
        ml_prediction = total_energy  # fallback
    
    # Keep existing calculation for consistency
    final_energy = (total_energy * 0.7) + (ml_prediction * 0.3)

    co2 = final_energy * 0.82
    cost = final_energy * ELECTRICITY_RATE
    next_month = final_energy * factor

    warning = None
    if co2 > 500:
        warning = "⚠️ High carbon emission! Reduce AC usage."

    return jsonify({
        "energy": round(final_energy, 2),
        "co2": round(co2, 2),
        "cost": round(cost, 2),
        "next_month": round(next_month, 2),
        "season": season,
        "warning": warning,
        "breakdown": {
            "ac": round(ac_energy, 2),
            "fan": round(fan_energy, 2),
            "light": round(light_energy, 2),
            "fridge": round(fridge_energy, 2),
            "washing": round(washing_energy, 2)
        }
    })


# =========================
# 🚀 RUN APP
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)