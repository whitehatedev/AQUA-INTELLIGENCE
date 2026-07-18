import os
import time
import threading
import joblib
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# ================================================================
# 1. LOAD ML MODELS (Fish + Spice)
# ================================================================
def load_models():
    try:
        model = joblib.load("fish_species_model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        spice_model = joblib.load("spice_model.pkl")
        spice_scaler = joblib.load("spice_scaler.pkl")
        spice_encoder = joblib.load("spice_encoder.pkl")
        return model, scaler, label_encoder, spice_model, spice_scaler, spice_encoder
    except Exception as e:
        print("❌ Error loading models:", e)
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        scaler = StandardScaler()
        label_encoder = LabelEncoder()
        spice_model = RandomForestClassifier()
        spice_scaler = StandardScaler()
        spice_encoder = LabelEncoder()
        return model, scaler, label_encoder, spice_model, spice_scaler, spice_encoder

model, scaler, label_encoder, spice_model, spice_scaler, spice_encoder = load_models()

# ================================================================
# 2. LOAD / COMPUTE PER-SPECIES STATS FROM CSV
# ================================================================
STATS_FILE = "fish_species_stats.pkl"
CSV_FILE = "realfishdataset (1).csv"

def compute_species_stats():
    if not os.path.exists(CSV_FILE):
        print("⚠️ CSV file not found – using empty stats.")
        return {}
    try:
        df = pd.read_csv(CSV_FILE)
        df.columns = df.columns.str.strip()
        stats = {}
        for species in df['fish'].unique():
            sub = df[df['fish'] == species]
            stats[species] = {
                'ph_mean': sub['ph'].mean(),
                'ph_std': sub['ph'].std(),
                'ph_min': sub['ph'].min(),
                'ph_max': sub['ph'].max(),
                'temperature_mean': sub['temperature'].mean(),
                'temperature_std': sub['temperature'].std(),
                'temperature_min': sub['temperature'].min(),
                'temperature_max': sub['temperature'].max(),
                'turbidity_mean': sub['turbidity'].mean(),
                'turbidity_std': sub['turbidity'].std(),
                'turbidity_min': sub['turbidity'].min(),
                'turbidity_max': sub['turbidity'].max(),
                'count': len(sub)
            }
        joblib.dump(stats, STATS_FILE)
        print(f"✅ Species stats computed and saved to {STATS_FILE}")
        return stats
    except Exception as e:
        print("❌ Error computing stats:", e)
        return {}

# Load stats from file if exists, else compute
if os.path.exists(STATS_FILE):
    SPECIES_STATS = joblib.load(STATS_FILE)
    print(f"✅ Loaded species stats for {len(SPECIES_STATS)} species")
else:
    SPECIES_STATS = compute_species_stats()

# Print species summary
if SPECIES_STATS:
    print("="*50)
    print(f"📊 Total number of fish species: {len(SPECIES_STATS)}")
    print("="*50)
    print("\nSpecies and their record counts:")
    for species, stats in sorted(SPECIES_STATS.items(), key=lambda x: x[1]['count'], reverse=True):
        print(f"  {species}: {stats['count']} records")
    print("="*50)

# ================================================================
# 3. BUILD FISH SPECIES RULES (with image URLs)
# ================================================================
# Hard-coded rules with metadata and image URLs
HARD_CODED_RULES = {
    "Tilapia": {
        "ph": (6.5, 8.5), "temperature": (24, 32), "turbidity": (0, 10),
        "icon": "🐟", "growth_rate": "Fast", "market_value": "Medium", "hardiness": "High",
        "image_url": "https://ui-avatars.com/api/?name=Tilapia&background=4f46e5&color=fff&size=40"
    },
    "Pangas": {
        "ph": (6.5, 8.2), "temperature": (24, 30), "turbidity": (0, 12),
        "icon": "🐠", "growth_rate": "Very Fast", "market_value": "High", "hardiness": "Medium",
        "image_url": "https://ui-avatars.com/api/?name=Pangas&background=059669&color=fff&size=40"
    },
    "Rui": {
        "ph": (6.8, 8.0), "temperature": (22, 30), "turbidity": (0, 10),
        "icon": "🎏", "growth_rate": "Fast", "market_value": "High", "hardiness": "High",
        "image_url": "https://ui-avatars.com/api/?name=Rui&background=2563eb&color=fff&size=40"
    },
    "Katla": {
        "ph": (7.0, 8.5), "temperature": (22, 30), "turbidity": (0, 10),
        "icon": "🐋", "growth_rate": "Fast", "market_value": "High", "hardiness": "Medium",
        "image_url": "https://ui-avatars.com/api/?name=Katla&background=7c3aed&color=fff&size=40"
    },
    "Koi": {
        "ph": (6.5, 8.0), "temperature": (24, 30), "turbidity": (0, 8),
        "icon": "🪸", "growth_rate": "Medium", "market_value": "Very High", "hardiness": "Medium",
        "image_url": "https://ui-avatars.com/api/?name=Koi&background=dc2626&color=fff&size=40"
    },
    "Magur": {
        "ph": (6.5, 8.0), "temperature": (24, 32), "turbidity": (0, 12),
        "icon": "🐍", "growth_rate": "Slow", "market_value": "Very High", "hardiness": "Very High",
        "image_url": "https://ui-avatars.com/api/?name=Magur&background=ca8a04&color=fff&size=40"
    },
    "Sing": {
        "ph": (6.5, 8.0), "temperature": (24, 32), "turbidity": (0, 12),
        "icon": "🐙", "growth_rate": "Medium", "market_value": "High", "hardiness": "High",
        "image_url": "https://ui-avatars.com/api/?name=Sing&background=0891b2&color=fff&size=40"
    },
    "Shrimp": {
        "ph": (7.0, 8.5), "temperature": (24, 30), "turbidity": (0, 8),
        "icon": "🦐", "growth_rate": "Fast", "market_value": "Very High", "hardiness": "Low",
        "image_url": "https://ui-avatars.com/api/?name=Shrimp&background=db2777&color=fff&size=40"
    },
    "Prawn": {
        "ph": (7.0, 8.5), "temperature": (24, 31), "turbidity": (0, 8),
        "icon": "🦞", "growth_rate": "Fast", "market_value": "Very High", "hardiness": "Medium",
        "image_url": "https://ui-avatars.com/api/?name=Prawn&background=65a30d&color=fff&size=40"
    },
    "Silver Cup": {
        "ph": (6.8, 8.2), "temperature": (20, 28), "turbidity": (0, 10),
        "icon": "🥈", "growth_rate": "Slow", "market_value": "Medium", "hardiness": "Medium",
        "image_url": "https://ui-avatars.com/api/?name=Silver+Cup&background=6b7280&color=fff&size=40"
    },
    "Karpio": {
        "ph": (6.8, 8.2), "temperature": (20, 28), "turbidity": (0, 10),
        "icon": "🐡", "growth_rate": "Medium", "market_value": "Low", "hardiness": "Very High",
        "image_url": "https://ui-avatars.com/api/?name=Karpio&background=92400e&color=fff&size=40"
    }
}

def build_species_rules(stats, hard_coded):
    rules = {}
    for species, s in stats.items():
        if species in hard_coded:
            rules[species] = hard_coded[species]
        else:
            # New species – generate default image using species name
            rules[species] = {
                "ph": (s['ph_min'], s['ph_max']),
                "temperature": (s['temperature_min'], s['temperature_max']),
                "turbidity": (s['turbidity_min'], s['turbidity_max']),
                "icon": "🐟",
                "growth_rate": "Medium",
                "market_value": "Medium",
                "hardiness": "Medium",
                "image_url": f"https://ui-avatars.com/api/?name={species.replace(' ', '+')}&background=4b5563&color=fff&size=40"
            }
    return rules

if SPECIES_STATS:
    FISH_SPECIES_RULES = build_species_rules(SPECIES_STATS, HARD_CODED_RULES)
else:
    FISH_SPECIES_RULES = HARD_CODED_RULES

print(f"✅ Loaded {len(FISH_SPECIES_RULES)} fish species for scoring.")

# ================================================================
# 4. SPICE RULES (unchanged)
# ================================================================
SPICE_RULES = {
    "Basil": {"ph": (5.5, 6.5), "temperature": (20, 30), "ec": (1.0, 1.6),
              "icon": "🌿", "growth_time": "30-40 days", "yield": "High", "difficulty": "Easy"},
    "Mint": {"ph": (6.0, 7.0), "temperature": (18, 25), "ec": (1.2, 1.8),
             "icon": "🍃", "growth_time": "40-50 days", "yield": "Very High", "difficulty": "Easy"},
    "Coriander": {"ph": (6.0, 7.0), "temperature": (18, 25), "ec": (1.0, 1.5),
                  "icon": "🌱", "growth_time": "30-40 days", "yield": "Medium", "difficulty": "Easy"},
    "Dill": {"ph": (5.5, 6.5), "temperature": (18, 24), "ec": (0.8, 1.4),
             "icon": "🌿", "growth_time": "40-50 days", "yield": "Medium", "difficulty": "Moderate"},
    "Parsley": {"ph": (5.5, 6.5), "temperature": (18, 25), "ec": (1.0, 1.6),
                "icon": "🌿", "growth_time": "50-70 days", "yield": "Medium", "difficulty": "Moderate"},
    "Chives": {"ph": (6.0, 7.0), "temperature": (15, 25), "ec": (1.0, 1.8),
               "icon": "🧅", "growth_time": "60-80 days", "yield": "High", "difficulty": "Easy"},
    "Oregano": {"ph": (6.0, 7.5), "temperature": (18, 27), "ec": (0.8, 1.6),
                "icon": "🌿", "growth_time": "40-50 days", "yield": "Medium", "difficulty": "Easy"},
    "Thyme": {"ph": (5.5, 7.0), "temperature": (18, 26), "ec": (0.8, 1.5),
              "icon": "🌿", "growth_time": "50-60 days", "yield": "Low", "difficulty": "Moderate"},
    "Rosemary": {"ph": (5.5, 7.0), "temperature": (20, 28), "ec": (1.0, 1.6),
                 "icon": "🌿", "growth_time": "60-90 days", "yield": "Low", "difficulty": "Hard"},
    "Sage": {"ph": (5.5, 6.5), "temperature": (18, 25), "ec": (1.0, 1.6),
             "icon": "🌿", "growth_time": "60-80 days", "yield": "Medium", "difficulty": "Moderate"},
    "Lettuce": {"ph": (5.5, 6.5), "temperature": (15, 22), "ec": (0.8, 1.2),
                "icon": "🥬", "growth_time": "30-40 days", "yield": "Very High", "difficulty": "Easy"},
    "Spinach": {"ph": (6.0, 7.0), "temperature": (15, 22), "ec": (0.8, 1.5),
                "icon": "🥬", "growth_time": "40-50 days", "yield": "High", "difficulty": "Easy"}
}

# ================================================================
# 5. PREDICTION FUNCTIONS (updated to include image_url)
# ================================================================
def predict_fish(ph, temperature, turbidity):
    try:
        sample = pd.DataFrame({
            "ph": [float(ph)],
            "temperature": [float(temperature)],
            "turbidity": [float(turbidity)]
        })
        sample_scaled = scaler.transform(sample)
        pred_encoded = model.predict(sample_scaled)[0]
        species = label_encoder.inverse_transform([pred_encoded])[0]
        proba = model.predict_proba(sample_scaled)[0]
        confidence = max(proba) * 100
        return species, confidence
    except:
        scored = score_fish(ph, temperature, turbidity)
        if scored:
            return scored[0][0], 0.0
        return "Unknown", 0.0

def predict_spice(ph, temperature, ec):
    try:
        sample = pd.DataFrame({
            "ph": [float(ph)],
            "temperature": [float(temperature)],
            "ec": [float(ec)]
        })
        sample_scaled = spice_scaler.transform(sample)
        pred_encoded = spice_model.predict(sample_scaled)[0]
        spice = spice_encoder.inverse_transform([pred_encoded])[0]
        proba = spice_model.predict_proba(sample_scaled)[0]
        confidence = max(proba) * 100
        return spice, confidence
    except:
        scored = score_spice(ph, temperature, ec)
        if scored:
            return scored[0][0], 0.0
        return "Unknown", 0.0

def score_fish(ph, temperature, turbidity):
    scored = []
    for species, rule in FISH_SPECIES_RULES.items():
        score = 0
        if rule["ph"][0] <= ph <= rule["ph"][1]:
            score += 1
        if rule["temperature"][0] <= temperature <= rule["temperature"][1]:
            score += 1
        if rule["turbidity"][0] <= turbidity <= rule["turbidity"][1]:
            score += 1
        scored.append((
            species,
            score,
            rule["icon"],
            rule["growth_rate"],
            rule["market_value"],
            rule["hardiness"],
            rule.get("image_url", ""),
            SPECIES_STATS.get(species, {}).get("count", 0)  # record count
        ))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def score_spice(ph, temperature, ec):
    scored = []
    for spice, rule in SPICE_RULES.items():
        score = 0
        if rule["ph"][0] <= ph <= rule["ph"][1]:
            score += 1
        if rule["temperature"][0] <= temperature <= rule["temperature"][1]:
            score += 1
        if rule["ec"][0] <= ec <= rule["ec"][1]:
            score += 1
        scored.append((spice, score, rule["icon"], rule["growth_time"], rule["yield"], rule["difficulty"]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

# ================================================================
# 6. WATER ANALYSIS (updated to include image URL in result)
# ================================================================
def analyze_water(ph, temperature, turbidity):
    ec = 0.5 + turbidity * 0.1
    problems = []
    suggestions = []

    # General health checks
    if ph < 6.5:
        problems.append("pH too low (acidic)"); suggestions.append("Add agricultural lime slowly.")
    elif ph > 8.5:
        problems.append("pH too high (alkaline)"); suggestions.append("Use buffering or partial water change.")
    if temperature < 20:
        problems.append("Temperature too low"); suggestions.append("Use heater or choose cold‑tolerant species.")
    elif temperature > 32:
        problems.append("Temperature too high"); suggestions.append("Add shade, improve aeration, refresh water.")
    if turbidity > 10:
        problems.append("Turbidity high"); suggestions.append("Improve filtration, reduce feed waste, partial water change.")

    # ---- ML prediction ----
    predicted_fish, conf_fish = predict_fish(ph, temperature, turbidity)
    predicted_spice, conf_spice = predict_spice(ph, temperature, ec)

    # ---- Species-specific dataset comparison ----
    dataset_alerts = []
    if predicted_fish in SPECIES_STATS:
        stats = SPECIES_STATS[predicted_fish]
        if abs(ph - stats['ph_mean']) > 2 * stats['ph_std']:
            dataset_alerts.append(f"pH significantly differs from typical {predicted_fish} (mean {stats['ph_mean']:.2f} ± {stats['ph_std']:.2f})")
        elif abs(ph - stats['ph_mean']) > stats['ph_std']:
            dataset_alerts.append(f"pH slightly off from typical {predicted_fish} (mean {stats['ph_mean']:.2f} ± {stats['ph_std']:.2f})")
        if abs(temperature - stats['temperature_mean']) > 2 * stats['temperature_std']:
            dataset_alerts.append(f"Temperature significantly differs from typical {predicted_fish} (mean {stats['temperature_mean']:.1f} ± {stats['temperature_std']:.1f})")
        elif abs(temperature - stats['temperature_mean']) > stats['temperature_std']:
            dataset_alerts.append(f"Temperature slightly off from typical {predicted_fish} (mean {stats['temperature_mean']:.1f} ± {stats['temperature_std']:.1f})")
        if abs(turbidity - stats['turbidity_mean']) > 2 * stats['turbidity_std']:
            dataset_alerts.append(f"Turbidity significantly differs from typical {predicted_fish} (mean {stats['turbidity_mean']:.1f} ± {stats['turbidity_std']:.1f})")
        elif abs(turbidity - stats['turbidity_mean']) > stats['turbidity_std']:
            dataset_alerts.append(f"Turbidity slightly off from typical {predicted_fish} (mean {stats['turbidity_mean']:.1f} ± {stats['turbidity_std']:.1f})")
    else:
        dataset_alerts.append(f"No reference data available for {predicted_fish} – using general rules.")

    # ---- Risk assessment ----
    abnormal = (not (6.5 <= ph <= 8.5)) + (not (20 <= temperature <= 32)) + (turbidity > 10)
    if abnormal == 0:
        risk, status = "LOW", "Good"
        suggestions.append("Maintain current condition and monitor regularly.")
    elif abnormal == 1:
        risk, status = "MEDIUM", "Needs Attention"
    else:
        risk, status = "HIGH", "Poor"
        suggestions.append("Do not introduce sensitive fish until water stabilizes.")

    # ---- Ranking ----
    fish_ranking = score_fish(ph, temperature, turbidity)
    spice_ranking = score_spice(ph, temperature, ec)
    ideal_for = [name for name, sc, _, _, _, _, _, _ in fish_ranking if sc >= 2][:3]
    avoid_for = [name for name, sc, _, _, _, _, _, _ in fish_ranking if sc == 0][:3]

    # ---- Species-specific advice (from rules) ----
    if predicted_fish in FISH_SPECIES_RULES:
        r = FISH_SPECIES_RULES[predicted_fish]
        if not (r["ph"][0] <= ph <= r["ph"][1]):
            suggestions.append(f"Adjust pH toward {r['ph'][0]}–{r['ph'][1]} for {predicted_fish}.")
        if not (r["temperature"][0] <= temperature <= r["temperature"][1]):
            suggestions.append(f"Adjust temperature toward {r['temperature'][0]}–{r['temperature'][1]}°C for {predicted_fish}.")
        if not (r["turbidity"][0] <= turbidity <= r["turbidity"][1]):
            suggestions.append(f"Reduce turbidity to {r['turbidity'][0]}–{r['turbidity'][1]} NTU for {predicted_fish}.")

    # ---- Merge suggestions and alerts ----
    if dataset_alerts:
        problems.append("Deviation from species reference dataset")
        suggestions.extend(dataset_alerts)

    final_advice = " ".join(dict.fromkeys(suggestions)) or "Water condition is acceptable. Continue routine monitoring."

    # Get image URL for predicted fish
    pred_image = FISH_SPECIES_RULES.get(predicted_fish, {}).get("image_url", "")

    return {
        "ph": ph,
        "temperature": temperature,
        "turbidity": turbidity,
        "ec": ec,
        "predicted_fish": predicted_fish,
        "pred_image": pred_image,
        "confidence_fish": conf_fish,
        "predicted_spice": predicted_spice,
        "confidence_spice": conf_spice,
        "risk_level": risk,
        "water_status": status,
        "problems": problems,
        "advice": final_advice,
        "ideal_for": ideal_for,
        "avoid_for": avoid_for,
        "dataset_alerts": dataset_alerts,
        "fish_ranking": fish_ranking,
        "spice_ranking": spice_ranking
    }

# ================================================================
# 7. FLASK ROUTES
# ================================================================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    ph = data.get('ph')
    temp = data.get('temperature')
    turb = data.get('turbidity')
    if ph is None or temp is None or turb is None:
        return jsonify({"error": "Missing parameters"}), 400
    try:
        ph = float(ph)
        temp = float(temp)
        turb = float(turb)
    except:
        return jsonify({"error": "Invalid numbers"}), 400

    result = analyze_water(ph, temp, turb)
    return jsonify(result)

# ================================================================
# 8. HTML TEMPLATE (with images and all species)
# ================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AQUA·INTELLIGENCE | Live Dashboard</title>
  <!-- Firebase SDKs -->
  <script src="https://www.gstatic.com/firebasejs/9.23.0/firebase-app-compat.js"></script>
  <script src="https://www.gstatic.com/firebasejs/9.23.0/firebase-database-compat.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    /* ===== CSS Variables for Theme ===== */
    :root {
      --bg: #0a0e27;
      --surface: #141a33;
      --surface-light: #1a2245;
      --text: #e0e0e0;
      --text-secondary: #8892b0;
      --border: #233065;
      --card-bg: rgba(20, 26, 51, 0.8);
      --shadow: 0 18px 48px rgba(0,0,0,0.3);
      --input-bg: rgba(0,0,0,0.2);
      --accent: #00d4ff;
      --success: #00e676;
      --warning: #ffab00;
      --danger: #ff5252;
      --gauge-bg: rgba(255,255,255,0.06);
    }
    [data-theme="light"] {
      --bg: #f0f4f8;
      --surface: #ffffff;
      --surface-light: #e8edf3;
      --text: #0f172a;
      --text-secondary: #475569;
      --border: #cbd5e1;
      --card-bg: rgba(255,255,255,0.85);
      --shadow: 0 18px 48px rgba(0,0,0,0.08);
      --input-bg: #ffffff;
      --accent: #4f46e5;
      --success: #16a34a;
      --warning: #d97706;
      --danger: #dc2626;
      --gauge-bg: rgba(0,0,0,0.06);
    }
    /* ===== Global Styles ===== */
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 20px;
      transition: background 0.4s ease, color 0.4s ease;
      background: radial-gradient(circle at top left, rgba(99,102,241,0.08), transparent 30%),
                  radial-gradient(circle at bottom right, rgba(56,189,248,0.06), transparent 30%),
                  var(--bg);
      overflow-x: hidden;
    }
    .particles {
      position: fixed;
      top: 0; left: 0; width: 100%; height: 100%;
      pointer-events: none;
      z-index: 0;
    }
    .particle {
      position: absolute;
      width: 8px; height: 8px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(79,70,229,0.4), transparent);
      animation: drift linear infinite;
      opacity: 0;
    }
    @keyframes drift {
      0% { transform: translateY(0) translateX(0) scale(0.8); opacity: 0; }
      10% { opacity: 0.8; }
      90% { opacity: 0.4; }
      100% { transform: translateY(-120vh) translateX(80px) scale(1.2); opacity: 0; }
    }
    .orb {
      position: fixed;
      border-radius: 50%;
      filter: blur(70px);
      opacity: 0.2;
      pointer-events: none;
      animation: floatOrb 12s ease-in-out infinite;
      z-index: 0;
      transition: opacity 0.4s ease;
    }
    .orb1 { width: 280px; height: 280px; top: 40px; right: 6%; background: #8b5cf6; }
    .orb2 { width: 240px; height: 240px; bottom: 40px; left: 5%; background: #38bdf8; animation-delay: -5s; }
    [data-theme="light"] .orb { opacity: 0.08; }
    @keyframes floatOrb {
      0%, 100% { transform: translateY(0) scale(1); }
      50% { transform: translateY(-30px) scale(1.05); }
    }
    .container {
      position: relative;
      z-index: 1;
      max-width: 1400px;
      margin: auto;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 20px 0;
      border-bottom: 1px solid var(--border);
      margin-bottom: 30px;
      flex-wrap: wrap;
      gap: 16px;
    }
    .header h1 {
      font-size: 28px;
      font-weight: 800;
      background: linear-gradient(90deg, var(--accent), #8b5cf6);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -0.02em;
    }
    .header-controls {
      display: flex;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }
    .toggle-wrap {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--text-secondary);
    }
    .toggle {
      position: relative;
      width: 44px;
      height: 24px;
      background: var(--border);
      border-radius: 999px;
      cursor: pointer;
      transition: 0.3s ease;
      flex-shrink: 0;
    }
    .toggle .slider {
      position: absolute;
      top: 3px;
      left: 3px;
      width: 18px;
      height: 18px;
      background: #fff;
      border-radius: 50%;
      transition: 0.3s ease;
    }
    .toggle.active { background: var(--accent); }
    .toggle.active .slider { left: 23px; }
    .mode-group {
      display: flex;
      background: var(--surface);
      border-radius: 999px;
      border: 1px solid var(--border);
      overflow: hidden;
    }
    .mode-btn {
      padding: 6px 16px;
      font-size: 13px;
      font-weight: 600;
      background: transparent;
      border: none;
      color: var(--text-secondary);
      cursor: pointer;
      transition: 0.25s ease;
    }
    .mode-btn.active {
      background: var(--accent);
      color: #fff;
    }
    .mode-btn:hover:not(.active) { background: var(--surface-light); }
    .live-indicator {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--text-secondary);
    }
    .live-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #00e676;
      animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.8); }
    }
    .live-dot.off { animation: none; opacity: 0.3; }
    .help-btn {
      background: transparent;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 6px 12px;
      color: var(--text-secondary);
      cursor: pointer;
      font-size: 18px;
      line-height: 1;
      transition: 0.25s ease;
    }
    .help-btn:hover { background: var(--surface-light); }

    /* ===== Gauges Row ===== */
    .gauges-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 24px;
      margin-bottom: 30px;
    }
    .gauge-card {
      background: var(--card-bg);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border);
      border-radius: 28px;
      padding: 20px 16px;
      text-align: center;
      box-shadow: var(--shadow);
      transition: transform 0.25s ease, box-shadow 0.25s ease, background 0.4s ease;
    }
    .gauge-card:hover { transform: translateY(-4px); box-shadow: 0 24px 64px rgba(0,0,0,0.15); }
    .gauge-card .label {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }
    .gauge-svg {
      width: 100%;
      max-width: 140px;
      height: auto;
      display: block;
      margin: 0 auto;
    }
    .gauge-value {
      font-size: 28px;
      font-weight: 700;
      margin-top: 6px;
      color: var(--text);
    }
    .gauge-unit {
      font-size: 14px;
      color: var(--text-secondary);
      font-weight: 400;
    }

    /* ===== Main Grid ===== */
    .grid {
      display: grid;
      grid-template-columns: 1.2fr 2fr;
      gap: 24px;
      align-items: start;
    }
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border);
      border-radius: 28px;
      padding: 24px 28px;
      box-shadow: var(--shadow);
      transition: transform 0.25s ease, box-shadow 0.25s ease, background 0.4s ease;
    }
    .card:hover { transform: translateY(-6px); box-shadow: 0 24px 64px rgba(0,0,0,0.15); }
    .card-title {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.2em;
      color: var(--text-secondary);
      margin-bottom: 16px;
      font-weight: 700;
    }

    .prediction-box {
      display: flex;
      flex-wrap: wrap;
      gap: 24px;
      margin-top: 12px;
    }
    .pred-item {
      flex: 1;
      min-width: 150px;
      background: var(--input-bg);
      border-radius: 18px;
      padding: 16px 20px;
      border: 1px solid var(--border);
      transition: 0.3s ease;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .pred-item:hover { background: var(--surface-light); }
    .pred-item .label { font-size: 12px; color: var(--text-secondary); }
    .pred-item .name {
      font-size: 22px;
      font-weight: 700;
      margin: 6px 0;
    }
    .pred-item .conf { font-size: 13px; color: var(--text-secondary); }
    .pred-item.fish .name { color: var(--success); }
    .pred-item.spice .name { color: var(--warning); }
    .pred-item img {
      width: 48px;
      height: 48px;
      border-radius: 50%;
      border: 2px solid var(--border);
    }

    .status-badge {
      display: inline-block;
      padding: 6px 16px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 14px;
    }
    .badge-low { background: var(--success)22; color: var(--success); border: 1px solid var(--success)44; }
    .badge-medium { background: var(--warning)22; color: var(--warning); border: 1px solid var(--warning)44; }
    .badge-high { background: var(--danger)22; color: var(--danger); border: 1px solid var(--danger)44; }
    .badge-critical { background: #ff174422; color: #ff1744; border: 1px solid #ff174444; }

    .rec-list { list-style: none; padding: 0; }
    .rec-list li {
      padding: 8px 0;
      border-bottom: 1px solid var(--border);
      font-size: 15px;
      color: var(--text);
    }
    .rec-list li::before { content: "• "; color: var(--accent); font-weight: 700; }

    .advice-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 12px;
    }
    .advice-item {
      background: var(--input-bg);
      border-radius: 16px;
      padding: 14px 16px;
      border-left: 4px solid var(--accent);
    }
    .advice-item .param {
      font-weight: 600;
      font-size: 14px;
      color: var(--text-secondary);
    }
    .advice-item .action {
      margin-top: 4px;
      font-size: 14px;
      color: var(--text);
    }
    .advice-item.urgent { border-left-color: var(--danger); }
    .advice-item.warning { border-left-color: var(--warning); }
    .advice-item.good { border-left-color: var(--success); }

    .manual-inputs {
      display: none;
      margin-top: 16px;
      gap: 12px;
      flex-wrap: wrap;
    }
    .manual-inputs.active { display: flex; }
    .manual-inputs input {
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--input-bg);
      color: var(--text);
      font-size: 14px;
      width: 100px;
      transition: 0.3s ease;
    }
    .manual-inputs input:focus { outline: none; border-color: var(--accent); }
    .manual-inputs button {
      padding: 10px 24px;
      border-radius: 999px;
      border: none;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
      transition: 0.25s ease;
    }
    .manual-inputs button:hover { transform: scale(1.05); }

    /* ===== Ranking Tables ===== */
    .ranking-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-top: 16px;
    }
    .rank-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    .rank-table th {
      text-align: left;
      color: var(--text-secondary);
      font-weight: 600;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      padding: 6px 4px;
      border-bottom: 1px solid var(--border);
    }
    .rank-table td {
      padding: 6px 4px;
      border-bottom: 1px solid var(--border);
    }
    .rank-table .star { color: #fbbf24; }
    .rank-table .species-img {
      width: 30px;
      height: 30px;
      border-radius: 50%;
      vertical-align: middle;
      margin-right: 6px;
    }

    .timestamp {
      margin-top: 20px;
      font-size: 13px;
      color: var(--text-secondary);
      text-align: right;
    }

    /* ===== Modal ===== */
    .modal-overlay {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.6);
      backdrop-filter: blur(6px);
      z-index: 999;
      justify-content: center;
      align-items: center;
    }
    .modal-overlay.active { display: flex; }
    .modal-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 32px;
      padding: 40px 48px;
      max-width: 700px;
      width: 90%;
      max-height: 80vh;
      overflow-y: auto;
      box-shadow: 0 32px 80px rgba(0,0,0,0.5);
    }
    .modal-box h2 {
      font-size: 26px;
      margin-bottom: 16px;
      color: var(--accent);
    }
    .modal-box p, .modal-box li {
      color: var(--text-secondary);
      font-size: 15px;
      line-height: 1.7;
    }
    .modal-box ul { padding-left: 20px; }
    .modal-box .close-btn {
      margin-top: 24px;
      padding: 10px 28px;
      border: none;
      border-radius: 999px;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }

    /* ===== Responsive ===== */
    @media (max-width: 1100px) {
      .gauges-row { grid-template-columns: repeat(2, 1fr); }
      .grid { grid-template-columns: 1fr; }
      .advice-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 600px) {
      .gauges-row { grid-template-columns: 1fr; }
      .header { flex-direction: column; align-items: stretch; }
      .header-controls { justify-content: space-between; }
      .gauge-svg { max-width: 100px; }
      .pred-item { flex-direction: column; align-items: center; }
    }
  </style>
</head>
<body>
  <div class="particles" id="particles"></div>
  <div class="orb orb1"></div>
  <div class="orb orb2"></div>

  <div class="container">
    <!-- Header -->
    <div class="header">
      <div>
        <h1>🌊 AQUA·INTELLIGENCE</h1>
        <span style="color: var(--text-secondary); font-size: 14px;">Live Aquaculture Monitoring & Predictive Analytics</span>
      </div>
      <div class="header-controls">
        <button class="help-btn" onclick="openModal()">❓</button>
        <div class="toggle-wrap">
          <span>☀️</span>
          <div class="toggle" id="themeToggle" onclick="toggleTheme()">
            <div class="slider"></div>
          </div>
          <span>🌙</span>
        </div>
        <div class="mode-group">
          <button class="mode-btn active" data-mode="auto" onclick="setMode('auto')">Auto</button>
          <button class="mode-btn" data-mode="manual" onclick="setMode('manual')">Manual</button>
        </div>
        <div class="live-indicator">
          <span class="live-dot" id="liveDot"></span>
          <span id="liveLabel">Live</span>
        </div>
      </div>
    </div>

    <!-- Gauges Row -->
    <div class="gauges-row" id="gaugesRow">
      <div class="gauge-card" id="gaugePh">
        <div class="label">pH</div>
        <svg class="gauge-svg" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="50" fill="none" stroke="var(--gauge-bg)" stroke-width="10"/>
          <circle cx="60" cy="60" r="50" fill="none" stroke="var(--accent)" stroke-width="10"
                  stroke-linecap="round" stroke-dasharray="0 314" id="gaugePhArc"/>
          <text x="60" y="60" text-anchor="middle" dy="0.3em" fill="var(--text)" font-size="22" font-weight="700" id="gaugePhVal">--</text>
        </svg>
        <div class="gauge-value" id="gaugePhNum">--</div>
      </div>
      <div class="gauge-card" id="gaugeTemp">
        <div class="label">Temperature</div>
        <svg class="gauge-svg" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="50" fill="none" stroke="var(--gauge-bg)" stroke-width="10"/>
          <circle cx="60" cy="60" r="50" fill="none" stroke="var(--warning)" stroke-width="10"
                  stroke-linecap="round" stroke-dasharray="0 314" id="gaugeTempArc"/>
          <text x="60" y="60" text-anchor="middle" dy="0.3em" fill="var(--text)" font-size="22" font-weight="700" id="gaugeTempVal">--</text>
        </svg>
        <div class="gauge-value" id="gaugeTempNum">-- °C</div>
      </div>
      <div class="gauge-card" id="gaugeTurb">
        <div class="label">Turbidity</div>
        <svg class="gauge-svg" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="50" fill="none" stroke="var(--gauge-bg)" stroke-width="10"/>
          <circle cx="60" cy="60" r="50" fill="none" stroke="var(--accent)" stroke-width="10"
                  stroke-linecap="round" stroke-dasharray="0 314" id="gaugeTurbArc"/>
          <text x="60" y="60" text-anchor="middle" dy="0.3em" fill="var(--text)" font-size="22" font-weight="700" id="gaugeTurbVal">--</text>
        </svg>
        <div class="gauge-value" id="gaugeTurbNum">-- NTU</div>
      </div>
      <div class="gauge-card" id="gaugeEc">
        <div class="label">EC</div>
        <svg class="gauge-svg" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="50" fill="none" stroke="var(--gauge-bg)" stroke-width="10"/>
          <circle cx="60" cy="60" r="50" fill="none" stroke="var(--accent)" stroke-width="10"
                  stroke-linecap="round" stroke-dasharray="0 314" id="gaugeEcArc"/>
          <text x="60" y="60" text-anchor="middle" dy="0.3em" fill="var(--text)" font-size="22" font-weight="700" id="gaugeEcVal">--</text>
        </svg>
        <div class="gauge-value" id="gaugeEcNum">-- mS/cm</div>
      </div>
    </div>

    <!-- Main Grid -->
    <div class="grid">
      <!-- Left Column -->
      <div>
        <!-- Predictions -->
        <div class="card">
          <div class="card-title">🧠 Predictions</div>
          <div class="prediction-box">
            <div class="pred-item fish">
              <img id="predFishImg" src="" alt="Fish">
              <div>
                <div class="label">Best Fish Species</div>
                <div class="name" id="predFish">--</div>
                <div class="conf" id="confFish">Confidence: --</div>
              </div>
            </div>
            <div class="pred-item spice">
              <div style="font-size: 32px;">🌿</div>
              <div>
                <div class="label">Best Herb/Spice</div>
                <div class="name" id="predSpice">--</div>
                <div class="conf" id="confSpice">Confidence: --</div>
              </div>
            </div>
            <div style="flex:1; min-width:140px;">
              <div class="label">Risk Level</div>
              <div id="riskBadge" class="status-badge badge-low">--</div>
              <div style="margin-top:8px; font-size:14px; color:var(--text-secondary);" id="quality">Quality: --</div>
            </div>
          </div>
          <!-- Manual inputs -->
          <div class="manual-inputs" id="manualInputs">
            <input type="number" id="manPh" step="0.1" placeholder="pH" value="7.0">
            <input type="number" id="manTemp" step="0.1" placeholder="Temp °C" value="26">
            <input type="number" id="manTurb" step="0.1" placeholder="Turbidity" value="5">
            <button onclick="manualAnalyze()">Analyze</button>
          </div>
        </div>

        <!-- Water Management Advice -->
        <div class="card" style="margin-top: 20px;">
          <div class="card-title">💧 Water Management Advice</div>
          <div class="advice-grid" id="adviceGrid">
            <div class="advice-item"><div class="param">pH</div><div class="action">Waiting for data...</div></div>
            <div class="advice-item"><div class="param">Temperature</div><div class="action">Waiting for data...</div></div>
            <div class="advice-item"><div class="param">Turbidity</div><div class="action">Waiting for data...</div></div>
            <div class="advice-item"><div class="param">EC</div><div class="action">Waiting for data...</div></div>
          </div>
        </div>

        <!-- Dataset Comparison Alerts -->
        <div class="card" style="margin-top: 20px;">
          <div class="card-title">📊 Dataset Comparison Alerts</div>
          <ul class="rec-list" id="datasetAlerts">
            <li>Waiting for data...</li>
          </ul>
        </div>
      </div>

      <!-- Right Column -->
      <div>
        <!-- Recommendations -->
        <div class="card">
          <div class="card-title">💡 Recommendations</div>
          <ul class="rec-list" id="recList">
            <li>Waiting for data...</li>
          </ul>
        </div>

        <!-- Fish Ranking (ALL species) -->
        <div class="card" style="margin-top: 20px;">
          <div class="card-title">🏆 Fish Compatibility Ranking</div>
          <div style="max-height: 500px; overflow-y: auto;">
            <table class="rank-table" id="fishRankTable">
              <thead><tr><th>Rank</th><th>Species</th><th>Score</th><th>Growth</th><th>Value</th><th>Hardiness</th><th>Records</th></tr></thead>
              <tbody id="fishRankBody"><tr><td colspan="7" style="text-align:center; color:var(--text-secondary);">No data</td></tr></tbody>
            </table>
          </div>
        </div>

        <!-- Spice Ranking -->
        <div class="card" style="margin-top: 20px;">
          <div class="card-title">🌿 Herb/Spice Compatibility Ranking</div>
          <div style="max-height: 300px; overflow-y: auto;">
            <table class="rank-table" id="spiceRankTable">
              <thead><tr><th>Rank</th><th>Herb/Spice</th><th>Score</th><th>Growth Time</th><th>Yield</th><th>Difficulty</th></tr></thead>
              <tbody id="spiceRankBody"><tr><td colspan="6" style="text-align:center; color:var(--text-secondary);">No data</td></tr></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <div class="timestamp" id="timestamp">🕒 Last updated: --</div>
  </div>

  <!-- Modal Instructions -->
  <div class="modal-overlay" id="instructionsModal">
    <div class="modal-box">
      <h2>📖 How to Use the Dashboard</h2>
      <p><strong>Auto Mode</strong> – data is read live from Firebase, then sent to the ML server for prediction.</p>
      <p><strong>Manual Mode</strong> – enter your own pH, temperature, and turbidity, then click Analyze to get ML predictions.</p>
      <ul>
        <li><strong>Theme Toggle</strong> – switch between dark and light themes.</li>
        <li><strong>Live Indicator</strong> – blinks green to confirm the system is active.</li>
        <li><strong>Fish & Spice Predictions</strong> – best species for current water conditions, with confidence scores.</li>
        <li><strong>Risk & Quality</strong> – overall risk level and water quality assessment.</li>
        <li><strong>Water Management Advice</strong> – specific actionable steps for each parameter.</li>
        <li><strong>Dataset Comparison Alerts</strong> – compares current readings against the reference dataset for the predicted fish species.</li>
        <li><strong>Rankings</strong> – lists of compatible fish and herbs/spices ordered by suitability.</li>
      </ul>
      <p><strong>Data Source:</strong> The dashboard reads from Firebase path <code>/aquaculture/sensors</code>. Ensure your ESP8266 is writing data there.</p>
      <button class="close-btn" onclick="closeModal()">Got it!</button>
    </div>
  </div>

  <script>
    // ===== Firebase Config =====
    const firebaseConfig = {
      apiKey: "AIzaSyACk3Kj9agZgUhYmEKGVt0Ul7avCJSoNxY",
      authDomain: "aquaculture-af54d.firebaseapp.com",
      databaseURL: "https://aquaculture-af54d-default-rtdb.firebaseio.com",
      projectId: "aquaculture-af54d",
      storageBucket: "aquaculture-af54d.firebasestorage.app",
      messagingSenderId: "998675976516",
      appId: "1:998675976516:web:c60d56eedc99744c768711",
      measurementId: "G-3GDY6FBCQK"
    };
    firebase.initializeApp(firebaseConfig);
    const db = firebase.database();

    // ===== DOM refs =====
    const gaugePhArc = document.getElementById('gaugePhArc');
    const gaugePhVal = document.getElementById('gaugePhVal');
    const gaugePhNum = document.getElementById('gaugePhNum');
    const gaugeTempArc = document.getElementById('gaugeTempArc');
    const gaugeTempVal = document.getElementById('gaugeTempVal');
    const gaugeTempNum = document.getElementById('gaugeTempNum');
    const gaugeTurbArc = document.getElementById('gaugeTurbArc');
    const gaugeTurbVal = document.getElementById('gaugeTurbVal');
    const gaugeTurbNum = document.getElementById('gaugeTurbNum');
    const gaugeEcArc = document.getElementById('gaugeEcArc');
    const gaugeEcVal = document.getElementById('gaugeEcVal');
    const gaugeEcNum = document.getElementById('gaugeEcNum');

    const predFishEl = document.getElementById('predFish');
    const predFishImg = document.getElementById('predFishImg');
    const confFishEl = document.getElementById('confFish');
    const predSpiceEl = document.getElementById('predSpice');
    const confSpiceEl = document.getElementById('confSpice');
    const riskBadge = document.getElementById('riskBadge');
    const qualityEl = document.getElementById('quality');
    const adviceGrid = document.getElementById('adviceGrid');
    const recList = document.getElementById('recList');
    const datasetAlerts = document.getElementById('datasetAlerts');
    const fishRankBody = document.getElementById('fishRankBody');
    const spiceRankBody = document.getElementById('spiceRankBody');
    const timestampEl = document.getElementById('timestamp');

    // ===== Gauge update =====
    function updateGauge(arcEl, valEl, numEl, value, min, max, unit, colorThresholds) {
      const clamped = Math.min(Math.max(value, min), max);
      const pct = (clamped - min) / (max - min);
      const circumference = 314;
      const dash = pct * circumference;
      arcEl.style.strokeDasharray = dash + ' ' + (circumference - dash);
      let color = colorThresholds.good;
      if (value < colorThresholds.warnLow || value > colorThresholds.warnHigh) color = colorThresholds.warning;
      if (value < colorThresholds.dangerLow || value > colorThresholds.dangerHigh) color = colorThresholds.danger;
      arcEl.style.stroke = color;
      valEl.textContent = value.toFixed(1);
      numEl.textContent = value.toFixed(1) + ' ' + unit;
    }

    // ===== Fetch prediction from server =====
    async function fetchPrediction(ph, temp, turb) {
      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ph, temperature: temp, turbidity: turb })
        });
        const data = await response.json();
        if (data.error) {
          console.error('Prediction error:', data.error);
          return;
        }
        updateUI(data);
      } catch (error) {
        console.error('Error calling /predict:', error);
      }
    }

    // ===== Update UI =====
    function updateUI(data) {
      // Gauges
      updateGauge(gaugePhArc, gaugePhVal, gaugePhNum, data.ph, 0, 14, '',
        { good: 'var(--success)', warnLow: 6.0, warnHigh: 8.0, dangerLow: 5.0, dangerHigh: 9.0 });
      updateGauge(gaugeTempArc, gaugeTempVal, gaugeTempNum, data.temperature, 0, 40, '°C',
        { good: 'var(--success)', warnLow: 20, warnHigh: 33, dangerLow: 15, dangerHigh: 38 });
      updateGauge(gaugeTurbArc, gaugeTurbVal, gaugeTurbNum, data.turbidity, 0, 20, 'NTU',
        { good: 'var(--success)', warnLow: 5, warnHigh: 12, dangerLow: 10, dangerHigh: 18 });
      updateGauge(gaugeEcArc, gaugeEcVal, gaugeEcNum, data.ec, 0, 3.0, 'mS/cm',
        { good: 'var(--success)', warnLow: 1.0, warnHigh: 2.0, dangerLow: 0.6, dangerHigh: 2.5 });

      // Predictions
      predFishEl.textContent = data.predicted_fish;
      predFishImg.src = data.pred_image || 'https://ui-avatars.com/api/?name=Fish&background=4b5563&color=fff&size=40';
      confFishEl.textContent = `Confidence: ${data.confidence_fish.toFixed(0)}%`;
      predSpiceEl.textContent = data.predicted_spice;
      confSpiceEl.textContent = `Confidence: ${data.confidence_spice.toFixed(0)}%`;

      // Risk & Quality
      riskBadge.textContent = data.risk_level;
      riskBadge.className = 'status-badge';
      if (data.risk_level === 'LOW') riskBadge.classList.add('badge-low');
      else if (data.risk_level === 'MEDIUM') riskBadge.classList.add('badge-medium');
      else if (data.risk_level === 'HIGH') riskBadge.classList.add('badge-high');
      else riskBadge.classList.add('badge-critical');
      qualityEl.textContent = `Quality: ${data.water_status}`;

      // Water management advice
      const adviceData = [
        { param: 'pH', action: getPhAdvice(data.ph), urgency: getPhUrgency(data.ph) },
        { param: 'Temperature', action: getTempAdvice(data.temperature), urgency: getTempUrgency(data.temperature) },
        { param: 'Turbidity', action: getTurbAdvice(data.turbidity), urgency: getTurbUrgency(data.turbidity) },
        { param: 'EC', action: getEcAdvice(data.ec), urgency: getEcUrgency(data.ec) }
      ];
      adviceGrid.innerHTML = '';
      adviceData.forEach(item => {
        const div = document.createElement('div');
        div.className = `advice-item ${item.urgency}`;
        div.innerHTML = `<div class="param">${item.param}</div><div class="action">${item.action}</div>`;
        adviceGrid.appendChild(div);
      });

      // Recommendations (general)
      recList.innerHTML = '';
      if (data.recommendations && data.recommendations.length > 0) {
        data.recommendations.forEach(r => {
          const li = document.createElement('li');
          li.textContent = r;
          recList.appendChild(li);
        });
      } else {
        recList.innerHTML = '<li>✅ All parameters optimal! Maintain current conditions.</li>';
      }

      // Dataset alerts
      datasetAlerts.innerHTML = '';
      if (data.dataset_alerts && data.dataset_alerts.length > 0) {
        data.dataset_alerts.forEach(a => {
          const li = document.createElement('li');
          li.textContent = a;
          datasetAlerts.appendChild(li);
        });
      } else {
        datasetAlerts.innerHTML = '<li>✅ No significant deviation from species reference data.</li>';
      }

      // Fish ranking – show ALL species
      fishRankBody.innerHTML = '';
      data.fish_ranking.forEach((item, idx) => {
        const tr = document.createElement('tr');
        const stars = '★'.repeat(item[1]) + '☆'.repeat(3 - item[1]);
        const imgUrl = item[6] || 'https://ui-avatars.com/api/?name=' + encodeURIComponent(item[0]) + '&background=4b5563&color=fff&size=30';
        tr.innerHTML = `
          <td>${idx+1}</td>
          <td><img src="${imgUrl}" class="species-img" alt="${item[0]}"> ${item[0]}</td>
          <td class="star">${stars}</td>
          <td>${item[3]}</td>
          <td>${item[4]}</td>
          <td>${item[5]}</td>
          <td>${item[7]}</td>
        `;
        fishRankBody.appendChild(tr);
      });

      // Spice ranking
      spiceRankBody.innerHTML = '';
      data.spice_ranking.slice(0, 10).forEach((item, idx) => {
        const tr = document.createElement('tr');
        const stars = '★'.repeat(item[1]) + '☆'.repeat(3 - item[1]);
        tr.innerHTML = `<td>${idx+1}</td><td>${item[2]} ${item[0]}</td><td class="star">${stars}</td><td>${item[3]}</td><td>${item[4]}</td><td>${item[5]}</td>`;
        spiceRankBody.appendChild(tr);
      });

      timestampEl.textContent = `🕒 Last updated: ${new Date().toLocaleString()}`;
    }

    // ===== Helpers for advice =====
    function getPhAdvice(ph) {
      if (ph < 6.0) return '⚠️ pH too low! Add agricultural lime or potassium bicarbonate.';
      if (ph < 6.5) return 'pH slightly low – add lime to bring to 6.5–7.5.';
      if (ph > 8.5) return '⚠️ pH too high! Reduce using phosphoric acid or organic matter.';
      if (ph > 7.5) return 'pH slightly high – consider acidifying for optimal plant growth.';
      return '✅ pH optimal (6.5–7.5).';
    }
    function getPhUrgency(ph) {
      if (ph < 6.0 || ph > 8.5) return 'urgent';
      if (ph < 6.5 || ph > 7.5) return 'warning';
      return 'good';
    }
    function getTempAdvice(temp) {
      if (temp < 20) return '❄️ Too cold! Use heater or choose cold-tolerant species.';
      if (temp < 24) return 'Cool – good for herbs; fish may grow slower.';
      if (temp > 32) return '🔥 Too hot! Increase aeration, add shade, refresh water.';
      if (temp > 30) return 'Warm – acceptable for tropical fish, but monitor.';
      return '✅ Temperature optimal (24–30°C).';
    }
    function getTempUrgency(temp) {
      if (temp < 20 || temp > 32) return 'urgent';
      if (temp < 24 || temp > 30) return 'warning';
      return 'good';
    }
    function getTurbAdvice(turb) {
      if (turb > 10) return '⚠️ Very cloudy! Improve filtration, reduce feeding, partial water change.';
      if (turb > 5) return 'Moderate turbidity – monitor and clean filters.';
      return '✅ Water clear (≤5 NTU).';
    }
    function getTurbUrgency(turb) {
      if (turb > 10) return 'urgent';
      if (turb > 5) return 'warning';
      return 'good';
    }
    function getEcAdvice(ec) {
      if (ec < 0.8) return '⚠️ EC too low – nutrients insufficient. Add feed or nutrients.';
      if (ec < 1.2) return 'EC low – good for leafy greens, add minerals for fish.';
      if (ec > 2.0) return '⚠️ EC too high – possible toxicity. Dilute with fresh water.';
      if (ec > 1.8) return 'EC high – monitor salt accumulation, partial water change.';
      return '✅ EC optimal (1.2–1.8).';
    }
    function getEcUrgency(ec) {
      if (ec < 0.8 || ec > 2.0) return 'urgent';
      if (ec < 1.2 || ec > 1.8) return 'warning';
      return 'good';
    }

    // ===== Modes =====
    let currentMode = 'auto';
    let manualData = null;

    function setMode(mode) {
      currentMode = mode;
      document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
      });
      const manualInputs = document.getElementById('manualInputs');
      if (mode === 'manual') {
        manualInputs.classList.add('active');
        if (manualData) {
          fetchPrediction(manualData.ph, manualData.temp, manualData.turb);
        } else {
          manualAnalyze();
        }
      } else {
        manualInputs.classList.remove('active');
        if (lastFirebaseData) {
          fetchPrediction(lastFirebaseData.ph, lastFirebaseData.temp, lastFirebaseData.turb);
        }
      }
    }

    function manualAnalyze() {
      const ph = parseFloat(document.getElementById('manPh').value);
      const temp = parseFloat(document.getElementById('manTemp').value);
      const turb = parseFloat(document.getElementById('manTurb').value);
      if (isNaN(ph) || isNaN(temp) || isNaN(turb)) {
        alert('Please enter valid numbers');
        return;
      }
      manualData = { ph, temp, turb };
      fetchPrediction(ph, temp, turb);
    }

    // ===== Firebase Listener =====
    let lastFirebaseData = null;
    const sensorsRef = db.ref('aquaculture/sensors');
    sensorsRef.on('value', (snapshot) => {
      const val = snapshot.val();
      if (val) {
        const ph = val.ph || 0;
        const temp = val.temperature || 0;
        const turb = val.turbidity || 0;
        if (ph > 0 || temp > 0 || turb > 0) {
          lastFirebaseData = { ph, temp, turb };
          if (currentMode === 'auto') {
            fetchPrediction(ph, temp, turb);
          }
        } else {
          const demo = { ph: 7.2, temp: 27.5, turb: 4.8 };
          lastFirebaseData = demo;
          if (currentMode === 'auto') fetchPrediction(demo.ph, demo.temp, demo.turb);
        }
      } else {
        const demo = { ph: 7.2, temp: 27.5, turb: 4.8 };
        lastFirebaseData = demo;
        if (currentMode === 'auto') fetchPrediction(demo.ph, demo.temp, demo.turb);
      }
    }, (error) => {
      console.error('Firebase read error:', error);
      const demo = { ph: 7.0, temp: 26.0, turb: 5.0 };
      lastFirebaseData = demo;
      if (currentMode === 'auto') fetchPrediction(demo.ph, demo.temp, demo.turb);
    });

    // ===== Theme Toggle =====
    function toggleTheme() {
      const toggle = document.getElementById('themeToggle');
      const isLight = document.documentElement.getAttribute('data-theme') === 'light';
      if (isLight) {
        document.documentElement.removeAttribute('data-theme');
        toggle.classList.remove('active');
      } else {
        document.documentElement.setAttribute('data-theme', 'light');
        toggle.classList.add('active');
      }
    }

    // ===== Modal =====
    function openModal() {
      document.getElementById('instructionsModal').classList.add('active');
    }
    function closeModal() {
      document.getElementById('instructionsModal').classList.remove('active');
    }
    document.getElementById('instructionsModal').addEventListener('click', function(e) {
      if (e.target === this) closeModal();
    });

    // ===== Particles =====
    (function createParticles() {
      const container = document.getElementById('particles');
      for (let i = 0; i < 25; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        p.style.left = Math.random() * 100 + '%';
        p.style.bottom = (-10 - Math.random() * 30) + 'px';
        p.style.animationDuration = (12 + Math.random() * 14) + 's';
        p.style.animationDelay = (-Math.random() * 14) + 's';
        p.style.width = p.style.height = (6 + Math.random() * 12) + 'px';
        container.appendChild(p);
      }
    })();

    // ===== Initial load =====
    setTimeout(() => {
      if (!lastFirebaseData) {
        const demo = { ph: 7.2, temp: 27.5, turb: 4.8 };
        lastFirebaseData = demo;
        if (currentMode === 'auto') fetchPrediction(demo.ph, demo.temp, demo.turb);
      }
    }, 2000);
  </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)