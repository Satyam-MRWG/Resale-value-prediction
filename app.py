from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil, uuid, os
import pandas as pd
from typing import Tuple, Dict, List

# ── Import helpers from predict.py ─────────────
from src.predict import predict_price, calculate_damage_penalty, validate_input

app = FastAPI(title="Vehicle Health AI API")

# Allow requests from your Node.js backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models","best.pt")
model = YOLO(MODEL_PATH)

# ✅ UPDATED: Realistic Indian market repair costs (₹)
DAMAGE_COST = {
    "dent":          8000,   # panel beating / body shop
    "scratch":       3000,   # paint + polish
    "crack":        15000,   # bumper or panel replacement
    "glass shatter": 8000,   # windshield replacement
    "lamp broken":   6000,   # headlight / taillight unit
    "tire flat":     4000,   # tyre replacement
    "Car-Damage":   20000,   # generic heavy damage
}

# Severity levels
SEVERITY = {
    "dent":          "Medium",
    "scratch":       "Low",
    "crack":         "High",
    "glass shatter": "Critical",
    "lamp broken":   "High",
    "tire flat":     "Critical",
    "Car-Damage":    "High",
}

# ✅ UPDATED: Smarter health score — accounts for severity + confidence + count
def calculate_health_score(damage_counts: dict, damage_details: list) -> int:
    """100 = perfect condition, 0 = total wreck"""
    if not damage_counts:
        return 100

    penalty = 0

    # Base penalty per damage type (weighted by severity)
    type_penalty = {
        "glass shatter": 35,
        "tire flat":     30,
        "crack":         22,
        "Car-Damage":    18,
        "lamp broken":   15,
        "dent":          12,
        "scratch":        6,
    }

    for damage, count in damage_counts.items():
        base = type_penalty.get(damage, 10)
        # Multiple damages of same type = exponentially worse
        penalty += base * count * (1 + (count - 1) * 0.4)

    # Extra penalty based on detection confidence
    # High confidence = model is very sure = definitely damaged
    for detail in damage_details:
        conf = detail["confidence"]
        if conf >= 85:
            penalty += 10   # very confident = serious damage
        elif conf >= 70:
            penalty += 5
        elif conf >= 55:
            penalty += 2

    # Extra penalty for sheer number of damages
    total = sum(damage_counts.values())
    if total >= 6:
        penalty += 30
    elif total >= 4:
        penalty += 18
    elif total >= 3:
        penalty += 10
    elif total >= 2:
        penalty += 5

    return max(0, 100 - int(penalty))


def get_health_label(score: int) -> str:
    if score >= 85: return "Excellent"
    if score >= 70: return "Good"
    if score >= 50: return "Fair"
    if score >= 30: return "Poor"
    return "Critical"


# ✅ UPDATED: Realistic repair cost with labour multiplier
def calculate_repair_cost(damage_counts: dict) -> int:
    base_cost = sum(
        DAMAGE_COST.get(k, 5000) * v
        for k, v in damage_counts.items()
    )
    total = sum(damage_counts.values())

    # Labour surcharge for complex multi-damage repairs
    if total >= 5:
        return int(base_cost * 1.35)   # 35% extra
    elif total >= 3:
        return int(base_cost * 1.20)   # 20% extra
    elif total >= 2:
        return int(base_cost * 1.10)   # 10% extra
    return base_cost


# ✅ UPDATED: Smarter, context-aware recommendations
def get_recommendations(damage_counts: dict, health_score: int) -> list:
    recommendations = []

    # Overall vehicle condition warning
    if health_score < 30:
        recommendations.append(
            "🚨 CRITICAL: Vehicle is in very poor condition — unsafe to drive. Get immediate professional inspection."
        )
    elif health_score < 50:
        recommendations.append(
            "⚠️ WARNING: Significant damage detected — avoid driving until repairs are done."
        )
    elif health_score < 70:
        recommendations.append(
            "🔧 Moderate damage found — schedule a repair appointment within the next 2 weeks."
        )
    elif health_score < 85:
        recommendations.append(
            "🔍 Minor damage found — repairs recommended at your next scheduled service."
        )

    # Per-damage specific advice
    for dmg, count in damage_counts.items():
        if dmg == "glass shatter":
            recommendations.append(
                f"🚨 URGENT: Shattered glass detected — critical safety hazard. Do NOT drive. Replace windshield/glass immediately."
            )
        elif dmg == "tire flat":
            recommendations.append(
                f"🚨 URGENT: Flat tyre detected — do NOT drive the vehicle. Replace tyre before moving."
            )
        elif dmg == "crack":
            recommendations.append(
                f"⚠️ {count} crack(s) found — may affect structural integrity. Get bumper/panel inspected and replaced if needed."
            )
        elif dmg == "lamp broken":
            recommendations.append(
                f"⚠️ {count} broken lamp(s) — driving at night is illegal without working lights. Replace headlight/taillight soon."
            )
        elif dmg == "dent":
            recommendations.append(
                f"🔧 {count} dent(s) detected — visit a body shop for panel beating or PDR (Paintless Dent Repair). Estimated cost: ₹{8000 * count:,}"
            )
        elif dmg == "scratch":
            recommendations.append(
                f"🔍 {count} scratch(es) found — polish and paint touch-up recommended to prevent rust. Estimated cost: ₹{3000 * count:,}"
            )
        elif dmg == "Car-Damage":
            recommendations.append(
                f"⚠️ General heavy damage detected — full professional inspection recommended."
            )

    if not damage_counts:
        recommendations.append("✅ No visible damage detected — your vehicle appears to be in great condition!")

    return recommendations


# ── Shared YOLO detection helper ───────────────
def run_yolo_detection(temp_path: str) -> Tuple[Dict, List]:
    """Run YOLO on image, return (damage_counts, damage_details)."""
    results        = model(temp_path, conf=0.40)
    damage_counts  = {}
    damage_details = []

    for box in results[0].boxes:
        cls_id       = int(box.cls)
        conf         = float(box.conf)
        label        = model.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        damage_counts[label] = damage_counts.get(label, 0) + 1
        damage_details.append({
            "type":       label,
            "confidence": round(conf * 100, 1),
            "severity":   SEVERITY.get(label, "Medium"),
            "location": {
                "x1": round(x1),
                "y1": round(y1),
                "x2": round(x2),
                "y2": round(y2),
            }
        })

    damage_details.sort(key=lambda x: x["confidence"], reverse=True)
    return damage_counts, damage_details


# ── Shared image save/cleanup helper ──────────
async def save_temp_image(file: UploadFile) -> Tuple[str, bytes]:
    """Validate, read, and save uploaded image. Returns (temp_path, contents)."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are allowed (jpg, png, webp)")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "Image too large — maximum size is 10MB")

    temp_dir  = os.path.join(os.path.dirname(__file__), "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")

    with open(temp_path, "wb") as f:
        f.write(contents)

    return temp_path, contents


@app.get("/")
def root():
    return {"status": "Vehicle Health AI running ✅", "version": "2.0"}


# ── /analyze — image only (original, unchanged) ─
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    temp_path, _ = await save_temp_image(file)

    try:
        damage_counts, damage_details = run_yolo_detection(temp_path)

        health_score         = calculate_health_score(damage_counts, damage_details)
        health_label         = get_health_label(health_score)
        total_damages        = sum(damage_counts.values())
        estimated_cost       = calculate_repair_cost(damage_counts)
        recommendations      = get_recommendations(damage_counts, health_score)

        if health_score >= 85:
            risk_level = "Low"
        elif health_score >= 60:
            risk_level = "Medium"
        elif health_score >= 35:
            risk_level = "High"
        else:
            risk_level = "Critical"

        return {
            "success":               True,
            "health_score":          health_score,
            "health_label":          health_label,
            "risk_level":            risk_level,
            "total_damages":         total_damages,
            "damage_counts":         damage_counts,
            "damage_details":        damage_details,
            "estimated_repair_cost": estimated_cost,
            "recommendations":       recommendations,
            "model_version":         "CarDD-YOLOv8n-v2",
            "note": "Costs are estimates based on average Indian market rates. Actual costs may vary by city and workshop."
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── /predict — image + tabular columns → resale price ─
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),

    # Numeric features
    engine_capacity_cc  : float = Form(...),
    car_age_in_year     : float = Form(...),
    kms_driven          : float = Form(...),
    kms_per_year        : float = Form(...),
    max_power           : float = Form(...),
    mileage             : float = Form(...),
    seats               : int   = Form(...),

    # Encoded categoricals
    owner_type          : int   = Form(...),   # 1=First, 2=Second, 3=Third
    transmission_type   : int   = Form(...),   # 0=Manual, 1=Automatic
    insurance           : int   = Form(...),   # 0=No, 1=Yes

    # One-hot: brand
    brand_Maruti        : int   = Form(0),
    brand_Tata          : int   = Form(0),

    # One-hot: fuel
    fuel_type_Petrol    : int   = Form(0),
    fuel_type_Diesel    : int   = Form(0),

    # One-hot: body
    body_type_Hatchback : int   = Form(0),
    body_type_Sedan     : int   = Form(0),

    # One-hot: city
    city_Agra           : int   = Form(0),
):
    temp_path, _ = await save_temp_image(file)

    try:
        # ── Build tabular DataFrame ──
        input_df = pd.DataFrame([{
            "engine_capacity_cc":  engine_capacity_cc,
            "car_age_in_year":     car_age_in_year,
            "kms_driven":          kms_driven,
            "kms_per_year":        kms_per_year,
            "max_power":           max_power,
            "mileage":             mileage,
            "seats":               seats,
            "owner_type":          owner_type,
            "transmission_type":   transmission_type,
            "insurance":           insurance,
            "brand_Maruti":        brand_Maruti,
            "brand_Tata":          brand_Tata,
            "fuel_type_Petrol":    fuel_type_Petrol,
            "fuel_type_Diesel":    fuel_type_Diesel,
            "body_type_Hatchback": body_type_Hatchback,
            "body_type_Sedan":     body_type_Sedan,
            "city_Agra":           city_Agra,
        }])

        # Validate tabular input (raises ValueError on bad values)
        try:
            validate_input(input_df)
        except ValueError as e:
            raise HTTPException(400, str(e))

        # ── YOLO damage detection ──
        damage_counts, damage_details = run_yolo_detection(temp_path)

        # ── Health metrics ──
        health_score    = calculate_health_score(damage_counts, damage_details)
        health_label    = get_health_label(health_score)
        total_damages   = sum(damage_counts.values())
        repair_cost     = calculate_repair_cost(damage_counts)
        recommendations = get_recommendations(damage_counts, health_score)
        risk_level      = (
            "Low"    if health_score >= 85 else
            "Medium" if health_score >= 60 else
            "High"   if health_score >= 35 else "Critical"
        )

        # ── Resale price from predict.py ──
        base_price     = predict_price(input_df)
        damage_penalty = calculate_damage_penalty(damage_counts, damage_details)
        final_price    = round(base_price * damage_penalty, 2)
        penalty_pct    = round((1 - damage_penalty) * 100, 1)

        return {
            "success": True,

            "resale": {
                "base_price":         round(base_price, 2),
                "damage_penalty_pct": penalty_pct,
                "final_resale_price": final_price,
                "currency":           "INR",
            },

            "vehicle_health": {
                "health_score":          health_score,
                "health_label":          health_label,
                "risk_level":            risk_level,
                "total_damages":         total_damages,
                "damage_counts":         damage_counts,
                "damage_details":        damage_details,
                "estimated_repair_cost": repair_cost,
                "recommendations":       recommendations,
            },

            "model_version": "CarDD-YOLOv8n-v2",
            "note": "Resale price and repair costs are estimates based on Indian market rates.",
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)