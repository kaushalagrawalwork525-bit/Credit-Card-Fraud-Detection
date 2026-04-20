import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from geopy.distance import geodesic


st.set_page_config(
    page_title="FraudShield 360",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


MODEL_PATH = Path(__file__).with_name("fraud_model.pkl")
model = joblib.load(MODEL_PATH)


MERCHANT_PROFILES = {
    "fraud_Rippin, Kub and Mann": {
        "category": "grocery_pos",
        "merchant_city": "Delhi NCR",
        "merch_lat": 28.7041,
        "merch_lon": 77.1025,
        "typical_amount": 2400.0,
    },
    "fraud_Heller, Gutmann and Zieme": {
        "category": "shopping_pos",
        "merchant_city": "Mumbai",
        "merch_lat": 19.0760,
        "merch_lon": 72.8777,
        "typical_amount": 6800.0,
    },
    "fraud_Lind-Buckridge": {
        "category": "gas_transport",
        "merchant_city": "Jaipur",
        "merch_lat": 26.9124,
        "merch_lon": 75.7873,
        "typical_amount": 1800.0,
    },
    "fraud_Kutch, Hermiston and Farrell": {
        "category": "shopping_net",
        "merchant_city": "Bengaluru",
        "merch_lat": 12.9716,
        "merch_lon": 77.5946,
        "typical_amount": 9800.0,
    },
    "fraud_Keeling-Crist": {
        "category": "misc_net",
        "merchant_city": "Hyderabad",
        "merch_lat": 17.3850,
        "merch_lon": 78.4867,
        "typical_amount": 5600.0,
    },
}

CATEGORY_DEFAULTS = {
    "grocery_pos": 2200.0,
    "shopping_pos": 6200.0,
    "gas_transport": 1500.0,
    "shopping_net": 9200.0,
    "misc_net": 5400.0,
    "misc_pos": 2800.0,
    "food_dining": 1800.0,
    "entertainment": 3500.0,
    "health_fitness": 2600.0,
    "travel": 12500.0,
    "kids_pets": 2100.0,
    "home": 7400.0,
    "personal_care": 1700.0,
}

CUSTOMER_CITIES = {
    "New Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Jaipur": (26.9124, 75.7873),
    "Pune": (18.5204, 73.8567),
}

CUSTOMER_PROFILES = {
    "Aarav Sharma - Retail": {
        "customer_city": "New Delhi",
        "gender": "M",
        "card_segment": "Classic",
        "device_trust": 74,
        "velocity_24h": 2,
        "international_txn": False,
        "card_number": "4539682995824395",
        "card_mode": "Credit Card",
    },
    "Neha Verma - Premium": {
        "customer_city": "Mumbai",
        "gender": "F",
        "card_segment": "Gold",
        "device_trust": 66,
        "velocity_24h": 4,
        "international_txn": False,
        "card_number": "5123456789012345",
        "card_mode": "Credit Card",
    },
    "Rohan Iyer - Travel Heavy": {
        "customer_city": "Bengaluru",
        "gender": "M",
        "card_segment": "Platinum",
        "device_trust": 58,
        "velocity_24h": 5,
        "international_txn": True,
        "card_number": "6011222233334444",
        "card_mode": "Debit Card",
    },
}

CARD_NETWORKS = {
    "4": "Visa",
    "5": "Mastercard",
    "6": "RuPay / Discover",
    "3": "American Express",
}

BANK_BY_PREFIX = {
    "4": "Horizon Bank",
    "5": "Zenith Bank",
    "6": "National Secure Bank",
    "3": "Elite Capital Bank",
}

CARD_SEGMENTS = {
    "Platinum": "High-value customer segment",
    "Gold": "Premium segment with elevated limits",
    "Classic": "Standard retail card segment",
}

merchant_map = {name: index for index, name in enumerate(MERCHANT_PROFILES)}
category_map = {
    "grocery_pos": 0,
    "shopping_pos": 1,
    "gas_transport": 2,
    "shopping_net": 3,
    "misc_net": 4,
    "misc_pos": 5,
    "food_dining": 6,
    "entertainment": 7,
    "health_fitness": 8,
    "travel": 9,
    "kids_pets": 10,
    "home": 11,
    "personal_care": 12,
}
gender_map = {"M": 0, "F": 1}


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(8, 145, 178, 0.22), transparent 26%),
                    radial-gradient(circle at top right, rgba(59, 130, 246, 0.18), transparent 24%),
                    radial-gradient(circle at bottom left, rgba(34, 197, 94, 0.10), transparent 18%),
                    linear-gradient(150deg, #030712 0%, #08111f 35%, #0f172a 100%);
                color: #e5eef9;
                font-family: "Segoe UI", "Trebuchet MS", sans-serif;
            }
            .block-container {
                padding-top: 1rem;
                padding-bottom: 2.2rem;
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(2, 6, 23, 0.95), rgba(15, 23, 42, 0.92));
                border-right: 1px solid rgba(148, 163, 184, 0.12);
            }
            .hero-shell, .glass-card, .alert-card, .kpi-card {
                border: 1px solid rgba(148, 163, 184, 0.18);
                backdrop-filter: blur(12px);
                box-shadow: 0 18px 45px rgba(2, 8, 23, 0.34);
            }
            .hero-shell {
                padding: 28px;
                border-radius: 28px;
                background: linear-gradient(140deg, rgba(8, 47, 73, 0.82), rgba(15, 23, 42, 0.90));
                margin-bottom: 1rem;
                position: relative;
                overflow: hidden;
            }
            .glass-card {
                padding: 22px;
                border-radius: 24px;
                background: linear-gradient(145deg, rgba(15, 23, 42, 0.82), rgba(17, 24, 39, 0.72));
                margin-bottom: 1rem;
            }
            .kpi-card {
                padding: 18px;
                border-radius: 20px;
                background: linear-gradient(135deg, rgba(14, 165, 233, 0.22), rgba(15, 23, 42, 0.8));
                min-height: 132px;
            }
            .alert-card {
                padding: 20px;
                border-radius: 22px;
                background: linear-gradient(135deg, rgba(127, 29, 29, 0.58), rgba(15, 23, 42, 0.86));
            }
            .soft-card {
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.04);
                padding: 14px 16px;
                border: 1px solid rgba(148, 163, 184, 0.1);
            }
            .eyebrow {
                letter-spacing: 0.16em;
                text-transform: uppercase;
                font-size: 0.78rem;
                color: #7dd3fc;
                font-weight: 700;
            }
            .hero-title {
                font-size: 2.3rem;
                font-weight: 800;
                margin-bottom: 0.4rem;
            }
            .hero-copy {
                color: #cbd5e1;
                font-size: 1.02rem;
                line-height: 1.65;
                max-width: 900px;
            }
            .hero-strip {
                display:grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 12px;
                margin-top: 18px;
            }
            .hero-pill {
                padding: 12px 14px;
                border-radius: 18px;
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(148, 163, 184, 0.14);
                color: #dbeafe;
                font-size: 0.9rem;
            }
            div[data-testid="stTabs"] button {
                border-radius: 999px;
                padding: 0.55rem 1rem;
            }
            div[data-testid="stTabs"] button[aria-selected="true"] {
                background: linear-gradient(135deg, rgba(14, 165, 233, 0.28), rgba(37, 99, 235, 0.22));
            }
            div[data-testid="stMetric"] {
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(148, 163, 184, 0.08);
                border-radius: 18px;
                padding: 12px;
            }
            .signal-row {
                display:flex;
                justify-content:space-between;
                align-items:center;
                padding: 12px 14px;
                border-radius: 16px;
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(148,163,184,0.08);
                margin-bottom: 10px;
            }
            .footer {
                color: #94a3b8;
                text-align: center;
                margin-top: 1.4rem;
                padding: 0.6rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return geodesic((lat1, lon1), (lat2, lon2)).km


def infer_network(card_number: str) -> str:
    number = "".join(ch for ch in str(card_number) if ch.isdigit())
    if not number:
        return "Unknown"
    return CARD_NETWORKS.get(number[0], "Unknown")


def infer_bank(card_number: str) -> str:
    number = "".join(ch for ch in str(card_number) if ch.isdigit())
    if not number:
        return "Unknown Bank"
    return BANK_BY_PREFIX.get(number[0], "Unknown Bank")


def mask_card(card_number: str) -> str:
    digits = "".join(ch for ch in str(card_number) if ch.isdigit())
    if len(digits) < 4:
        return "••••"
    return f"•••• •••• •••• {digits[-4:]}"


def risk_palette(risk_percent: float) -> tuple[str, str]:
    if risk_percent >= 75:
        return "#ef4444", "rgba(127, 29, 29, 0.55)"
    if risk_percent >= 50:
        return "#f97316", "rgba(124, 45, 18, 0.52)"
    if risk_percent >= 30:
        return "#eab308", "rgba(113, 63, 18, 0.42)"
    return "#22c55e", "rgba(20, 83, 45, 0.42)"


def render_card_visual(title: str, card_number: str, card_mode: str, risk_percent: float, segment: str) -> None:
    accent, surface = risk_palette(risk_percent)
    network = infer_network(card_number)
    bank = infer_bank(card_number)
    st.markdown(
        f"""
        <div style="
            border-radius: 28px;
            padding: 22px;
            min-height: 220px;
            background:
                radial-gradient(circle at 15% 20%, rgba(255,255,255,0.18), transparent 24%),
                radial-gradient(circle at 85% 18%, rgba(255,255,255,0.14), transparent 16%),
                linear-gradient(145deg, {surface}, rgba(15, 23, 42, 0.95));
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 20px 50px rgba(2, 8, 23, 0.34);
            position: relative;
            overflow: hidden;
            margin-bottom: 1rem;
        ">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div style="font-size:0.78rem; letter-spacing:0.16em; color:#bfdbfe; text-transform:uppercase;">{title}</div>
                    <div style="font-size:1.25rem; font-weight:800; margin-top:8px;">{bank}</div>
                </div>
                <div style="text-align:right;">
                    <div style="display:inline-block; padding:8px 12px; border-radius:999px; background:rgba(255,255,255,0.08); color:#f8fafc; font-size:0.82rem;">
                        {card_mode}
                    </div>
                </div>
            </div>
            <div style="margin-top:28px; width:58px; height:42px; border-radius:12px; background:linear-gradient(160deg, rgba(252,211,77,0.9), rgba(161,98,7,0.85));"></div>
            <div style="margin-top:20px; font-size:1.55rem; letter-spacing:0.14em; font-weight:700;">{mask_card(card_number)}</div>
            <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-top:24px;">
                <div>
                    <div style="font-size:0.75rem; color:#cbd5e1;">Segment</div>
                    <div style="font-size:1rem; font-weight:700;">{segment}</div>
                </div>
                <div>
                    <div style="font-size:0.75rem; color:#cbd5e1;">Network</div>
                    <div style="font-size:1rem; font-weight:700; text-align:right;">{network}</div>
                </div>
                <div>
                    <div style="font-size:0.75rem; color:#cbd5e1;">Risk</div>
                    <div style="font-size:1rem; font-weight:700; color:{accent}; text-align:right;">{risk_percent:.2f}%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults = {
        "customer_profile": list(CUSTOMER_PROFILES.keys())[0],
        "merchant": list(MERCHANT_PROFILES.keys())[0],
        "category": MERCHANT_PROFILES[list(MERCHANT_PROFILES.keys())[0]]["category"],
        "amt": 2400.0,
        "card_number": "4539682995824395",
        "card_mode": "Credit Card",
        "customer_city": "New Delhi",
        "merchant_city": MERCHANT_PROFILES[list(MERCHANT_PROFILES.keys())[0]]["merchant_city"],
        "gender": "M",
        "lat": CUSTOMER_CITIES["New Delhi"][0],
        "lon": CUSTOMER_CITIES["New Delhi"][1],
        "merch_lat": MERCHANT_PROFILES[list(MERCHANT_PROFILES.keys())[0]]["merch_lat"],
        "merch_lon": MERCHANT_PROFILES[list(MERCHANT_PROFILES.keys())[0]]["merch_lon"],
        "hour": 14,
        "day": 15,
        "month": 4,
        "card_segment": "Classic",
        "device_trust": 72,
        "velocity_24h": 3,
        "international_txn": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def sync_from_merchant() -> None:
    profile = MERCHANT_PROFILES[st.session_state["merchant"]]
    st.session_state["category"] = profile["category"]
    st.session_state["merchant_city"] = profile["merchant_city"]
    st.session_state["merch_lat"] = profile["merch_lat"]
    st.session_state["merch_lon"] = profile["merch_lon"]
    st.session_state["amt"] = profile["typical_amount"]


def sync_from_category() -> None:
    st.session_state["amt"] = CATEGORY_DEFAULTS.get(st.session_state["category"], st.session_state["amt"])
    for merchant, profile in MERCHANT_PROFILES.items():
        if profile["category"] == st.session_state["category"]:
            st.session_state["merchant"] = merchant
            st.session_state["merchant_city"] = profile["merchant_city"]
            st.session_state["merch_lat"] = profile["merch_lat"]
            st.session_state["merch_lon"] = profile["merch_lon"]
            break


def sync_from_customer_profile() -> None:
    profile = CUSTOMER_PROFILES[st.session_state["customer_profile"]]
    st.session_state["customer_city"] = profile["customer_city"]
    st.session_state["gender"] = profile["gender"]
    st.session_state["card_segment"] = profile["card_segment"]
    st.session_state["device_trust"] = profile["device_trust"]
    st.session_state["velocity_24h"] = profile["velocity_24h"]
    st.session_state["international_txn"] = profile["international_txn"]
    st.session_state["card_number"] = profile["card_number"]
    st.session_state["card_mode"] = profile["card_mode"]
    sync_customer_city()


def sync_customer_city() -> None:
    lat, lon = CUSTOMER_CITIES[st.session_state["customer_city"]]
    st.session_state["lat"] = lat
    st.session_state["lon"] = lon


def card_mode_from_number(card_number: str) -> str:
    digits = "".join(ch for ch in str(card_number) if ch.isdigit())
    if not digits:
        return st.session_state.get("card_mode", "Credit Card")
    if digits.startswith("6"):
        return "Debit Card"
    return "Credit Card"


def sync_from_card_number() -> None:
    st.session_state["card_mode"] = card_mode_from_number(st.session_state["card_number"])


def sync_live_intelligence() -> None:
    sync_from_card_number()
    if st.session_state["category"] in CATEGORY_DEFAULTS and st.session_state["amt"] == 0:
        st.session_state["amt"] = CATEGORY_DEFAULTS[st.session_state["category"]]


def feature_frame() -> pd.DataFrame:
    distance = calculate_distance(
        st.session_state["lat"],
        st.session_state["lon"],
        st.session_state["merch_lat"],
        st.session_state["merch_lon"],
    )
    frame = pd.DataFrame(
        [
            {
                "merchant": merchant_map[st.session_state["merchant"]],
                "category": category_map[st.session_state["category"]],
                "amt": float(st.session_state["amt"]),
                "cc_num": int("".join(ch for ch in str(st.session_state["card_number"]) if ch.isdigit()) or 0),
                "hour": int(st.session_state["hour"]),
                "day": int(st.session_state["day"]),
                "month": int(st.session_state["month"]),
                "gender": gender_map[st.session_state["gender"]],
                "distance": distance,
            }
        ]
    )
    return frame


def compute_risk() -> dict:
    data = feature_frame()
    distance = float(data.loc[0, "distance"])
    prediction = int(model.predict(data)[0])
    model_probability = (
        float(model.predict_proba(data)[0][1]) if hasattr(model, "predict_proba") else float(prediction)
    )

    risk_probability = model_probability
    reasons = []
    expected_amount = CATEGORY_DEFAULTS.get(st.session_state["category"], st.session_state["amt"] or 1)
    amount_ratio = float(st.session_state["amt"]) / max(expected_amount, 1)
    digits = "".join(ch for ch in str(st.session_state["card_number"]) if ch.isdigit())

    if st.session_state["amt"] > 10000:
        risk_probability += 0.08
        reasons.append("Amount is unusually high for normal retail activity.")
    if amount_ratio >= 2.2:
        risk_probability += 0.06
        reasons.append("Amount is far above the normal category spending baseline.")
    if distance > 120:
        risk_probability += 0.10
        reasons.append("Merchant and customer locations are far apart.")
    if st.session_state["hour"] < 5 or st.session_state["hour"] >= 23:
        risk_probability += 0.06
        reasons.append("Transaction time falls in a high-risk late-night window.")
    if st.session_state["velocity_24h"] >= 6:
        risk_probability += 0.09
        reasons.append("Card activity volume is high within the last 24 hours.")
    if st.session_state["device_trust"] < 40:
        risk_probability += 0.08
        reasons.append("Device trust score is weak and suggests anomaly.")
    if st.session_state["international_txn"]:
        risk_probability += 0.07
        reasons.append("Cross-border activity raises verification requirements.")
    if st.session_state["card_mode"] == "Debit Card":
        risk_probability += 0.03
        reasons.append("Debit card transactions are checked with stricter protection rules.")
    if len(digits) not in {15, 16}:
        risk_probability += 0.05
        reasons.append("Card number pattern looks incomplete or unusual.")
    if st.session_state["category"] in {"shopping_net", "misc_net"} and st.session_state["device_trust"] < 55:
        risk_probability += 0.05
        reasons.append("Online transaction from a weak-trust device needs extra review.")
    if st.session_state["day"] in {1, 28, 29, 30, 31} and st.session_state["amt"] > 8000:
        risk_probability += 0.03
        reasons.append("High-value end-of-cycle spending pattern is being monitored closely.")
    if not reasons:
        reasons.append("Behavior is close to known normal transaction patterns.")

    risk_probability = max(0.01, min(risk_probability, 0.99))
    final_prediction = 1 if risk_probability >= 0.50 or prediction == 1 else 0
    risk_percent = round(risk_probability * 100, 2)

    if risk_percent >= 75:
        severity = "Critical"
        recommendation = "Block immediately and require customer verification."
    elif risk_percent >= 50:
        severity = "High"
        recommendation = "Step-up authentication is recommended before approval."
    elif risk_percent >= 30:
        severity = "Moderate"
        recommendation = "Allow with monitoring and post-transaction review."
    else:
        severity = "Low"
        recommendation = "Transaction may proceed with routine monitoring."

    return {
        "input": data,
        "distance": distance,
        "model_probability": round(model_probability * 100, 2),
        "risk_percent": risk_percent,
        "prediction": final_prediction,
        "severity": severity,
        "recommendation": recommendation,
        "reasons": reasons,
    }


def build_sample_batch() -> pd.DataFrame:
    rows = [
        {
            "merchant": "fraud_Rippin, Kub and Mann",
            "category": "grocery_pos",
            "amt": 2300,
            "card_number": "4539682995824395",
            "card_mode": "Credit Card",
            "gender": "M",
            "lat": 28.6139,
            "lon": 77.2090,
            "merch_lat": 28.7041,
            "merch_lon": 77.1025,
            "hour": 14,
            "day": 15,
            "month": 4,
            "device_trust": 82,
            "velocity_24h": 2,
            "international_txn": False,
        },
        {
            "merchant": "fraud_Kutch, Hermiston and Farrell",
            "category": "shopping_net",
            "amt": 18900,
            "card_number": "6011222233334444",
            "card_mode": "Debit Card",
            "gender": "F",
            "lat": 19.0760,
            "lon": 72.8777,
            "merch_lat": 12.9716,
            "merch_lon": 77.5946,
            "hour": 1,
            "day": 19,
            "month": 4,
            "device_trust": 24,
            "velocity_24h": 8,
            "international_txn": True,
        },
        {
            "merchant": "fraud_Keeling-Crist",
            "category": "misc_net",
            "amt": 5900,
            "card_number": "5123456789012345",
            "card_mode": "Credit Card",
            "gender": "M",
            "lat": 17.3850,
            "lon": 78.4867,
            "merch_lat": 17.3850,
            "merch_lon": 78.4867,
            "hour": 11,
            "day": 18,
            "month": 4,
            "device_trust": 68,
            "velocity_24h": 4,
            "international_txn": False,
        },
    ]
    return pd.DataFrame(rows)


def score_batch(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "merchant",
        "category",
        "amt",
        "card_number",
        "card_mode",
        "gender",
        "lat",
        "lon",
        "merch_lat",
        "merch_lon",
        "hour",
        "day",
        "month",
        "device_trust",
        "velocity_24h",
        "international_txn",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    results = []
    for _, row in df.iterrows():
        distance = calculate_distance(row["lat"], row["lon"], row["merch_lat"], row["merch_lon"])
        digits = "".join(ch for ch in str(row["card_number"]) if ch.isdigit())
        expected_amount = CATEGORY_DEFAULTS.get(row["category"], float(row["amt"]) or 1)
        amount_ratio = float(row["amt"]) / max(expected_amount, 1)
        model_input = pd.DataFrame(
            [
                {
                    "merchant": merchant_map[row["merchant"]],
                    "category": category_map[row["category"]],
                    "amt": float(row["amt"]),
                    "cc_num": int("".join(ch for ch in str(row["card_number"]) if ch.isdigit()) or 0),
                    "hour": int(row["hour"]),
                    "day": int(row["day"]),
                    "month": int(row["month"]),
                    "gender": gender_map[row["gender"]],
                    "distance": distance,
                }
            ]
        )
        base_probability = (
            float(model.predict_proba(model_input)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(model_input)[0])
        )
        adjusted = base_probability
        if float(row["amt"]) > 10000:
            adjusted += 0.08
        if amount_ratio >= 2.2:
            adjusted += 0.06
        if distance > 120:
            adjusted += 0.10
        if int(row["velocity_24h"]) >= 6:
            adjusted += 0.09
        if int(row["device_trust"]) < 40:
            adjusted += 0.08
        if bool(row["international_txn"]):
            adjusted += 0.07
        if row["card_mode"] == "Debit Card":
            adjusted += 0.03
        if len(digits) not in {15, 16}:
            adjusted += 0.05
        if row["category"] in {"shopping_net", "misc_net"} and int(row["device_trust"]) < 55:
            adjusted += 0.05
        adjusted = max(0.01, min(adjusted, 0.99))

        results.append(
            {
                "card_masked": mask_card(str(row["card_number"])),
                "card_mode": row["card_mode"],
                "merchant": row["merchant"],
                "amount": float(row["amt"]),
                "distance_km": round(distance, 2),
                "risk_score": round(adjusted * 100, 2),
                "status": "Fraud" if adjusted >= 0.50 else "Safe",
            }
        )
    return pd.DataFrame(results)


inject_css()
initialize_state()


st.markdown(
    """
    <div class="hero-shell">
        <div class="eyebrow">Unified Fraud Intelligence</div>
        <div class="hero-title">FraudShield 360</div>
        <div class="hero-copy">
            A professional banking-style screening dashboard for credit and debit card transactions.
            It combines your trained fraud model with extra rule-based intelligence, automatic form filling,
            batch screening, and analyst-friendly dashboards for project demo, viva, and presentation.
        </div>
        <div class="hero-strip">
            <div class="hero-pill">Live card screening with auto-intake</div>
            <div class="hero-pill">Credit + debit card fraud monitoring</div>
            <div class="hero-pill">Hybrid ML + behavioral risk scoring</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


sidebar_page = st.sidebar.radio(
    "Workspace",
    ["Executive Overview", "Live Detection", "Batch Screening", "Risk Intelligence", "Project Overview"],
)

st.sidebar.markdown("### Smart Auto Fill")
st.sidebar.caption(
    "Merchant, category, customer city, and card number now auto-sync related details so one input can fill the rest."
)
st.sidebar.markdown("### Supported Cards")
st.sidebar.success("Credit Card screening")
st.sidebar.success("Debit Card screening")
st.sidebar.info("Debit transactions use the same ML core with stricter protection rules layered on top.")


if sidebar_page == "Executive Overview":
    risk_snapshot = compute_risk()
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        f"<div class='kpi-card'><h4>Current Risk</h4><h2>{risk_snapshot['risk_percent']}%</h2><p>{risk_snapshot['severity']} alert level</p></div>",
        unsafe_allow_html=True,
    )
    k2.markdown(
        f"<div class='kpi-card'><h4>Model Score</h4><h2>{risk_snapshot['model_probability']}%</h2><p>Pure ML confidence</p></div>",
        unsafe_allow_html=True,
    )
    k3.markdown(
        f"<div class='kpi-card'><h4>Card Type</h4><h2>{st.session_state['card_mode'].split()[0]}</h2><p>{infer_network(st.session_state['card_number'])}</p></div>",
        unsafe_allow_html=True,
    )
    k4.markdown(
        f"<div class='kpi-card'><h4>Distance</h4><h2>{risk_snapshot['distance']:.1f} km</h2><p>Customer to merchant</p></div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.5, 1])
    with left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Command Center")
        history = pd.DataFrame(
            {
                "Window": ["08:00-10:00", "10:00-12:00", "12:00-14:00", "14:00-16:00", "16:00-18:00"],
                "Screened Txns": [124, 142, 156, 138, 149],
                "Fraud Alerts": [6, 8, 5, 7, 9],
            }
        )
        st.area_chart(history.set_index("Window"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Recent Analyst Notes")
        st.write("• High-value online shopping patterns remain the strongest fraud trigger.")
        st.write("• Debit cards are currently scored with extra caution for low-trust devices.")
        st.write("• Late-night cross-city transactions continue to show elevated risk behavior.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        render_card_visual(
            "Protected Card",
            st.session_state["card_number"],
            st.session_state["card_mode"],
            risk_snapshot["risk_percent"],
            st.session_state["card_segment"],
        )
        st.markdown("<div class='alert-card'>", unsafe_allow_html=True)
        st.subheader("Recommended Action")
        st.write(risk_snapshot["recommendation"])
        st.write(f"Prediction result: {'Fraud detected' if risk_snapshot['prediction'] else 'Safe transaction'}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Top Triggers")
        for reason in risk_snapshot["reasons"]:
            st.write(f"• {reason}")
        st.markdown("</div>", unsafe_allow_html=True)

elif sidebar_page == "Live Detection":
    st.subheader("Live Transaction Detection")
    tabs = st.tabs(["Smart Intake", "Fraud Decision", "Analyst View", "Model Payload"])

    with tabs[0]:
        c1, c2 = st.columns([1.05, 0.95])
        with c1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### Auto-Populating Transaction Intake")
            st.selectbox(
                "Customer Profile",
                list(CUSTOMER_PROFILES.keys()),
                key="customer_profile",
                on_change=sync_from_customer_profile,
            )
            st.selectbox(
                "Merchant",
                list(MERCHANT_PROFILES.keys()),
                key="merchant",
                on_change=sync_from_merchant,
            )
            st.selectbox(
                "Category",
                list(category_map.keys()),
                key="category",
                on_change=sync_from_category,
            )
            st.selectbox(
                "Customer City",
                list(CUSTOMER_CITIES.keys()),
                key="customer_city",
                on_change=sync_customer_city,
            )
            st.text_input(
                "Card Number",
                key="card_number",
                on_change=sync_from_card_number,
                help="Card number automatically suggests bank, network, and credit/debit mode from the prefix.",
            )
            st.selectbox("Card Mode", ["Credit Card", "Debit Card"], key="card_mode")
            st.selectbox("Customer Gender", ["M", "F"], key="gender")
            st.selectbox("Card Segment", list(CARD_SEGMENTS.keys()), key="card_segment")
            quick1, quick2, quick3 = st.columns(3)
            if quick1.button("Retail Preset", use_container_width=True):
                st.session_state["category"] = "grocery_pos"
                sync_from_category()
            if quick2.button("E-Commerce Preset", use_container_width=True):
                st.session_state["category"] = "shopping_net"
                sync_from_category()
            if quick3.button("Travel Preset", use_container_width=True):
                st.session_state["category"] = "travel"
                st.session_state["amt"] = CATEGORY_DEFAULTS["travel"]
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### Dynamic Fraud Signals")
            st.number_input("Transaction Amount", min_value=0.0, step=100.0, key="amt")
            st.number_input("Customer Latitude", format="%.6f", key="lat")
            st.number_input("Customer Longitude", format="%.6f", key="lon")
            st.number_input("Merchant Latitude", format="%.6f", key="merch_lat")
            st.number_input("Merchant Longitude", format="%.6f", key="merch_lon")
            st.slider("Transaction Hour", 0, 23, key="hour")
            st.slider("Transaction Day", 1, 31, key="day")
            st.slider("Transaction Month", 1, 12, key="month")
            st.slider("Device Trust Score", 0, 100, key="device_trust")
            st.slider("Transaction Count in Last 24h", 0, 12, key="velocity_24h")
            st.checkbox("International Transaction", key="international_txn")
            st.markdown("</div>", unsafe_allow_html=True)

        risk = compute_risk()
        a1, a2 = st.columns([1, 1])
        with a1:
            render_card_visual(
                "Primary Card",
                st.session_state["card_number"],
                st.session_state["card_mode"],
                risk["risk_percent"],
                st.session_state["card_segment"],
            )
        with a2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Live Intake Summary")
            st.write(f"Customer city: {st.session_state['customer_city']}")
            st.write(f"Merchant city: {st.session_state['merchant_city']}")
            st.write(f"Bank: {infer_bank(st.session_state['card_number'])}")
            st.write(f"Network: {infer_network(st.session_state['card_number'])}")
            st.write(f"Card mode: {st.session_state['card_mode']}")
            st.write(f"Current risk: {risk['risk_percent']}%")
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        risk = compute_risk()
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Fraud Decision Engine")
            if risk["prediction"]:
                st.error(f"Fraud detected with {risk['risk_percent']}% risk")
            else:
                st.success(f"Safe transaction with {risk['risk_percent']}% risk")
            st.progress(min(math.floor(risk["risk_percent"]), 100))
            st.write(f"Severity: {risk['severity']}")
            st.write(f"Recommendation: {risk['recommendation']}")
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Card Intelligence")
            st.metric("Masked Card", mask_card(st.session_state["card_number"]))
            st.metric("Network", infer_network(st.session_state["card_number"]))
            st.metric("Mode", st.session_state["card_mode"])
            st.metric("Segment", st.session_state["card_segment"])
            st.caption(CARD_SEGMENTS[st.session_state["card_segment"]])
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Risk Reasons")
        for item in risk["reasons"]:
            st.write(f"• {item}")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        risk = compute_risk()
        left, right = st.columns([1.15, 0.85])
        with left:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Analyst Investigation Panel")
            signal_rows = [
                ("Amount deviation vs category", f"{(float(st.session_state['amt']) / max(CATEGORY_DEFAULTS.get(st.session_state['category'], 1), 1)):.2f}x"),
                ("Device trust score", st.session_state["device_trust"]),
                ("24h velocity", st.session_state["velocity_24h"]),
                ("Geo distance", f"{risk['distance']:.2f} km"),
                ("Card-network identified", infer_network(st.session_state["card_number"])),
            ]
            for label, value in signal_rows:
                st.markdown(
                    f"<div class='signal-row'><span>{label}</span><strong>{value}</strong></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            render_card_visual(
                "Analyst Snapshot",
                st.session_state["card_number"],
                st.session_state["card_mode"],
                risk["risk_percent"],
                st.session_state["card_segment"],
            )

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Action Workflow")
            st.write("• Step 1: validate device and geo consistency")
            st.write("• Step 2: confirm whether amount matches customer pattern")
            st.write("• Step 3: trigger OTP or account hold for high-risk alerts")
            st.write("• Step 4: escalate to fraud analyst if risk stays above 50%")
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        risk = compute_risk()
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Model Input Payload")
        st.dataframe(risk["input"], use_container_width=True)
        st.caption(
            "The trained ML model still uses the original project features. Credit/debit handling and extra intelligence are layered on top in the risk engine."
        )
        st.markdown("</div>", unsafe_allow_html=True)

elif sidebar_page == "Batch Screening":
    st.subheader("Batch Transaction Screening")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write(
        "Upload a CSV with supported transaction columns or use the sample batch below to demonstrate multiple fraud checks at once."
    )
    sample_batch = build_sample_batch()
    st.dataframe(sample_batch, use_container_width=True)
    scored = score_batch(sample_batch)
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Scored Results")
        st.dataframe(scored, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Batch Summary")
        st.metric("Transactions", len(scored))
        st.metric("Fraud Alerts", int((scored["status"] == "Fraud").sum()))
        st.metric("Avg Risk", f"{scored['risk_score'].mean():.2f}%")
        st.metric("Highest Risk", f"{scored['risk_score'].max():.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            uploaded_scored = score_batch(uploaded_df)
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Uploaded File Results")
            st.dataframe(uploaded_scored, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Unable to score uploaded file: {exc}")

elif sidebar_page == "Risk Intelligence":
    st.subheader("Risk Intelligence Dashboard")
    charts_tab, signals_tab, rules_tab = st.tabs(["Analytics", "Signals", "Rules"])

    with charts_tab:
        chart_df = pd.DataFrame(
            {
                "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                "Fraud Alerts": [14, 11, 17, 12, 16, 23, 20],
                "Debit Alerts": [6, 5, 8, 4, 7, 10, 9],
                "Credit Alerts": [8, 6, 9, 8, 9, 13, 11],
            }
        )
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.line_chart(chart_df.set_index("Day"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with signals_tab:
        signal_df = pd.DataFrame(
            {
                "Signal": ["High Amount", "Long Distance", "Low Device Trust", "Velocity Spike", "International Use"],
                "Impact Score": [89, 81, 84, 78, 73],
            }
        )
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.bar_chart(signal_df.set_index("Signal"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with rules_tab:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Hybrid Detection Rules")
        st.write("• ML model predicts the base probability using your original fraud features.")
        st.write("• Debit card mode adds stronger safeguards for risky environments.")
        st.write("• Device trust, velocity, and international activity increase risk in the advanced engine.")
        st.write("• Auto-fill helps turn one selected detail into a full realistic demo transaction.")
        st.markdown("</div>", unsafe_allow_html=True)

elif sidebar_page == "Project Overview":
    st.subheader("Project Overview")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        ### Major Project Title
        **FraudShield 360: Advanced Credit and Debit Card Fraud Detection Platform**

        ### What's New in This Advanced Version
        - Professional banking-style UI with multi-section workspace
        - Live fraud detection for both credit and debit cards
        - Smart auto-fill from merchant, category, city, and card number
        - Batch screening dashboard for demoing multiple transactions
        - Hybrid risk engine combining ML prediction with operational fraud rules
        - Analyst-facing command center and risk intelligence views

        ### Important Note
        The current trained model file still uses the original project features:
        merchant, category, amount, card number, time, gender, and distance.
        The debit-specific intelligence is added as an advanced rules layer on top of the model.
        For a true debit-vs-credit learned model, the notebook should be retrained with card-type features.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    "<div class='footer'>FraudShield 360 • Advanced academic project interface for fraud monitoring and transaction intelligence</div>",
    unsafe_allow_html=True,
)
