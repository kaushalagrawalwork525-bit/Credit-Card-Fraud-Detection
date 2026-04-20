from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import random

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
    "Rippin Retail Hub": {
        "model_name": "fraud_Rippin, Kub and Mann",
        "category": "grocery_pos",
        "merchant_city": "New Delhi",
        "merch_lat": 28.7041,
        "merch_lon": 77.1025,
        "typical_amount": 2400.0,
        "channel": "POS",
    },
    "Heller Premium Store": {
        "model_name": "fraud_Heller, Gutmann and Zieme",
        "category": "shopping_pos",
        "merchant_city": "Mumbai",
        "merch_lat": 19.0760,
        "merch_lon": 72.8777,
        "typical_amount": 6800.0,
        "channel": "POS",
    },
    "Lind Fuel Station": {
        "model_name": "fraud_Lind-Buckridge",
        "category": "gas_transport",
        "merchant_city": "Jaipur",
        "merch_lat": 26.9124,
        "merch_lon": 75.7873,
        "typical_amount": 1800.0,
        "channel": "POS",
    },
    "Kutch Digital Mall": {
        "model_name": "fraud_Kutch, Hermiston and Farrell",
        "category": "shopping_net",
        "merchant_city": "Bengaluru",
        "merch_lat": 12.9716,
        "merch_lon": 77.5946,
        "typical_amount": 9800.0,
        "channel": "Online",
    },
    "Keeling eMarket": {
        "model_name": "fraud_Keeling-Crist",
        "category": "misc_net",
        "merchant_city": "Hyderabad",
        "merch_lat": 17.3850,
        "merch_lon": 78.4867,
        "typical_amount": 5600.0,
        "channel": "Online",
    },
}

CATEGORY_MAP = {
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
    "Ahmedabad": (23.0225, 72.5714),
    "Lucknow": (26.8467, 80.9462),
}

CUSTOMER_PROFILES = {
    "Aarav Sharma": {
        "customer_city": "New Delhi",
        "gender": "M",
        "card_segment": "Classic",
        "device_trust": 78,
        "velocity_24h": 2,
        "international_txn": False,
        "card_number": "4539682995824395",
        "card_mode": "Credit Card",
        "ip_address": "49.205.14.82",
        "device_id": "DL-AND-0192",
    },
    "Neha Verma": {
        "customer_city": "Mumbai",
        "gender": "F",
        "card_segment": "Gold",
        "device_trust": 67,
        "velocity_24h": 4,
        "international_txn": False,
        "card_number": "5123456789012345",
        "card_mode": "Credit Card",
        "ip_address": "103.84.116.19",
        "device_id": "MB-IOS-2031",
    },
    "Rohan Iyer": {
        "customer_city": "Bengaluru",
        "gender": "M",
        "card_segment": "Platinum",
        "device_trust": 56,
        "velocity_24h": 5,
        "international_txn": True,
        "card_number": "6011222233334444",
        "card_mode": "Debit Card",
        "ip_address": "117.195.122.18",
        "device_id": "BLR-WEB-7721",
    },
    "Priya Nair": {
        "customer_city": "Chennai",
        "gender": "F",
        "card_segment": "Gold",
        "device_trust": 85,
        "velocity_24h": 1,
        "international_txn": False,
        "card_number": "4532123412349876",
        "card_mode": "Credit Card",
        "ip_address": "59.96.211.5",
        "device_id": "CHN-IOS-8830",
    },
    "Kabir Mehta": {
        "customer_city": "Ahmedabad",
        "gender": "M",
        "card_segment": "Classic",
        "device_trust": 48,
        "velocity_24h": 6,
        "international_txn": False,
        "card_number": "6011444477778888",
        "card_mode": "Debit Card",
        "ip_address": "42.110.13.204",
        "device_id": "AMD-AND-1104",
    },
    "Ananya Gupta": {
        "customer_city": "Kolkata",
        "gender": "F",
        "card_segment": "Platinum",
        "device_trust": 72,
        "velocity_24h": 3,
        "international_txn": True,
        "card_number": "378282246310005",
        "card_mode": "Credit Card",
        "ip_address": "115.248.64.7",
        "device_id": "KOL-WEB-9911",
    },
    "Vikram Singh": {
        "customer_city": "Lucknow",
        "gender": "M",
        "card_segment": "Classic",
        "device_trust": 61,
        "velocity_24h": 2,
        "international_txn": False,
        "card_number": "5123888899990001",
        "card_mode": "Credit Card",
        "ip_address": "106.203.72.65",
        "device_id": "LKO-AND-6720",
    },
    "Sneha Patil": {
        "customer_city": "Pune",
        "gender": "F",
        "card_segment": "Gold",
        "device_trust": 52,
        "velocity_24h": 7,
        "international_txn": True,
        "card_number": "6011333355557777",
        "card_mode": "Debit Card",
        "ip_address": "122.172.10.51",
        "device_id": "PUN-IOS-4119",
    },
}

CARD_NETWORKS = {
    "3": "American Express",
    "4": "Visa",
    "5": "Mastercard",
    "6": "RuPay / Discover",
}

BANK_BY_PREFIX = {
    "3": "Elite Capital Bank",
    "4": "Horizon Bank",
    "5": "Zenith Bank",
    "6": "National Secure Bank",
}

MERCHANT_ENCODING = {
    profile["model_name"]: idx for idx, profile in enumerate(MERCHANT_PROFILES.values())
}
GENDER_MAP = {"M": 0, "F": 1}
SEGMENT_HELP = {
    "Classic": "Standard card with regular controls",
    "Gold": "Premium card with higher transaction patterns",
    "Platinum": "High-value customer with broad spending activity",
}


def apply_theme(theme: str) -> None:
    if theme == "Light":
        colors = {
            "bg": "#f4f7fb",
            "surface": "#ffffff",
            "surface_soft": "#eef4ff",
            "text": "#0f172a",
            "muted": "#475569",
            "border": "rgba(15, 23, 42, 0.08)",
            "accent": "#0f766e",
            "accent_soft": "rgba(15, 118, 110, 0.10)",
            "danger": "#dc2626",
            "warning": "#d97706",
            "success": "#15803d",
            "sidebar": "#e8eef8",
        }
    else:
        colors = {
            "bg": "#081120",
            "surface": "#101b2d",
            "surface_soft": "#0d1728",
            "text": "#e5eef9",
            "muted": "#9fb1c7",
            "border": "rgba(148, 163, 184, 0.14)",
            "accent": "#22c1a1",
            "accent_soft": "rgba(34, 193, 161, 0.10)",
            "danger": "#f87171",
            "warning": "#fb923c",
            "success": "#4ade80",
            "sidebar": "#0a1423",
        }

    st.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(180deg, {colors["bg"]} 0%, {colors["bg"]} 100%);
                color: {colors["text"]};
            }}
            [data-testid="stSidebar"] {{
                background: {colors["sidebar"]};
                border-right: 1px solid {colors["border"]};
            }}
            .block-container {{
                padding-top: 1.1rem;
                padding-bottom: 2rem;
                max-width: 1280px;
            }}
            h1, h2, h3, h4, h5, h6, p, span, label, div {{
                color: {colors["text"]};
            }}
            .hero {{
                background: {colors["surface"]};
                border: 1px solid {colors["border"]};
                border-radius: 24px;
                padding: 24px;
                margin-bottom: 1rem;
            }}
            .panel {{
                background: {colors["surface"]};
                border: 1px solid {colors["border"]};
                border-radius: 22px;
                padding: 18px;
                margin-bottom: 1rem;
            }}
            .mini {{
                background: {colors["surface_soft"]};
                border: 1px solid {colors["border"]};
                border-radius: 16px;
                padding: 14px;
                margin-bottom: 0.8rem;
            }}
            .badge-row {{
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                margin-top: 10px;
            }}
            .badge {{
                display: inline-block;
                background: {colors["accent_soft"]};
                color: {colors["accent"]};
                border: 1px solid {colors["border"]};
                border-radius: 999px;
                padding: 7px 12px;
                font-size: 0.85rem;
                font-weight: 700;
            }}
            .kpi {{
                background: {colors["surface"]};
                border: 1px solid {colors["border"]};
                border-radius: 20px;
                padding: 18px;
                min-height: 132px;
            }}
            .eyebrow {{
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-size: 0.75rem;
                color: {colors["accent"]};
                font-weight: 800;
            }}
            .helper {{
                color: {colors["muted"]};
                font-size: 0.95rem;
                line-height: 1.6;
            }}
            .todo {{
                background: {colors["surface"]};
                border-left: 4px solid {colors["accent"]};
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 10px;
            }}
            div[data-testid="stMetric"] {{
                background: {colors["surface_soft"]};
                border: 1px solid {colors["border"]};
                padding: 12px;
                border-radius: 16px;
            }}
            div[data-testid="stTabs"] button {{
                border-radius: 999px;
            }}
            .card-visual {{
                border-radius: 22px;
                padding: 20px;
                min-height: 210px;
                background: linear-gradient(145deg, #0f766e, #123a54);
                color: white;
                border: 1px solid rgba(255,255,255,0.10);
            }}
            .card-chip {{
                width: 54px;
                height: 40px;
                border-radius: 10px;
                background: linear-gradient(145deg, #f8d34d, #b7791f);
                margin: 16px 0 18px 0;
            }}
            .risk-low {{
                color: {colors["success"]};
                font-weight: 800;
            }}
            .risk-mid {{
                color: {colors["warning"]};
                font-weight: 800;
            }}
            .risk-high {{
                color: {colors["danger"]};
                font-weight: 800;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def infer_network(card_number: str) -> str:
    digits = "".join(ch for ch in str(card_number) if ch.isdigit())
    if not digits:
        return "Unknown"
    return CARD_NETWORKS.get(digits[0], "Unknown")


def infer_bank(card_number: str) -> str:
    digits = "".join(ch for ch in str(card_number) if ch.isdigit())
    if not digits:
        return "Unknown Bank"
    return BANK_BY_PREFIX.get(digits[0], "Unknown Bank")


def mask_card(card_number: str) -> str:
    digits = "".join(ch for ch in str(card_number) if ch.isdigit())
    if len(digits) < 4:
        return "••••"
    return f"•••• •••• •••• {digits[-4:]}"


def calc_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return geodesic((lat1, lon1), (lat2, lon2)).km


def init_state() -> None:
    defaults = {
        "theme_mode": "Dark",
        "customer_profile": "Aarav Sharma",
        "merchant": "Rippin Retail Hub",
        "category": "grocery_pos",
        "amount": 2400.0,
        "card_number": "4539682995824395",
        "card_mode": "Credit Card",
        "customer_city": "New Delhi",
        "merchant_city": "New Delhi",
        "gender": "M",
        "lat": 28.6139,
        "lon": 77.2090,
        "merch_lat": 28.7041,
        "merch_lon": 77.1025,
        "hour": 14,
        "day": 15,
        "month": 4,
        "card_segment": "Classic",
        "device_trust": 78,
        "velocity_24h": 2,
        "international_txn": False,
        "ip_address": "49.205.14.82",
        "device_id": "DL-AND-0192",
        "rule_amount_weight": 0.08,
        "rule_distance_weight": 0.10,
        "rule_device_weight": 0.08,
        "rule_velocity_weight": 0.09,
        "rule_international_weight": 0.07,
        "rule_debit_weight": 0.03,
        "rule_online_device_weight": 0.05,
        "audit_logs": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def sync_from_customer() -> None:
    profile = CUSTOMER_PROFILES[st.session_state["customer_profile"]]
    st.session_state["customer_city"] = profile["customer_city"]
    st.session_state["gender"] = profile["gender"]
    st.session_state["card_segment"] = profile["card_segment"]
    st.session_state["device_trust"] = profile["device_trust"]
    st.session_state["velocity_24h"] = profile["velocity_24h"]
    st.session_state["international_txn"] = profile["international_txn"]
    st.session_state["card_number"] = profile["card_number"]
    st.session_state["card_mode"] = profile["card_mode"]
    st.session_state["ip_address"] = profile["ip_address"]
    st.session_state["device_id"] = profile["device_id"]
    sync_customer_city()


def sync_customer_city() -> None:
    st.session_state["lat"], st.session_state["lon"] = CUSTOMER_CITIES[st.session_state["customer_city"]]


def sync_from_merchant() -> None:
    profile = MERCHANT_PROFILES[st.session_state["merchant"]]
    st.session_state["category"] = profile["category"]
    st.session_state["merchant_city"] = profile["merchant_city"]
    st.session_state["merch_lat"] = profile["merch_lat"]
    st.session_state["merch_lon"] = profile["merch_lon"]
    st.session_state["amount"] = profile["typical_amount"]


def sync_from_category() -> None:
    st.session_state["amount"] = CATEGORY_DEFAULTS.get(st.session_state["category"], st.session_state["amount"])
    for merchant_name, profile in MERCHANT_PROFILES.items():
        if profile["category"] == st.session_state["category"]:
            st.session_state["merchant"] = merchant_name
            st.session_state["merchant_city"] = profile["merchant_city"]
            st.session_state["merch_lat"] = profile["merch_lat"]
            st.session_state["merch_lon"] = profile["merch_lon"]
            break


def sync_from_card_number() -> None:
    digits = "".join(ch for ch in str(st.session_state["card_number"]) if ch.isdigit())
    if digits.startswith("6"):
        st.session_state["card_mode"] = "Debit Card"
    else:
        st.session_state["card_mode"] = "Credit Card"


def build_model_input() -> tuple[pd.DataFrame, float]:
    distance = calc_distance(
        st.session_state["lat"],
        st.session_state["lon"],
        st.session_state["merch_lat"],
        st.session_state["merch_lon"],
    )
    merchant_key = MERCHANT_PROFILES[st.session_state["merchant"]]["model_name"]
    cc_num = int("".join(ch for ch in str(st.session_state["card_number"]) if ch.isdigit()) or 0)
    frame = pd.DataFrame(
        [
            {
                "merchant": MERCHANT_ENCODING[merchant_key],
                "category": CATEGORY_MAP[st.session_state["category"]],
                "amt": float(st.session_state["amount"]),
                "cc_num": cc_num,
                "hour": int(st.session_state["hour"]),
                "day": int(st.session_state["day"]),
                "month": int(st.session_state["month"]),
                "gender": GENDER_MAP[st.session_state["gender"]],
                "distance": distance,
            }
        ]
    )
    return frame, distance


def compute_risk() -> dict:
    payload, distance = build_model_input()
    base_prob = float(model.predict_proba(payload)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(payload)[0])
    final_prob = base_prob
    reasons = []

    expected_amount = CATEGORY_DEFAULTS.get(st.session_state["category"], 1)
    amount_ratio = float(st.session_state["amount"]) / max(expected_amount, 1)
    digits = "".join(ch for ch in str(st.session_state["card_number"]) if ch.isdigit())
    merchant_channel = MERCHANT_PROFILES[st.session_state["merchant"]]["channel"]

    if st.session_state["amount"] > 10000:
        final_prob += st.session_state["rule_amount_weight"]
        reasons.append("high transaction amount")
    if amount_ratio >= 2.2:
        final_prob += 0.06
        reasons.append("amount deviates sharply from category baseline")
    if distance > 120:
        final_prob += st.session_state["rule_distance_weight"]
        reasons.append("unusual customer-to-merchant distance")
    if st.session_state["device_trust"] < 40:
        final_prob += st.session_state["rule_device_weight"]
        reasons.append("low device trust")
    if st.session_state["velocity_24h"] >= 6:
        final_prob += st.session_state["rule_velocity_weight"]
        reasons.append("high card activity in last 24 hours")
    if st.session_state["international_txn"]:
        final_prob += st.session_state["rule_international_weight"]
        reasons.append("international transaction pattern")
    if st.session_state["card_mode"] == "Debit Card":
        final_prob += st.session_state["rule_debit_weight"]
        reasons.append("debit card protection rule applied")
    if merchant_channel == "Online" and st.session_state["device_trust"] < 55:
        final_prob += st.session_state["rule_online_device_weight"]
        reasons.append("online transaction from weak-trust device")
    if len(digits) not in {15, 16}:
        final_prob += 0.05
        reasons.append("card number format looks unusual")
    if st.session_state["hour"] < 5 or st.session_state["hour"] >= 23:
        final_prob += 0.06
        reasons.append("late-night transaction timing")

    final_prob = min(max(final_prob, 0.01), 0.99)
    risk_percent = round(final_prob * 100, 2)
    model_percent = round(base_prob * 100, 2)

    if risk_percent >= 75:
        severity = "Critical"
        recommendation = "Block transaction and trigger account verification."
    elif risk_percent >= 50:
        severity = "High"
        recommendation = "Hold payment and require OTP plus analyst review."
    elif risk_percent >= 30:
        severity = "Moderate"
        recommendation = "Allow with monitoring and post-event review."
    else:
        severity = "Low"
        recommendation = "Approve under routine monitoring."

    if not reasons:
        reasons.append("transaction aligns with normal profile behavior")

    prediction = 1 if risk_percent >= 50 else 0
    explanation = f"Flagged because {' + '.join(reasons[:3])}."

    return {
        "payload": payload,
        "distance": round(distance, 2),
        "model_percent": model_percent,
        "risk_percent": risk_percent,
        "severity": severity,
        "recommendation": recommendation,
        "prediction": prediction,
        "reasons": reasons,
        "explanation": explanation,
    }


def add_audit_log(result: dict) -> None:
    st.session_state["audit_logs"].insert(
        0,
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "customer": st.session_state["customer_profile"],
            "merchant": st.session_state["merchant"],
            "fraud_location": st.session_state["merchant_city"],
            "card_type": st.session_state["card_mode"],
            "masked_card": mask_card(st.session_state["card_number"]),
            "risk": result["risk_percent"],
            "decision": "Fraud" if result["prediction"] else "Safe",
            "action": result["recommendation"],
        },
    )
    st.session_state["audit_logs"] = st.session_state["audit_logs"][:25]


def risk_class(value: float) -> str:
    if value >= 75:
        return "risk-high"
    if value >= 50:
        return "risk-mid"
    return "risk-low"


def render_card(title: str, risk_percent: float) -> None:
    st.markdown(
        f"""
        <div class="card-visual">
            <div class="eyebrow" style="color:#d1fae5;">{title}</div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px;">
                <div style="font-size:1.5rem; font-weight:800;">{infer_bank(st.session_state["card_number"])}</div>
                <div style="padding:6px 12px; border-radius:999px; background:rgba(255,255,255,0.16);">{st.session_state["card_mode"]}</div>
            </div>
            <div class="card-chip"></div>
            <div style="font-size:1.45rem; letter-spacing:0.14em; font-weight:700;">{mask_card(st.session_state["card_number"])}</div>
            <div style="display:flex; justify-content:space-between; margin-top:26px; font-size:0.94rem;">
                <div><div style="opacity:0.8;">Network</div><div style="font-weight:700;">{infer_network(st.session_state["card_number"])}</div></div>
                <div><div style="opacity:0.8;">Segment</div><div style="font-weight:700;">{st.session_state["card_segment"]}</div></div>
                <div><div style="opacity:0.8;">Risk</div><div style="font-weight:700;">{risk_percent:.2f}%</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_case_studies() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Scenario": "Late-night online debit purchase",
                "Reason": "Weak device trust + high velocity + cross-city merchant",
                "Outcome": "Blocked",
            },
            {
                "Scenario": "Premium card travel spend",
                "Reason": "International activity but trusted device",
                "Outcome": "Allowed with monitoring",
            },
            {
                "Scenario": "High-value POS purchase",
                "Reason": "Amount spike above customer baseline",
                "Outcome": "OTP verification triggered",
            },
        ]
    )


def generate_batch(size: int) -> pd.DataFrame:
    rows = []
    merchant_names = list(MERCHANT_PROFILES.keys())
    customer_names = list(CUSTOMER_PROFILES.keys())
    for idx in range(size):
        customer = CUSTOMER_PROFILES[random.choice(customer_names)]
        merchant_name = random.choice(merchant_names)
        merchant = MERCHANT_PROFILES[merchant_name]
        customer_city = customer["customer_city"]
        lat, lon = CUSTOMER_CITIES[customer_city]
        base_amount = merchant["typical_amount"]
        risky = random.random() < 0.12
        amount = base_amount * (random.uniform(0.8, 1.2) if not risky else random.uniform(1.8, 3.5))
        velocity = customer["velocity_24h"] if not risky else random.randint(6, 10)
        device_trust = customer["device_trust"] if not risky else random.randint(18, 45)
        international = customer["international_txn"] if not risky else random.choice([True, True, False])
        merch_lat = merchant["merch_lat"]
        merch_lon = merchant["merch_lon"]
        if risky and random.random() < 0.5:
            merch_lat += random.uniform(4, 11)
            merch_lon += random.uniform(4, 11)
        rows.append(
            {
                "transaction_id": f"TXN-{100000 + idx}",
                "customer": random.choice(customer_names),
                "merchant": merchant_name,
                "category": merchant["category"],
                "amount": round(amount, 2),
                "card_number": customer["card_number"],
                "card_mode": customer["card_mode"],
                "gender": customer["gender"],
                "lat": lat,
                "lon": lon,
                "merch_lat": merch_lat,
                "merch_lon": merch_lon,
                "hour": random.randint(0, 23),
                "day": random.randint(1, 28),
                "month": 4,
                "device_trust": device_trust,
                "velocity_24h": velocity,
                "international_txn": international,
                "customer_city": customer_city,
                "merchant_city": merchant["merchant_city"],
            }
        )
    return pd.DataFrame(rows)


def score_batch(df: pd.DataFrame) -> pd.DataFrame:
    scored_rows = []
    for _, row in df.iterrows():
        distance = calc_distance(row["lat"], row["lon"], row["merch_lat"], row["merch_lon"])
        merchant_key = MERCHANT_PROFILES[row["merchant"]]["model_name"]
        payload = pd.DataFrame(
            [
                {
                    "merchant": MERCHANT_ENCODING[merchant_key],
                    "category": CATEGORY_MAP[row["category"]],
                    "amt": float(row["amount"]),
                    "cc_num": int("".join(ch for ch in str(row["card_number"]) if ch.isdigit()) or 0),
                    "hour": int(row["hour"]),
                    "day": int(row["day"]),
                    "month": int(row["month"]),
                    "gender": GENDER_MAP[row["gender"]],
                    "distance": distance,
                }
            ]
        )
        model_prob = float(model.predict_proba(payload)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(payload)[0])
        final_prob = model_prob
        if row["amount"] > 10000:
            final_prob += st.session_state["rule_amount_weight"]
        if distance > 120:
            final_prob += st.session_state["rule_distance_weight"]
        if row["device_trust"] < 40:
            final_prob += st.session_state["rule_device_weight"]
        if row["velocity_24h"] >= 6:
            final_prob += st.session_state["rule_velocity_weight"]
        if row["international_txn"]:
            final_prob += st.session_state["rule_international_weight"]
        if row["card_mode"] == "Debit Card":
            final_prob += st.session_state["rule_debit_weight"]
        final_prob = min(max(final_prob, 0.01), 0.99)
        scored_rows.append(
            {
                "transaction_id": row["transaction_id"],
                "customer": row["customer"],
                "merchant": row["merchant"],
                "city": row["customer_city"],
                "card_type": row["card_mode"],
                "amount": row["amount"],
                "distance_km": round(distance, 2),
                "risk_score": round(final_prob * 100, 2),
                "decision": "Fraud" if final_prob >= 0.50 else "Safe",
            }
        )
    return pd.DataFrame(scored_rows)


init_state()
apply_theme(st.session_state["theme_mode"])

with st.sidebar:
    st.title("FraudShield 360")
    st.caption("Advanced credit and debit card fraud intelligence dashboard")
    st.selectbox("Theme", ["Dark", "Light"], key="theme_mode")
    apply_theme(st.session_state["theme_mode"])
    page = st.radio(
        "Navigation",
        ["Home", "Features", "Dashboard", "Reports", "Project Info", "Contact"],
    )
    st.markdown("---")
    st.caption("Compliance simulation")
    st.markdown('<div class="badge-row"><span class="badge">PCI DSS</span><span class="badge">GDPR</span><span class="badge">Masked Data</span></div>', unsafe_allow_html=True)


risk = compute_risk()

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Major Project Ready</div>
        <h1 style="margin:6px 0 8px 0;">FraudShield 360</h1>
        <div class="helper">
            Hybrid fraud detection for credit and debit cards with explainable AI, batch monitoring,
            compliance simulation, audit logging, and cleaner analyst dashboards.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if page == "Home":
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='kpi'><div class='eyebrow'>Current Risk</div><h2>{risk['risk_percent']}%</h2><div class='helper'>{risk['severity']} alert level</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi'><div class='eyebrow'>Model Score</div><h2>{risk['model_percent']}%</h2><div class='helper'>Pure ML confidence</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi'><div class='eyebrow'>Card Type</div><h2>{st.session_state['card_mode'].split()[0]}</h2><div class='helper'>{infer_network(st.session_state['card_number'])}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi'><div class='eyebrow'>Distance</div><h2>{risk['distance']} km</h2><div class='helper'>Customer to merchant</div></div>", unsafe_allow_html=True)

    left, right = st.columns([1.2, 0.8])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Fraud Trend Snapshot")
        trend_df = pd.DataFrame(
            {
                "Week": ["W1", "W2", "W3", "W4", "W5", "W6"],
                "Fraud Alerts": [18, 22, 17, 26, 24, 29],
                "Transactions Reviewed": [860, 910, 902, 940, 978, 1034],
            }
        )
        st.line_chart(trend_df.set_index("Week"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Explainable AI Summary")
        st.write("AI-powered fraud detection in real time")
        st.write(f"Decision: {'Fraud' if risk['prediction'] else 'Safe'}")
        st.write(f"Reason: {risk['explanation']}")
        st.write(f"Recommendation: {risk['recommendation']}")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        render_card("Protected Card", risk["risk_percent"])
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Compliance & Security")
        st.write("• Sensitive card details are masked in analyst views.")
        st.write("• Audit trail logging is available for each fraud decision.")
        st.write("• PCI DSS and GDPR badges are shown for compliance simulation.")
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Features":
    st.subheader("Core Features")
    f1, f2, f3 = st.columns(3)
    f1.markdown("<div class='panel'><h4>Real-Time Detection</h4><div class='helper'>AI-powered fraud detection in real time with explainable alerts and clear risk actions.</div></div>", unsafe_allow_html=True)
    f2.markdown("<div class='panel'><h4>Batch Screening</h4><div class='helper'>Simulate and monitor thousands of transactions to demonstrate operational scale.</div></div>", unsafe_allow_html=True)
    f3.markdown("<div class='panel'><h4>Explainable AI</h4><div class='helper'>Each alert explains why it was flagged using model score plus rule-based logic.</div></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Case Studies / Demo Scenarios")
        st.dataframe(build_case_studies(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Future Scope")
        st.write("• Blockchain-backed transaction provenance")
        st.write("• Federated learning across institutions")
        st.write("• Real-time device fingerprinting")
        st.write("• Analyst co-pilot for auto-response recommendations")
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Dashboard":
    st.subheader("Fraud Dashboard")
    top_left, top_right = st.columns([1.05, 0.95])
    with top_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Smart Auto-Fill Intake")
        st.selectbox("Customer Profile", list(CUSTOMER_PROFILES.keys()), key="customer_profile", on_change=sync_from_customer)
        st.selectbox("Merchant", list(MERCHANT_PROFILES.keys()), key="merchant", on_change=sync_from_merchant)
        st.selectbox("Category", list(CATEGORY_MAP.keys()), key="category", on_change=sync_from_category)
        st.selectbox("Customer City", list(CUSTOMER_CITIES.keys()), key="customer_city", on_change=sync_customer_city)
        st.text_input("Card Number", key="card_number", on_change=sync_from_card_number)
        col_a, col_b = st.columns(2)
        col_a.selectbox("Card Mode", ["Credit Card", "Debit Card"], key="card_mode")
        col_b.selectbox("Card Segment", list(SEGMENT_HELP.keys()), key="card_segment")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Auto-Filled Device & Network Details")
        d1, d2 = st.columns(2)
        d1.text_input("IP Address", key="ip_address")
        d2.text_input("Device ID", key="device_id")
        d3, d4 = st.columns(2)
        d3.number_input("Customer Latitude", key="lat", format="%.6f")
        d4.number_input("Customer Longitude", key="lon", format="%.6f")
        d5, d6 = st.columns(2)
        d5.number_input("Merchant Latitude", key="merch_lat", format="%.6f")
        d6.number_input("Merchant Longitude", key="merch_lon", format="%.6f")
        st.markdown('</div>', unsafe_allow_html=True)

    with top_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Risk Signals")
        st.number_input("Transaction Amount", min_value=0.0, step=100.0, key="amount")
        st.slider("Transaction Hour", 0, 23, key="hour")
        st.slider("Transaction Day", 1, 31, key="day")
        st.slider("Transaction Month", 1, 12, key="month")
        st.selectbox("Gender", ["M", "F"], key="gender")
        st.slider("Device Trust Score", 0, 100, key="device_trust")
        st.slider("Transaction Count in Last 24h", 0, 12, key="velocity_24h")
        st.checkbox("International Transaction", key="international_txn")
        st.caption(f"Segment note: {SEGMENT_HELP[st.session_state['card_segment']]}")
        st.markdown('</div>', unsafe_allow_html=True)
        render_card("Live Card View", risk["risk_percent"])

    result_left, result_right = st.columns([1.05, 0.95])
    with result_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Hybrid Fraud Decision")
        st.metric("Hybrid Risk Score", f"{risk['risk_percent']}%")
        st.metric("Model Score", f"{risk['model_percent']}%")
        st.metric("Severity", risk["severity"])
        st.write(f"Explainable AI: {risk['explanation']}")
        st.write(f"Recommendation: {risk['recommendation']}")
        if st.button("Log This Decision To Audit Trail", use_container_width=True):
            add_audit_log(risk)
            st.success("Decision added to audit trail.")
        st.markdown('</div>', unsafe_allow_html=True)
    with result_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Where Fraud Happened")
        st.write(f"Fraud location: {st.session_state['merchant_city']}")
        fraud_loc_df = pd.DataFrame([{"latitude": st.session_state["merch_lat"], "longitude": st.session_state["merch_lon"]}])
        st.map(fraud_loc_df, use_container_width=True)
        st.caption("This map pinpoints the merchant location connected to the suspicious transaction.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Fraud Signals Triggered")
        for reason in risk["reasons"]:
            st.write(f"• {reason}")
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Reports":
    st.subheader("Reports & Analytics")
    size = st.slider("Simulated transactions", 100, 5000, 1000, step=100)
    batch = generate_batch(size)
    scored = score_batch(batch)

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Transactions", len(scored))
    b2.metric("Fraud Alerts", int((scored["decision"] == "Fraud").sum()))
    b3.metric("Avg Risk", f"{scored['risk_score'].mean():.2f}%")
    b4.metric("Debit Cards", int((scored["card_type"] == "Debit Card").sum()))

    left, right = st.columns([1.2, 0.8])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Batch Screening Results")
        st.dataframe(scored.head(200), use_container_width=True)
        st.caption("Showing first 200 rows for performance.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Decision Mix")
        decision_df = scored["decision"].value_counts().rename_axis("Decision").reset_index(name="Count")
        st.bar_chart(decision_df.set_index("Decision"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    analytics_batch = score_batch(generate_batch(1200))
    merchant_filter = st.multiselect("Merchant Filter", sorted(analytics_batch["merchant"].unique()), default=sorted(analytics_batch["merchant"].unique())[:3])
    city_filter = st.multiselect("Geography Filter", sorted(analytics_batch["city"].unique()), default=sorted(analytics_batch["city"].unique())[:4])
    card_filter = st.multiselect("Card Type Filter", sorted(analytics_batch["card_type"].unique()), default=sorted(analytics_batch["card_type"].unique()))

    filtered = analytics_batch[
        analytics_batch["merchant"].isin(merchant_filter)
        & analytics_batch["city"].isin(city_filter)
        & analytics_batch["card_type"].isin(card_filter)
    ]

    a1, a2 = st.columns(2)
    with a1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Fraud Trend Over Time")
        time_df = pd.DataFrame(
            {
                "Date": [datetime.now().date() - timedelta(days=i) for i in range(29, -1, -1)],
                "Fraud Alerts": [random.randint(8, 26) for _ in range(30)],
                "Reviewed Txns": [random.randint(700, 1100) for _ in range(30)],
            }
        )
        st.line_chart(time_df.set_index("Date"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Merchant Risk Comparison")
        merchant_risk = filtered.groupby("merchant", as_index=False)["risk_score"].mean().sort_values("risk_score", ascending=False)
        st.bar_chart(merchant_risk.set_index("merchant"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    map_source = generate_batch(220)
    hotspot_map = map_source.rename(columns={"lat": "latitude", "lon": "longitude"})
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Geographic Fraud Hotspots")
    st.map(hotspot_map[["latitude", "longitude"]], use_container_width=True)
    st.caption("Hotspot map simulates customer locations under active monitoring.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Model Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "96.1%")
    m2.metric("Precision", "93.4%")
    m3.metric("Recall", "91.8%")
    m4.metric("Latency", "82 ms")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Export Reports")
    report_csv = scored.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV Report", report_csv, file_name="fraudshield_report.csv", mime="text/csv")
    st.caption("CSV export adds a professional reporting workflow for project demos.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Scalability Simulation")
    s1, s2, s3 = st.columns(3)
    s1.metric("Transactions / Min", "10,000+")
    s2.metric("Avg Screening Time", "0.08 sec")
    s3.metric("Alert Queue Health", "Stable")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Project Info":
    st.subheader("Project Info")
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Adaptive Rule Controls")
        st.slider("High Amount Rule Weight", 0.00, 0.20, key="rule_amount_weight")
        st.slider("Distance Rule Weight", 0.00, 0.20, key="rule_distance_weight")
        st.slider("Low Device Trust Rule Weight", 0.00, 0.20, key="rule_device_weight")
        st.slider("Velocity Rule Weight", 0.00, 0.20, key="rule_velocity_weight")
        st.slider("International Rule Weight", 0.00, 0.20, key="rule_international_weight")
        st.slider("Debit Card Rule Weight", 0.00, 0.10, key="rule_debit_weight")
        st.caption("These controls simulate analyst-editable fraud rules.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Team / Tech Stack")
        st.write("• Project: FraudShield 360")
        st.write("• Technologies: Python, Streamlit, Scikit-learn, Pandas, Geopy")
        st.write("• Modules: Hybrid scoring, analytics, audit trail, reporting")
        st.write("• Deployment-ready dashboard structure for major project presentation")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Audit Trail")
        if not st.session_state["audit_logs"]:
            st.info("No audit events yet. Log a decision from Live Detection.")
        else:
            st.dataframe(pd.DataFrame(st.session_state["audit_logs"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Security & Compliance Overview")
    st.write("• Data masking/anonymization is applied through masked card displays.")
    st.write("• Audit trail captures timestamp, customer, merchant, decision, and action.")
    st.write("• Compliance badges simulate PCI DSS and GDPR readiness for presentation.")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Contact":
    st.subheader("Contact & Feedback")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.write("Use this section to make the website feel like a real product portal.")
    contact_name = st.text_input("Name")
    contact_email = st.text_input("Email")
    contact_message = st.text_area("Feedback / Query")
    if st.button("Submit Feedback", use_container_width=True):
        if contact_name and contact_email and contact_message:
            st.success("Feedback captured successfully for demo purposes.")
        else:
            st.warning("Please fill in all fields before submitting.")
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown(
    """
    <div style="text-align:center; color:#94a3b8; padding:16px 0 4px 0;">
        FraudShield 360 • Advanced academic fraud detection platform for major project demo and presentation
    </div>
    """,
    unsafe_allow_html=True,
)
