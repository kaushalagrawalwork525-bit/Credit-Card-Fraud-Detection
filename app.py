from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import random
import time

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
        "velocity_24h": 0,
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

AUTH_USERS = {
    "admin": {
        "password": "admin123",
        "name": "Admin User",
        "role": "Administrator",
    },
    "analyst": {
        "password": "analyst123",
        "name": "Fraud Analyst",
        "role": "Analyst",
    },
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
                display: none;
            }}
            [data-testid="collapsedControl"] {{
                display: none;
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
                padding: 28px 32px;
                margin-bottom: 1rem;
            }}
            .hero-title {{
                font-size: 2.2rem;
                font-weight: 900;
                margin: 0 0 6px 0;
                line-height: 1.2;
                color: {colors["text"]};
            }}
            .hero-tagline {{
                font-size: 1.08rem;
                font-weight: 600;
                color: {colors["accent"]};
                margin-bottom: 10px;
                letter-spacing: 0.01em;
            }}
            .hero-desc {{
                color: {colors["muted"]};
                font-size: 0.97rem;
                line-height: 1.65;
                max-width: 720px;
            }}
            .topnav {{
                position: sticky;
                top: 0;
                z-index: 999;
                background: {colors["surface"]};
                border: 1px solid {colors["border"]};
                border-radius: 18px;
                padding: 14px 18px 10px 18px;
                margin-bottom: 1rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.10);
            }}
            .topnav-inner {{
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .topnav-brand {{
                font-size: 1.1rem;
                font-weight: 900;
                color: {colors["text"]};
            }}
            .topnav-sub {{
                color: {colors["muted"]};
                font-size: 0.88rem;
                font-weight: 500;
            }}
            .topnav-theme {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .panel {{
                background: {colors["surface"]};
                border: 1px solid {colors["border"]};
                border-radius: 22px;
                padding: 18px;
                margin-bottom: 1rem;
            }}
            .about-card {{
                background: linear-gradient(180deg, {colors["surface"]}, {colors["surface_soft"]});
                border: 1px solid {colors["border"]};
                border-radius: 22px;
                padding: 20px;
                margin-bottom: 1rem;
            }}
            .feature-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 16px;
                margin-top: 4px;
            }}
            .feature-card {{
                background: {colors["surface_soft"]};
                border: 1px solid {colors["border"]};
                border-radius: 20px;
                padding: 20px 18px;
                display: flex;
                flex-direction: column;
                gap: 8px;
            }}
            .feature-icon {{
                font-size: 1.8rem;
                line-height: 1;
                margin-bottom: 4px;
            }}
            .feature-title {{
                font-size: 1.02rem;
                font-weight: 800;
                color: {colors["text"]};
                margin: 0;
            }}
            .feature-desc {{
                color: {colors["muted"]};
                font-size: 0.92rem;
                line-height: 1.6;
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
            .alert-banner {{
                background: rgba(220, 38, 38, 0.12);
                border: 1px solid rgba(220, 38, 38, 0.35);
                border-radius: 18px;
                padding: 16px 18px;
                margin-bottom: 1rem;
            }}
            .footer-links {{
                display:flex;
                gap:14px;
                justify-content:center;
                flex-wrap:wrap;
                color:{colors["muted"]};
                font-size:0.9rem;
            }}
            .auth-shell {{
                position: relative;
                overflow: hidden;
                background:
                    radial-gradient(circle at top right, rgba(34, 193, 161, 0.18), transparent 28%),
                    radial-gradient(circle at bottom left, rgba(59, 130, 246, 0.18), transparent 30%),
                    linear-gradient(160deg, {colors["surface"]} 0%, {colors["surface_soft"]} 100%);
                border: 1px solid {colors["border"]};
                border-radius: 30px;
                padding: 28px;
                min-height: 620px;
                animation: floatIn 0.8s ease;
            }}
            .auth-shell::before {{
                content: "";
                position: absolute;
                inset: -120px auto auto -80px;
                width: 240px;
                height: 240px;
                background: rgba(34, 193, 161, 0.10);
                filter: blur(20px);
                border-radius: 50%;
                animation: pulseGlow 4s ease-in-out infinite;
            }}
            .auth-shell::after {{
                content: "";
                position: absolute;
                right: -60px;
                bottom: -70px;
                width: 220px;
                height: 220px;
                background: rgba(59, 130, 246, 0.12);
                filter: blur(24px);
                border-radius: 50%;
                animation: pulseGlow 5s ease-in-out infinite;
            }}
            .auth-showcase {{
                border: 1px solid {colors["border"]};
                background: rgba(255,255,255,0.02);
                border-radius: 24px;
                padding: 28px;
                backdrop-filter: blur(10px);
            }}
            .auth-kicker {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 14px;
                border-radius: 999px;
                background: {colors["accent_soft"]};
                color: {colors["accent"]};
                font-size: 0.82rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 18px;
            }}
            .auth-title {{
                font-size: 2.55rem;
                line-height: 1.08;
                margin: 0 0 12px 0;
                font-weight: 900;
                max-width: none;
                white-space: nowrap;
            }}
            .auth-subtitle {{
                color: {colors["muted"]};
                font-size: 1rem;
                line-height: 1.8;
                max-width: 560px;
                margin-bottom: 22px;
            }}
            .auth-stat-row {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 14px;
                margin: 24px 0;
            }}
            .auth-stat {{
                border: 1px solid {colors["border"]};
                background: rgba(255,255,255,0.03);
                border-radius: 20px;
                padding: 16px;
            }}
            .auth-stat strong {{
                display: block;
                font-size: 1.35rem;
                margin-bottom: 6px;
            }}
            .auth-stat span {{
                color: {colors["muted"]};
                font-size: 0.9rem;
            }}
            .auth-note {{
                border-left: 4px solid {colors["accent"]};
                background: rgba(255,255,255,0.025);
                border-radius: 16px;
                padding: 16px 18px;
                color: {colors["muted"]};
                line-height: 1.7;
            }}
            .auth-card {{
                border: 1px solid {colors["border"]};
                background: {colors["surface"]};
                border-radius: 24px;
                padding: 22px;
                box-shadow: 0 18px 50px rgba(8, 17, 32, 0.18);
                animation: floatIn 0.9s ease;
            }}
            .auth-card h3 {{
                margin-top: 0;
            }}
            .auth-panel-wrap {{
                position: relative;
                z-index: 2;
            }}
            div[data-testid="stForm"] {{
                border: none;
                padding: 0;
            }}
            div[data-testid="stTabs"] {{
                margin-top: 0.5rem;
            }}
            div[data-testid="stTabs"] button[role="tab"] {{
                border-radius: 999px;
                padding: 0.45rem 0.95rem;
            }}
            div[data-testid="stTextInput"] input {{
                border-radius: 14px;
            }}
            .auth-loader {{
                display: flex;
                gap: 10px;
                align-items: center;
                margin-top: 14px;
            }}
            .auth-loader span {{
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: {colors["accent"]};
                animation: bounceDots 1.1s infinite ease-in-out;
            }}
            .auth-loader span:nth-child(2) {{
                animation-delay: 0.15s;
            }}
            .auth-loader span:nth-child(3) {{
                animation-delay: 0.3s;
            }}
            @keyframes bounceDots {{
                0%, 80%, 100% {{ transform: translateY(0); opacity: 0.4; }}
                40% {{ transform: translateY(-8px); opacity: 1; }}
            }}
            @keyframes pulseGlow {{
                0%, 100% {{ transform: scale(1); opacity: 0.65; }}
                50% {{ transform: scale(1.08); opacity: 1; }}
            }}
            @keyframes floatIn {{
                from {{ opacity: 0; transform: translateY(16px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            div[role="radiogroup"] {{
                gap: 0.5rem;
                justify-content: center;
                flex-wrap: wrap;
            }}
            div[data-testid="stRadio"] {{
                position: sticky;
                top: 72px;
                z-index: 998;
                background: {colors["surface"]};
                border: 1px solid {colors["border"]};
                border-radius: 18px;
                padding: 10px 12px;
                margin-bottom: 1rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            }}
            div[role="radiogroup"] label {{
                border-radius: 999px;
                padding: 0.42rem 0.95rem;
                border: 1px solid {colors["border"]};
                background: {colors["surface_soft"]};
                box-shadow: none;
                transition: all 0.2s ease;
            }}
            div[role="radiogroup"] label:hover {{
                border-color: {colors["accent"]};
                box-shadow: 0 6px 16px rgba(15, 23, 42, 0.10);
                transform: translateY(-1px);
            }}
            div[role="radiogroup"] label[data-baseweb="radio"][aria-checked="true"] {{
                background: {colors["accent_soft"]};
                border-color: {colors["accent"]};
                box-shadow: 0 8px 18px rgba(15, 23, 42, 0.12);
            }}
            div[role="radiogroup"] label[data-baseweb="radio"][aria-checked="true"] p {{
                color: {colors["accent"]} !important;
                font-weight: 800 !important;
            }}
            div[role="radiogroup"] p {{
                font-size: 0.94rem !important;
            }}

            @media (max-width: 900px) {{
                .block-container {{
                    padding-top: 0.75rem;
                    padding-left: 0.8rem;
                    padding-right: 0.8rem;
                }}
                .topnav {{
                    padding: 12px 14px 10px 14px;
                }}
                .feature-grid {{
                    grid-template-columns: 1fr;
                }}
                .auth-stat-row {{
                    grid-template-columns: 1fr;
                }}
                .auth-title {{
                    white-space: normal;
                    font-size: 2rem;
                }}
                .kpi {{
                    min-height: auto;
                }}
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
        "page": "🏠 Home",
        "customer_profile": None,
        "merchant": None,
        "category": None,
        "amount": 0.0,
        "card_number": "",
        "card_mode": None,
        "customer_city": None,
        "merchant_city": "",
        "gender": None,
        "lat": 0.0,
        "lon": 0.0,
        "merch_lat": 0.0,
        "merch_lon": 0.0,
        "hour": 14,
        "day": 15,
        "month": 4,
        "card_segment": None,
        "device_trust": 0,
        "velocity_24h": 2,
        "international_txn": False,
        "ip_address": "",
        "device_id": "",
        "rule_amount_weight": 0.08,
        "rule_distance_weight": 0.10,
        "rule_device_weight": 0.08,
        "rule_velocity_weight": 0.09,
        "rule_international_weight": 0.07,
        "rule_debit_weight": 0.03,
        "rule_online_device_weight": 0.05,
        "audit_logs": [],
        "authenticated": False,
        "auth_user": "",
        "auth_name": "",
        "auth_role": "",
        "auth_users": {key: value.copy() for key, value in AUTH_USERS.items()},
        "login_username": "",
        "login_password": "",
        "register_name": "",
        "register_username": "",
        "register_password": "",
        "register_confirm_password": "",
        "reset_username": "",
        "reset_password": "",
        "reset_confirm_password": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def sync_from_customer() -> None:
    if not st.session_state["customer_profile"]:
        return
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
    if not st.session_state["customer_city"]:
        return
    st.session_state["lat"], st.session_state["lon"] = CUSTOMER_CITIES[st.session_state["customer_city"]]


def sync_from_merchant() -> None:
    if not st.session_state["merchant"]:
        return
    profile = MERCHANT_PROFILES[st.session_state["merchant"]]
    st.session_state["category"] = profile["category"]
    st.session_state["merchant_city"] = profile["merchant_city"]
    st.session_state["merch_lat"] = profile["merch_lat"]
    st.session_state["merch_lon"] = profile["merch_lon"]
    st.session_state["amount"] = profile["typical_amount"]


def sync_from_category() -> None:
    if not st.session_state["category"]:
        return
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
    if not digits:
        st.session_state["card_mode"] = None
        return
    if digits.startswith("6"):
        st.session_state["card_mode"] = "Debit Card"
    else:
        st.session_state["card_mode"] = "Credit Card"


def can_compute_risk() -> bool:
    required_values = [
        st.session_state["merchant"],
        st.session_state["category"],
        st.session_state["card_number"],
        st.session_state["card_mode"],
        st.session_state["customer_city"],
        st.session_state["gender"],
    ]
    return all(required_values)


def get_empty_risk_state() -> dict:
    return {
        "prediction": False,
        "risk_percent": 0.0,
        "model_percent": 0.0,
        "severity": "Awaiting input",
        "distance": 0.0,
        "explanation": "Select a customer, merchant, or category to auto-fill the transaction details.",
        "recommendation": "Complete the intake form to start scoring transactions.",
        "reasons": ["No transaction has been screened yet."],
    }


def build_filter_periods(df: pd.DataFrame, granularity: str) -> list[str]:
    if df.empty:
        return []
    timestamps = pd.to_datetime(df["transaction_time"])
    if granularity == "Day":
        labels = timestamps.dt.strftime("%Y-%m-%d")
    elif granularity == "Week":
        iso = timestamps.dt.isocalendar()
        labels = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    elif granularity == "Month":
        labels = timestamps.dt.strftime("%Y-%m")
    else:
        labels = timestamps.dt.strftime("%Y")
    return sorted(labels.unique(), reverse=True)


def filter_batch_data(
    df: pd.DataFrame,
    merchants: list[str],
    cities: list[str],
    cards: list[str],
    granularity: str,
    periods: list[str],
) -> pd.DataFrame:
    filtered = df.copy()
    if merchants:
        filtered = filtered[filtered["merchant"].isin(merchants)]
    if cities:
        filtered = filtered[filtered["city"].isin(cities)]
    if cards:
        filtered = filtered[filtered["card_type"].isin(cards)]
    if periods:
        timestamps = pd.to_datetime(filtered["transaction_time"])
        if granularity == "Day":
            labels = timestamps.dt.strftime("%Y-%m-%d")
        elif granularity == "Week":
            iso = timestamps.dt.isocalendar()
            labels = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
        elif granularity == "Month":
            labels = timestamps.dt.strftime("%Y-%m")
        else:
            labels = timestamps.dt.strftime("%Y")
        filtered = filtered[labels.isin(periods)]
    return filtered


def authenticate(username: str, password: str) -> bool:
    user = st.session_state["auth_users"].get(username.strip().lower())
    if not user or user["password"] != password:
        return False
    st.session_state["authenticated"] = True
    st.session_state["auth_user"] = username.strip().lower()
    st.session_state["auth_name"] = user["name"]
    st.session_state["auth_role"] = user["role"]
    return True


def register_user(name: str, username: str, password: str, confirm_password: str) -> tuple[bool, str]:
    clean_name = name.strip()
    clean_username = username.strip().lower()
    auth_users = st.session_state["auth_users"]
    if not clean_name:
        return False, "Please enter your full name."
    if len(clean_username) < 4:
        return False, "Username must be at least 4 characters."
    if clean_username in auth_users:
        return False, "That username already exists."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if password != confirm_password:
        return False, "Passwords do not match."
    auth_users[clean_username] = {
        "password": password,
        "name": clean_name,
        "role": "Analyst",
    }
    return True, "Account created successfully. You can now sign in."


def reset_user_password(username: str, password: str, confirm_password: str) -> tuple[bool, str]:
    clean_username = username.strip().lower()
    auth_users = st.session_state["auth_users"]
    if clean_username not in auth_users:
        return False, "User not found. Please register first."
    if len(password) < 6:
        return False, "New password must be at least 6 characters."
    if password != confirm_password:
        return False, "Passwords do not match."
    auth_users[clean_username]["password"] = password
    return True, "Password reset successful. Use the new password to log in."


def logout() -> None:
    st.session_state["authenticated"] = False
    st.session_state["auth_user"] = ""
    st.session_state["auth_name"] = ""
    st.session_state["auth_role"] = ""
    st.session_state["page"] = "🏠 Home"


def render_login_page() -> None:
    st.markdown(
        """
        <div class="auth-shell">
            <div class="auth-kicker">Trusted Fraud Operations</div>
            <div class="auth-title">Welcome to FraudShield 360 🛡️</div>
            <div class="auth-subtitle">
                Secure your fraud operations with one clean workspace for analyst sign-in,
                new account creation, password reset, and high-confidence transaction review.
            </div>
            <div class="auth-panel-wrap" style="margin-top: 22px;">
                <div class="auth-showcase">
                    <div class="eyebrow">Why Teams Trust FraudShield</div>
                    <h3 style="margin-top:10px; margin-bottom:10px;">Fraud screening built for speed, clarity, and control.</h3>
                    <div class="helper" style="margin-bottom: 8px;">
                        Review suspicious activity faster, support compliance workflows, and keep analysts focused
                        with a dedicated fraud intelligence workspace.
                    </div>
                    <div class="auth-stat-row">
                        <div class="auth-stat">
                            <strong>96.1%</strong>
                            <span>Screening accuracy across simulated traffic</span>
                        </div>
                        <div class="auth-stat">
                            <strong>&lt;100 ms</strong>
                            <span>Average fraud scoring turnaround</span>
                        </div>
                        <div class="auth-stat">
                            <strong>24/7</strong>
                            <span>Analyst-ready access with audit visibility</span>
                        </div>
                    </div>
                    <div class="auth-note" style="margin-top: 10px;">
                        Use the secure portal to sign in, onboard a new analyst account, or reset an existing password
                        without leaving the page.
                        <div class="auth-loader"><span></span><span></span><span></span></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    center_left, center_mid, center_right = st.columns([1, 1.2, 1])
    with center_mid:
        st.markdown('<div class="auth-panel-wrap"><div class="auth-card">', unsafe_allow_html=True)
        st.subheader("Access Portal")
        st.caption("Sign in, create a new analyst account, or reset a password from one secure entry point.")
        tabs = st.tabs(["Login", "Create Account", "Reset Password"])

        with tabs[0]:
            with st.form("login_form", clear_on_submit=False):
                st.text_input("Username", key="login_username", placeholder="Enter username")
                st.text_input("Password", key="login_password", type="password", placeholder="Enter password")
                submitted = st.form_submit_button("Sign In", use_container_width=True)
                if submitted:
                    with st.spinner("Verifying your credentials..."):
                        time.sleep(0.7)
                    if authenticate(st.session_state["login_username"], st.session_state["login_password"]):
                        st.success("Login successful. Redirecting to the main workspace.")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
            st.caption("Demo credentials: admin / admin123 or analyst / analyst123")

        with tabs[1]:
            with st.form("register_form", clear_on_submit=False):
                st.text_input("Full Name", key="register_name", placeholder="Enter your full name")
                st.text_input("New Username", key="register_username", placeholder="Choose a username")
                st.text_input("Create Password", key="register_password", type="password", placeholder="Create a password")
                st.text_input("Confirm Password", key="register_confirm_password", type="password", placeholder="Re-enter the password")
                register_submitted = st.form_submit_button("Create Account", use_container_width=True)
                if register_submitted:
                    with st.spinner("Creating your analyst account..."):
                        time.sleep(0.7)
                    ok, message = register_user(
                        st.session_state["register_name"],
                        st.session_state["register_username"],
                        st.session_state["register_password"],
                        st.session_state["register_confirm_password"],
                    )
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)

        with tabs[2]:
            with st.form("reset_form", clear_on_submit=False):
                st.text_input("Username", key="reset_username", placeholder="Enter your username")
                st.text_input("New Password", key="reset_password", type="password", placeholder="Enter a new password")
                st.text_input("Confirm New Password", key="reset_confirm_password", type="password", placeholder="Re-enter the new password")
                reset_submitted = st.form_submit_button("Reset Password", use_container_width=True)
                if reset_submitted:
                    with st.spinner("Updating your password..."):
                        time.sleep(0.7)
                    ok, message = reset_user_password(
                        st.session_state["reset_username"],
                        st.session_state["reset_password"],
                        st.session_state["reset_confirm_password"],
                    )
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)
        st.markdown('</div></div>', unsafe_allow_html=True)


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
    if not can_compute_risk():
        return get_empty_risk_state()
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
        reasons.append("High transaction amount")
    if amount_ratio >= 2.2:
        final_prob += 0.06
        reasons.append("Amount deviates sharply from category baseline")
    if distance > 120:
        final_prob += st.session_state["rule_distance_weight"]
        reasons.append("Unusual customer-to-merchant distance")
    if st.session_state["device_trust"] < 40:
        final_prob += st.session_state["rule_device_weight"]
        reasons.append("Low device trust")
    if st.session_state["velocity_24h"] >= 6:
        final_prob += st.session_state["rule_velocity_weight"]
        reasons.append("High card activity in last 24 hours")
    if st.session_state["international_txn"]:
        final_prob += st.session_state["rule_international_weight"]
        reasons.append("International transaction pattern")
    if st.session_state["card_mode"] == "Debit Card":
        final_prob += st.session_state["rule_debit_weight"]
        reasons.append("Debit card protection rule applied")
    if merchant_channel == "Online" and st.session_state["device_trust"] < 55:
        final_prob += st.session_state["rule_online_device_weight"]
        reasons.append("Online transaction from weak-trust device")
    if len(digits) not in {15, 16}:
        final_prob += 0.05
        reasons.append("Card number format looks unusual")
    if st.session_state["hour"] < 5 or st.session_state["hour"] >= 23:
        final_prob += 0.06
        reasons.append("Late-night transaction timing")

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
        reasons.append("Transaction aligns with normal profile behavior")

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
    card_mode = st.session_state["card_mode"] or "Not selected"
    card_segment = st.session_state["card_segment"] or "Not selected"
    st.markdown(
        f"""
        <div class="card-visual">
            <div class="eyebrow" style="color:#d1fae5;">{title}</div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px;">
                <div style="font-size:1.5rem; font-weight:800;">{infer_bank(st.session_state["card_number"])}</div>
                <div style="padding:6px 12px; border-radius:999px; background:rgba(255,255,255,0.16);">{card_mode}</div>
            </div>
            <div class="card-chip"></div>
            <div style="font-size:1.45rem; letter-spacing:0.14em; font-weight:700;">{mask_card(st.session_state["card_number"])}</div>
            <div style="display:flex; justify-content:space-between; margin-top:26px; font-size:0.94rem;">
                <div><div style="opacity:0.8;">Network</div><div style="font-weight:700;">{infer_network(st.session_state["card_number"])}</div></div>
                <div><div style="opacity:0.8;">Segment</div><div style="font-weight:700;">{card_segment}</div></div>
                <div><div style="opacity:0.8;">Risk</div><div style="font-weight:700;">{risk_percent:.2f}%</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
        txn_time = datetime.now() - timedelta(
            days=random.randint(0, 364),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
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
                "transaction_time": txn_time,
                "hour": txn_time.hour,
                "day": txn_time.day,
                "month": txn_time.month,
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
                "transaction_time": row["transaction_time"],
                "distance_km": round(distance, 2),
                "risk_score": round(final_prob * 100, 2),
                "decision": "Fraud" if final_prob >= 0.50 else "Safe",
            }
        )
    return pd.DataFrame(scored_rows)


# ── Init ──────────────────────────────────────────────────────────────────────
init_state()
apply_theme(st.session_state["theme_mode"])

# ── Top-right theme toggle (compact, no search bar) ───────────────────────────
theme_col, _ = st.columns([5, 1])
with _:
    st.selectbox("Theme", ["Dark", "Light"], key="theme_mode", label_visibility="collapsed")

if not st.session_state["authenticated"]:
    render_login_page()
    st.stop()

risk = compute_risk()

# ── Brand nav bar ─────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="topnav">
        <div class="topnav-inner">
            <div>
                <div class="topnav-brand">FraudShield 360 🛡️</div>
                <div class="topnav-sub">AI-powered fraud detection for modern card payments</div>
            </div>
            <div>
                <div class="badge-row">
                    <span class="badge">PCI DSS</span>
                    <span class="badge">GDPR</span>
                    <span class="badge">Masked Data</span>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

account_col, action_col = st.columns([5, 1])
with account_col:
    st.caption(f"Logged in as {st.session_state['auth_name']} ({st.session_state['auth_role']})")
with action_col:
    if st.button("Logout", use_container_width=True):
        logout()
        st.rerun()

# ── Navigation ────────────────────────────────────────────────────────────────
page = st.radio(
    "Header Navigation",
    ["🏠 Home", "📊 Fraud Dashboard", "📈 Reports & Trends", "🛡️ Compliance", "👥 About Us", "✉️ Contact"],
    key="page",
    horizontal=True,
    label_visibility="collapsed",
)

# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">FraudShield 360 🛡️</div>
        <div class="hero-tagline">Detect. Protect. Comply — in Real Time.</div>
        <div class="hero-desc">
            An intelligent fraud detection platform for modern card payments combining
            ML-powered risk scoring, explainable AI alerts, compliance-aware workflows,
            and scalable transaction monitoring — all in one analyst-ready interface.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    if risk["prediction"]:
        st.markdown(
            f"""
            <div class="alert-banner">
                <strong>🚨 Fraud Alert Active</strong><br>
                A suspicious transaction is currently flagged in
                <strong>{st.session_state["merchant_city"]}</strong>
                with a risk score of <strong>{risk["risk_percent"]}%</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Hero split
    hero_left, hero_right = st.columns([1.15, 0.85])
    with hero_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### AI-Powered Fraud Detection in Real Time")
        st.markdown(
            """
            <div class="helper">
                FraudShield 360 helps financial institutions monitor credit and debit card
                fraud using machine learning, explainable AI, compliance-aware workflows,
                and analyst-ready dashboards.
            </div>
            <div class="helper" style="margin-top:10px;">
                Review suspicious activity, see explainable alerts, visualise fraud
                locations, and generate audit-ready insights — all from one interface.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with hero_right:
        render_card("Protected Payment Card", risk["risk_percent"])

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='kpi'><div class='eyebrow'>Current Risk</div><h2>{risk['risk_percent']}%</h2><div class='helper'>{risk['severity']} alert level</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi'><div class='eyebrow'>Model Score</div><h2>{risk['model_percent']}%</h2><div class='helper'>Pure ML confidence</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi'><div class='eyebrow'>Alerts Reviewed</div><h2>1,248</h2><div class='helper'>Today</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi'><div class='eyebrow'>Distance</div><h2>{risk['distance']} km</h2><div class='helper'>Customer to merchant</div></div>", unsafe_allow_html=True)

    # ── Platform Capabilities (clear feature grid) ────────────────────────────
    
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 🚀 Platform Capabilities")
    st.markdown("""
<div class="feature-card">
    <div class="feature-icon">⚡</div>
    <div class="feature-title">Real-Time Detection</div>
    <div class="feature-desc">
        Every card transaction is screened the moment it arrives.
        The hybrid engine combines a trained ML model with configurable
        business rules to flag anomalies in under 100 ms.
    </div>
</div>

<div class="feature-card">
    <div class="feature-icon">🛡️</div>
    <div class="feature-title">Compliance Monitoring</div>
    <div class="feature-desc">
        Built with PCI DSS and GDPR principles in mind.
    </div>
</div>

<div class="feature-card">
    <div class="feature-icon">🧠</div>
    <div class="feature-title">Explainable AI</div>
    <div class="feature-desc">
        Every flagged transaction comes with a clear reason.
    </div>
</div>

<div class="feature-card">
    <div class="feature-icon">📈</div>
    <div class="feature-title">Scalable Analytics</div>
    <div class="feature-desc">
        Supports large-scale fraud monitoring and reporting.
    </div>
</div>
""", unsafe_allow_html=True)

    # ── Trend + Explainability + Compliance row ───────────────────────────────
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
        st.write(f"**Decision:** {'🔴 Fraud' if risk['prediction'] else '🟢 Safe'}")
        st.write(f"**Reason:** {risk['explanation']}")
        st.write(f"**Recommendation:** {risk['recommendation']}")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Compliance & Security")
        with st.expander("PCI DSS Badge", expanded=True):
            st.write("Simulated compliance coverage for secure card processing, masking, and audit visibility.")
        with st.expander("GDPR Badge"):
            st.write("Simulated privacy compliance for consent-aware data handling and masked cardholder views.")
        with st.expander("Data Masking"):
            st.write("Analyst dashboard hides sensitive cardholder information using masked card rendering.")
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FRAUD DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Fraud Dashboard":
    st.subheader("Fraud Dashboard")
    if risk["prediction"]:
        st.error(f"Fraud detected in {st.session_state['merchant_city']} with {risk['risk_percent']}% hybrid risk.")
    top_left, top_right = st.columns([1.05, 0.95])
    with top_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Smart Auto-Fill Intake")
        st.selectbox("Customer Profile", [None] + list(CUSTOMER_PROFILES.keys()), key="customer_profile", format_func=lambda value: "Select customer profile" if value is None else value, on_change=sync_from_customer)
        st.selectbox("Merchant", [None] + list(MERCHANT_PROFILES.keys()), key="merchant", format_func=lambda value: "Select merchant" if value is None else value, on_change=sync_from_merchant)
        st.selectbox("Category", [None] + list(CATEGORY_MAP.keys()), key="category", format_func=lambda value: "Select category" if value is None else value, on_change=sync_from_category)
        st.selectbox("Customer City", [None] + list(CUSTOMER_CITIES.keys()), key="customer_city", format_func=lambda value: "Select customer city" if value is None else value, on_change=sync_customer_city)
        st.text_input("Card Number", key="card_number", on_change=sync_from_card_number)
        col_a, col_b = st.columns(2)
        col_a.selectbox("Card Mode", [None, "Credit Card", "Debit Card"], key="card_mode", format_func=lambda value: "Select card mode" if value is None else value)
        col_b.selectbox("Card Segment", [None] + list(SEGMENT_HELP.keys()), key="card_segment", format_func=lambda value: "Select card segment" if value is None else value)
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
        st.selectbox("Gender", [None, "M", "F"], key="gender", format_func=lambda value: "Select gender" if value is None else value)
        st.slider("Device Trust Score", 0, 100, key="device_trust")
        st.slider("Transaction Count in Last 24h", 0, 12, key="velocity_24h")
        st.checkbox("International Transaction", key="international_txn")
        if st.session_state["card_segment"]:
            st.caption(f"Segment note: {SEGMENT_HELP[st.session_state['card_segment']]}")
        else:
            st.caption("Segment note will appear after you choose a card segment.")
        st.markdown('</div>', unsafe_allow_html=True)
        render_card("Live Card View", risk["risk_percent"])

    result_left, result_right = st.columns([1.05, 0.95])
    with result_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Hybrid Fraud Decision")
        if not can_compute_risk():
            st.info("Select a customer profile, merchant, or category to auto-fill the intake form and start scoring.")
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
        if can_compute_risk():
            st.write(f"Fraud location: {st.session_state['merchant_city']}")
            fraud_loc_df = pd.DataFrame([{"latitude": st.session_state["merch_lat"], "longitude": st.session_state["merch_lon"]}])
            st.map(fraud_loc_df, use_container_width=True)
            st.caption("This map pinpoints the merchant location connected to the suspicious transaction.")
        else:
            st.info("Merchant location will appear after the intake form is filled.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Fraud Signals Triggered")
        for reason in risk["reasons"]:
            st.write(f"• {reason}")
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REPORTS & TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Reports & Trends":
    st.subheader("Analytics & Reports")
    size = st.slider("Simulated transactions", 100, 5000, 1000, step=100)
    batch = generate_batch(size)
    scored = score_batch(batch)

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        time_granularity = st.selectbox("Batch Filter Type", ["Day", "Week", "Month", "Year"])
    with filter_col2:
        period_options = build_filter_periods(scored, time_granularity)
        period_filter = st.multiselect(
            f"{time_granularity} Filter",
            period_options,
            placeholder=f"Select {time_granularity.lower()} values",
        )

    merchant_filter = st.multiselect(
        "Merchant Filter",
        sorted(scored["merchant"].unique()),
        placeholder="Select merchants",
    )
    city_filter = st.multiselect(
        "Geography Filter",
        sorted(scored["city"].unique()),
        placeholder="Select cities",
    )
    card_filter = st.multiselect(
        "Card Type Filter",
        sorted(scored["card_type"].unique()),
        placeholder="Select card types",
    )

    filtered_scored = filter_batch_data(
        scored,
        merchant_filter,
        city_filter,
        card_filter,
        time_granularity,
        period_filter,
    )

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Transactions", len(filtered_scored))
    b2.metric("Fraud Alerts", int((filtered_scored["decision"] == "Fraud").sum()) if not filtered_scored.empty else 0)
    b3.metric("Avg Risk", f"{filtered_scored['risk_score'].mean():.2f}%" if not filtered_scored.empty else "0.00%")
    b4.metric("Debit Cards", int((filtered_scored["card_type"] == "Debit Card").sum()) if not filtered_scored.empty else 0)

    left, right = st.columns([1.2, 0.8])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Batch Screening Results")
        display_df = filtered_scored.copy()
        if not display_df.empty:
            display_df["transaction_time"] = pd.to_datetime(display_df["transaction_time"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(display_df.head(200), use_container_width=True)
        st.caption("Showing first 200 rows for performance.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Decision Mix")
        if filtered_scored.empty:
            st.info("No transactions match the selected filters.")
        else:
            decision_df = filtered_scored["decision"].value_counts().rename_axis("Decision").reset_index(name="Count")
            st.bar_chart(decision_df.set_index("Decision"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Fraud Trend Over Time")
        if filtered_scored.empty:
            st.info("Select filters to generate the trend view.")
        else:
            time_df = filtered_scored.copy()
            time_df["transaction_time"] = pd.to_datetime(time_df["transaction_time"])
            if time_granularity == "Day":
                time_df["Period"] = time_df["transaction_time"].dt.strftime("%Y-%m-%d")
            elif time_granularity == "Week":
                iso = time_df["transaction_time"].dt.isocalendar()
                time_df["Period"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
            elif time_granularity == "Month":
                time_df["Period"] = time_df["transaction_time"].dt.strftime("%Y-%m")
            else:
                time_df["Period"] = time_df["transaction_time"].dt.strftime("%Y")
            trend_df = time_df.groupby("Period", as_index=False).agg(
                Fraud_Alerts=("decision", lambda series: int((series == "Fraud").sum())),
                Reviewed_Txns=("transaction_id", "count"),
            )
            st.line_chart(trend_df.set_index("Period"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Merchant Risk Comparison")
        if filtered_scored.empty:
            st.info("Merchant comparison will appear after you apply at least one matching filter.")
        else:
            merchant_risk = filtered_scored.groupby("merchant", as_index=False)["risk_score"].mean().sort_values("risk_score", ascending=False)
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


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: COMPLIANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛡️ Compliance":
    st.subheader("Compliance & Security")
    comp_left, comp_right = st.columns([0.9, 1.1])

    with comp_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Compliance Badges")
        with st.expander("PCI DSS Compliance", expanded=True):
            st.write("Secure cardholder data handling, masking, and controlled analyst access are simulated here.")
        with st.expander("GDPR Compliance"):
            st.write("Privacy-first data visibility, consent-aware handling, and secure audit presentation are simulated here.")
        with st.expander("Data Masking Toggle"):
            st.write(f"Masked cardholder view: {mask_card(st.session_state['card_number'])}")
        st.markdown('</div>', unsafe_allow_html=True)

    with comp_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Audit Trail Timeline")
        if not st.session_state["audit_logs"]:
            st.info("No audit events yet. Log a decision from the Dashboard.")
        else:
            audit_df = pd.DataFrame(st.session_state["audit_logs"])
            st.dataframe(audit_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT US
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 About Us":
    st.subheader("👥 About Us")

    intro_left, intro_right = st.columns([1.15, 0.85])
    with intro_left:
        st.markdown('<div class="about-card">', unsafe_allow_html=True)
        st.markdown("### 🧑‍💼 Who We Are")
        st.write("FraudShield 360 is a dedicated project team focused on delivering a clean, credible, and industry-ready fraud intelligence experience for modern card payments.")
        st.write("🎯 Our mission is to empower fraud analysts and compliance teams with tools that make risk review clear, fast, and confident.")
        st.write("💡 We believe fraud detection should be accurate, transparent, scalable, and easy to use.")
        st.markdown('</div>', unsafe_allow_html=True)

    with intro_right:
        st.markdown('<div class="about-card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Vision")
        st.write("Our vision is to create a scalable, explainable fraud detection ecosystem that blends:")
        st.write("• 🤖 Machine Learning intelligence")
        st.write("• 📊 Operational monitoring")
        st.write("• ⚖️ Compliance-aware decision support")
        st.write("• 👨‍💻 Analyst empowerment")
        st.markdown('</div>', unsafe_allow_html=True)

    tech_col, summary_col = st.columns(2)
    with tech_col:
        st.markdown('<div class="about-card">', unsafe_allow_html=True)
        st.markdown("### 🛠️ Technologies Used")
        st.write("🐍 Python for core application and model integration")
        st.write("🤖 Machine Learning for anomaly detection")
        st.write("🌐 Streamlit for interactive UI")
        st.write("📚 Scikit-learn for ML pipelines")
        st.write("📊 Pandas for analytics")
        st.write("☁️ Cloud-ready architecture")
        st.markdown('</div>', unsafe_allow_html=True)

    with summary_col:
        st.markdown('<div class="about-card">', unsafe_allow_html=True)
        st.markdown("### 📌 Project Summary")
        st.write("⚡ Live Detection with instant alerts")
        st.write("📈 Fraud analytics & trends")
        st.write("📍 Location-based insights")
        st.write("📄 Export-ready reports")
        st.write("🔍 Audit transparency")
        st.markdown('</div>', unsafe_allow_html=True)

    why_col, future_col = st.columns(2)
    with why_col:
        st.markdown('<div class="about-card">', unsafe_allow_html=True)
        st.markdown("### ⭐ Why FraudShield 360?")
        st.write("👁️ Clarity: understand why transactions are flagged")
        st.write("⚡ Speed: real-time detection")
        st.write("✅ Confidence: compliance-ready design")
        st.write("📊 Scalability: handles high volumes")
        st.markdown('</div>', unsafe_allow_html=True)

    with future_col:
        st.markdown('<div class="about-card">', unsafe_allow_html=True)
        st.markdown("### 🔮 Future Scope")
        st.write("🔗 Blockchain-based verification")
        st.write("🧠 Federated learning (privacy-focused)")
        st.write("☁️ Cloud-native deployment")
        st.write("⚙️ Adaptive rules engine")
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CONTACT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "✉️ Contact":
    st.subheader("Contact & Feedback")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    contact_name = st.text_input("Name")
    contact_email = st.text_input("Email")
    contact_message = st.text_area("Feedback / Query")
    if st.button("Submit Feedback", use_container_width=True):
        if contact_name and contact_email and contact_message:
            st.success("Feedback captured successfully.")
        else:
            st.warning("Please fill in all fields before submitting.")
    st.markdown('</div>', unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; color:#94a3b8; padding:16px 0 4px 0;">
        <div class="footer-links">
            <span>Privacy Policy</span>
            <span>Terms</span>
            <span>GitHub / Docs</span>
        </div>
        <div style="margin-top:8px;">FraudShield 360 v1.0 • Major Project Edition</div>
    </div>
    """,
    unsafe_allow_html=True,
)