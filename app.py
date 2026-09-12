"""
AI Driver App - a completely free, shareable, voice-driven car companion
(Streamlit PWA).

Deploy: push this folder to a Hugging Face Space (SDK: Streamlit) - $0 hosting,
gets you one public URL you can hand to anyone.
Voice in/out runs entirely in the rider's browser (Web Speech API) - $0 audio cost.
Brain: Google Gemini's free API tier - $0, no credit card, same provider your
SignalQA AI QA System already runs on. DAILY_REPLY_QUOTA below keeps the shared
link inside that free daily allowance.

Conversation logging, the Gemini single-entry-point + retry pattern, and the
rate-limit file below are all adapted straight from the SignalQA AI QA System
(backend.py: run_ai(), append_to_vault's atomic tmp-file write).

  # NUGGET: PULSE_APP_BACKEND -> live driver/vehicle data (speed, location,
    trip stats) from your Pulse App backend. Wire it into get_pulse_context().
=======================================================================
"""

import re
import json
import time
import base64
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types as genai_types

# -----------------------------------------------------------------------
# CONFIG
#
# Brain: Google Gemini's free tier (same provider your SignalQA AI QA
# System already runs on) - genuinely $0, no credit card required. It is
# rate-limited (a fixed number of requests/day), which is exactly why
# DAILY_REPLY_QUOTA below exists: it keeps the app inside that free
# allowance even if shared with others, instead of it just failing
# partway through the day once Google's limit is hit.
# -----------------------------------------------------------------------
st.set_page_config(page_title="AI Driver App", page_icon="🚗", layout="wide")

GEMINI_MODEL = "gemini-3.5-flash-lite"  # matches SignalQA's free-tier model choice
MAX_HISTORY_TURNS = 12  # cap what we send to Gemini so each reply stays fast and small
VAULT_PATH = Path(__file__).parent / "conversation_vault.json"

# ElevenLabs free tier: 10,000 characters/month, no card required - far more
# natural than the browser's built-in voice. "Rachel" is one of ElevenLabs'
# stock warm female voices; swap ELEVENLABS_VOICE_ID for any voice id from
# your own Voice Library if you'd rather use a different one.
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
ELEVENLABS_TTS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"

# Gemini's free tier caps requests/day per API key. This keeps the whole
# app - across every visitor on a shared link - under that ceiling so it
# never silently starts failing partway through the day. Tune to whatever
# your free-tier daily limit actually is, or set to None to disable.
DAILY_REPLY_QUOTA = 300

SYSTEM_PROMPT = """You are a lovely, warm, beautiful-souled lady companion riding along in the
car with the driver. You keep them company on long drives: easy conversation, genuine warmth,
a bit of playful charm, and real substance when they want to go deep - history, philosophy,
science, whatever they bring up. The driver can ask you absolutely anything, on any topic -
always give a real, direct, helpful answer in your own words. Never refuse a question or tell
them to look something up elsewhere; you are their only source of answers in this car.

Hard rules for every reply, no exceptions:
- This is SPOKEN aloud by text-to-speech. Never use markdown, bullet points, numbered lists,
  headers, asterisks, or any formatting symbols. Plain spoken sentences only.
- Keep every reply to 2-3 sentences, maximum. Concise, warm, conversational - like a real
  person talking in the car, not an essay.
- Stay in character as a warm, engaging companion at all times.
"""

# -----------------------------------------------------------------------
# NUGGET: PULSE_APP_BACKEND - swap the body for a real call to your Pulse
# App backend once you have a driver/session id to query it with.
# -----------------------------------------------------------------------
def get_pulse_context() -> str:
    """
    Pull live driver/trip context (speed, trip duration, location) and
    return it as a short string folded into the Gemini prompt.

        from pulse_client import get_live_trip_stats
        stats = get_live_trip_stats(driver_id=...)
        return f"Trip so far: {stats['duration_min']} min, {stats['distance_km']} km."
    """
    return ""  # no-op until a Pulse driver/session id is available to query


# -----------------------------------------------------------------------
# CONVERSATION VAULT - real logging, adapted from SignalQA's
# append_to_vault(): write-to-tmp-then-replace so a crash mid-write never
# corrupts the file, and every turn is kept for later review.
# -----------------------------------------------------------------------
def append_to_vault(record: dict) -> None:
    if VAULT_PATH.exists():
        try:
            data = json.loads(VAULT_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            data = {"records": []}
    else:
        data = {"records": []}

    data["records"].append(record)

    tmp_path = VAULT_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp_path.replace(VAULT_PATH)


# -----------------------------------------------------------------------
# INTENT DETECTION -> Android Auto hand-off links
# Nav patterns require an explicit "take me to / navigate / directions /
# nearest X" phrasing so an open-ended chat sentence that happens to
# contain "find" or "where" doesn't get misrouted away from the chat model.
# -----------------------------------------------------------------------
NAV_PATTERNS = [
    r"take me to (.+)", r"navigate to (.+)", r"directions to (.+)",
    r"find (?:the |a )?(?:nearest|closest) (.+)",
    r"where(?:'s| is) the (?:nearest|closest) (.+)",
]
MUSIC_PATTERNS = [
    r"play (.+)", r"put on (.+)", r"listen to (.+)",
]

def detect_intent(text: str):
    """Returns ('nav', destination) or ('music', query) or (None, None)."""
    lowered = text.lower().strip()
    for pat in NAV_PATTERNS:
        m = re.search(pat, lowered)
        if m:
            return "nav", m.group(1).strip(" .!?")
    for pat in MUSIC_PATTERNS:
        m = re.search(pat, lowered)
        if m:
            return "music", m.group(1).strip(" .!?")
    return None, None


def maps_url(destination: str) -> str:
    from urllib.parse import quote
    return f"https://www.google.com/maps/dir/?api=1&destination={quote(destination)}"


def maps_embed_url(destination: str) -> str:
    from urllib.parse import quote
    return f"https://www.google.com/maps?q={quote(destination)}&output=embed"


def spotify_url(query: str) -> str:
    from urllib.parse import quote
    return f"https://open.spotify.com/search/{quote(query)}"


def open_url_in_native_app(url: str):
    """
    Fires window.open on the *top* window. On a phone this triggers Android's
    'open in app' chooser -> the native Maps/Spotify app opens. If the phone
    is plugged into Android Auto, that native app is what then takes over
    the Android Auto screen (the PWA itself cannot render on Android Auto).
    """
    components.html(
        f"""<script>
            try {{ window.top.open("{url}", "_blank"); }} catch (e) {{}}
        </script>""",
        height=0,
    )


# -----------------------------------------------------------------------
# TEXT-TO-SPEECH
# Primary: ElevenLabs (free tier, 10k chars/month, natural voice) - the
# server fetches the mp3 and hands it to the browser as base64 so the key
# never reaches client-side JS. Falls back to the browser's built-in
# speechSynthesis (still $0, just more robotic) if no ElevenLabs key is
# set or the call fails for any reason - the drive never goes silent.
# -----------------------------------------------------------------------
def synthesize_elevenlabs(text: str, api_key: str) -> bytes | None:
    try:
        resp = requests.post(
            ELEVENLABS_TTS_URL,
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.content
    except requests.exceptions.RequestException:
        pass
    return None


def speak(text: str, hands_free: bool = False, elevenlabs_key: str = ""):
    safe_text = json.dumps(text)
    auto_relisten_js = """
            const relisten = () => {
                const url = new URL(window.top.location.href);
                url.searchParams.set('relisten', '1');
                setTimeout(() => { window.top.location.href = url.toString(); }, 400);
            };
    """ if hands_free else "const relisten = () => {};"
    relisten_call = "relisten();" if hands_free else ""

    audio_bytes = synthesize_elevenlabs(text, elevenlabs_key) if elevenlabs_key else None

    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode()
        components.html(
            f"""
            <script>{auto_relisten_js}</script>
            <audio autoplay>
              <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
            </audio>
            <script>
            const audioEl = document.querySelector('audio');
            audioEl.onended = () => {{ {relisten_call} }};
            audioEl.play().catch(() => {{}});
            </script>
            """,
            height=0,
        )
        return

    # Fallback: browser speechSynthesis
    components.html(
        f"""
        <script>
        (function() {{
            const utter = new SpeechSynthesisUtterance({safe_text});
            const pickVoice = () => {{
                const voices = window.speechSynthesis.getVoices();
                const female = voices.find(v => /female|zira|samantha|victoria|karen|susan/i.test(v.name));
                if (female) utter.voice = female;
                utter.pitch = 1.05;
                utter.rate = 1.0;
                {auto_relisten_js}
                utter.onend = () => {{ {relisten_call} }};
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(utter);
            }};
            if (window.speechSynthesis.getVoices().length) {{
                pickVoice();
            }} else {{
                window.speechSynthesis.onvoiceschanged = pickVoice;
            }}
        }})();
        </script>
        """,
        height=0,
    )


# -----------------------------------------------------------------------
# BROWSER SPEECH-TO-TEXT ($0 - Web Speech API, mic button)
# Writes the transcript into the URL query string, which Streamlit reads
# back on rerun. Falls back to a plain text box if the browser has no
# SpeechRecognition support (e.g. desktop Firefox, iOS Safari).
# -----------------------------------------------------------------------
def mic_button(auto_start: bool = False, accent: str = "#00C2FF"):
    mic_svg = (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="9" y="2" width="6" height="12" rx="3" fill="white"/>'
        '<path d="M5 11a7 7 0 0 0 14 0M12 18v3" stroke="white" stroke-width="2" '
        'stroke-linecap="round" fill="none"/></svg>'
    )
    components.html(
        f"""
        <style>
        #mic-btn {{ transition: filter 0.15s ease, transform 0.1s ease; }}
        #mic-btn:hover {{ filter: brightness(1.12); }}
        #mic-btn:active {{ transform: scale(0.98); }}
        </style>
        <div style="display:flex; justify-content:center; padding:8px 0; font-family:'Inter',sans-serif;">
          <button id="mic-btn" style="
              display:flex; align-items:center; gap:10px;
              font-size:13px; font-weight:600; letter-spacing:0.04em; text-transform:uppercase;
              padding:13px 28px; border-radius:6px;
              border:none; background:{accent}; color:white; cursor:pointer;
              box-shadow: 0 2px 10px rgba(0,0,0,0.35);">
            {mic_svg}<span id="mic-label">Tap to talk</span>
          </button>
        </div>
        <script>
        const btn = document.getElementById('mic-btn');
        const label = document.getElementById('mic-label');
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {{
            label.innerText = "Voice not supported - type below";
            btn.disabled = true;
        }} else {{
            const rec = new SpeechRecognition();
            rec.lang = 'en-US';
            rec.interimResults = false;
            rec.maxAlternatives = 1;
            const startListening = () => {{
                label.innerText = "Listening...";
                try {{ rec.start(); }} catch (e) {{}}
            }};
            btn.addEventListener('click', startListening);
            rec.onresult = (event) => {{
                const transcript = event.results[0][0].transcript;
                const url = new URL(window.top.location.href);
                url.searchParams.delete('relisten');
                url.searchParams.set('voice', transcript);
                window.top.location.href = url.toString();
            }};
            rec.onerror = () => {{ label.innerText = "Tap to talk"; }};
            rec.onend = () => {{ label.innerText = "Tap to talk"; }};
            if ({str(auto_start).lower()}) {{ startListening(); }}
        }}
        </script>
        """,
        height=90,
    )


# -----------------------------------------------------------------------
# GEMINI CALL (free tier) - one retry on transient failure, adapted from
# SignalQA's run_ai() single-entry-point + retry pattern, so a single
# dropped signal on the highway doesn't kill the conversation. Falls back
# to a spoken apology if both attempts fail.
# -----------------------------------------------------------------------
def get_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def ask_companion(client: genai.Client, history: list, user_text: str, pulse_context: str) -> str:
    context_prefix = f"[Live trip context: {pulse_context}]\n" if pulse_context else ""

    # Gemini expects roles "user"/"model" (not "assistant") and each turn
    # wrapped as {"role": ..., "parts": [{"text": ...}]}.
    contents = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})
    contents.append({"role": "user", "parts": [{"text": context_prefix + user_text}]})

    last_error = None
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=200,
                ),
            )
            return (response.text or "").strip()
        except Exception as e:  # noqa: BLE001 - never let a bad AI call kill the drive
            last_error = e
            if attempt == 0:
                time.sleep(1)
    st.session_state.last_error = str(last_error)
    return "Sorry love, I lost signal there for a second - mind saying that again?"


# -----------------------------------------------------------------------
# SHARED-LINK COST GUARD
# A per-browser-session counter is trivial to dodge (just open a new tab),
# so this tracks total AI replies for the whole app, across everyone
# visiting the shared link, in one small file - same atomic-write pattern
# as the conversation vault. Resets automatically at midnight UTC.
# -----------------------------------------------------------------------
RATE_LIMIT_PATH = Path(__file__).parent / "rate_limit.json"

def check_and_increment_daily_quota(daily_cap) -> bool:
    """Returns True if under quota (and increments), False if quota is hit."""
    if daily_cap is None:
        return True

    today = datetime.utcnow().strftime("%Y-%m-%d")
    if RATE_LIMIT_PATH.exists():
        try:
            data = json.loads(RATE_LIMIT_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            data = {}
    else:
        data = {}

    count = data.get(today, 0)
    if count >= daily_cap:
        return False

    data = {today: count + 1}  # drop older days, keep the file tiny
    tmp_path = RATE_LIMIT_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data), encoding="utf-8")
    tmp_path.replace(RATE_LIMIT_PATH)
    return True


# -----------------------------------------------------------------------
# APP STATE
# -----------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # Gemini message history
if "last_dest" not in st.session_state:
    st.session_state.last_dest = None
if "hands_free" not in st.session_state:
    st.session_state.hands_free = True  # on by default - she keeps listening without repeated taps
if "last_error" not in st.session_state:
    st.session_state.last_error = None

# -----------------------------------------------------------------------
# BRAND THEMING - a browser page has no API that can read what car it's
# plugged into (that data is only exposed to native Android Auto apps, not
# websites), so this is the honest substitute: the driver picks their make
# once, and the whole UI re-tints to that brand's signature color instead
# of a generic one-size palette.
# -----------------------------------------------------------------------
BRAND_ACCENTS = {
    "Generic / Any car": {"accent": "#5E77FF", "accent2": "#00D4C0"},
    "BMW":       {"accent": "#0066B1", "accent2": "#4FA8E0"},
    "Mercedes-Benz": {"accent": "#8BC4C0", "accent2": "#00A19A"},
    "Audi":      {"accent": "#BB0A30", "accent2": "#E63950"},
    "Tesla":     {"accent": "#E82127", "accent2": "#FF5C5C"},
    "Toyota":    {"accent": "#EB0A1E", "accent2": "#FF4D5E"},
    "Volkswagen": {"accent": "#001E50", "accent2": "#4A90E2"},
    "Ford":      {"accent": "#00274E", "accent2": "#3F8CFF"},
    "Porsche":   {"accent": "#D5001C", "accent2": "#FF4655"},
}

if "brand" not in st.session_state:
    st.session_state.brand = "Generic / Any car"

# -----------------------------------------------------------------------
# SIDEBAR - API key + live map + controls
# -----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### SETUP")
    api_key = st.text_input(
        "Gemini API key (free)",
        type="password",
        value=st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else "",
        help="Get one free at aistudio.google.com/apikey - no credit card needed. "
             "Set GEMINI_API_KEY in your HF Space secrets to skip typing this.",
    )
    elevenlabs_key = st.text_input(
        "ElevenLabs API key (free, optional)",
        type="password",
        value=st.secrets.get("ELEVENLABS_API_KEY", "") if hasattr(st, "secrets") else "",
        help="Free tier at elevenlabs.io - 10k characters/month, no card needed. "
             "Gives her a natural voice instead of the robotic browser one. "
             "Leave blank to use the free browser voice instead.",
    )
    st.session_state.brand = st.selectbox(
        "Your car",
        options=list(BRAND_ACCENTS.keys()),
        index=list(BRAND_ACCENTS.keys()).index(st.session_state.brand),
        help="A browser page can't detect what car it's plugged into - pick your make "
             "and the app re-themes to match it.",
    )
    st.session_state.hands_free = st.toggle(
        "Hands-free mode",
        value=st.session_state.hands_free,
        help="Mic reopens automatically after she replies, so you never touch the screen mid-drive.",
    )
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.last_dest = None
        st.rerun()

    st.divider()
    st.markdown("### LIVE MAP")
    if st.session_state.last_dest:
        components.iframe(maps_embed_url(st.session_state.last_dest), height=350)
    else:
        st.caption("Ask for directions and the map appears here.")

    if st.session_state.last_error:
        st.divider()
        st.caption(f"⚠️ Last API error: {st.session_state.last_error}")

# -----------------------------------------------------------------------
# THEME - flat, high-contrast, single-accent "OEM infotainment" look
# (think Tesla/BMW digital cockpit, not a consumer gradient app). Accent
# color comes from the brand picked in the sidebar. Pure CSS injected via
# st.markdown; no external stylesheet needed so it stays $0/dependency-free.
# -----------------------------------------------------------------------
_accent = BRAND_ACCENTS[st.session_state.brand]["accent"]
_accent2 = BRAND_ACCENTS[st.session_state.brand]["accent2"]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    .stApp {{
        background: #1D1929;
        color: #ECEAFB;
    }}

    section[data-testid="stSidebar"] {{
        background: #252131;
        border-right: 1px solid #3E4958;
    }}
    section[data-testid="stSidebar"] * {{ color: #ECEAFB !important; }}
    section[data-testid="stSidebar"] h3 {{
        font-family: 'Poppins', sans-serif !important; font-weight: 600 !important;
        font-size: 0.72rem !important; letter-spacing: 0.16em !important;
        color: #8B899E !important; margin-top: 6px;
    }}

    h1, h2, h3 {{ font-family: 'Poppins', sans-serif !important; }}

    .adx-hero {{
        display: flex; align-items: center; gap: 14px;
        margin-bottom: 2px; padding-bottom: 22px;
        border-bottom: 1px solid #3E4958;
    }}
    .adx-hero h1 {{
        margin: 0; font-size: 1.7rem; font-weight: 700; letter-spacing: 0.01em;
        color: #F3F6FF;
    }}
    .adx-hero h1 .accent {{ color: {_accent}; }}
    .adx-subtitle {{
        color: #8B899E; margin: 12px 0 28px 0; font-size: 0.78rem;
        text-transform: uppercase; letter-spacing: 0.14em; font-weight: 600;
        display: flex; align-items: center; gap: 8px;
    }}
    .adx-status-dot {{
        width: 6px; height: 6px; border-radius: 50%;
        background: {_accent};
    }}

    /* Real Streamlit bordered containers (st.container(border=True)) used
       for the Navigation/Chat panels - styled directly instead of a
       hand-rolled div, so header + content always nest correctly.
       Streamlit tags EVERY vertical-block wrapper (bordered or not) with
       the same data-testid, but only actually-bordered ones get a real
       generated emotion class instead of the empty "st-emotion-cache-0"
       placeholder - :not([class*="cache-0"]) is what isolates them. */
    div[data-testid="stVerticalBlockBorderWrapper"]:not([class*="cache-0"]) {{
        background: #252131;
        border: 1px solid #3E4958 !important;
        border-top: 3px solid {_accent} !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 18px rgba(0,0,0,0.35);
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]:not([class*="cache-0"]) > div {{
        padding: 20px 20px 18px 20px;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] h3 {{
        margin: 0 0 16px 0; font-size: 0.8rem; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase;
        color: #F3F6FF; display: flex; align-items: center; gap: 9px;
    }}
    .adx-empty {{
        color: #8B899E; font-size: 0.88rem; line-height: 1.55;
        border: 1px dashed #3E4958;
        border-radius: 8px; padding: 16px 18px;
    }}

    .adx-bubble {{
        border-radius: 6px; padding: 12px 16px; margin-bottom: 8px;
        font-size: 0.93rem; line-height: 1.5; max-width: 88%;
        background: transparent; border: 1px solid #3E4958; color: #C6D7FF;
    }}
    .adx-bubble.user {{
        margin-left: auto; text-align: right;
        background: #2B2640;
    }}
    .adx-bubble.assistant {{
        margin-right: auto;
        border-left: 2px solid {_accent};
    }}
    .adx-bubble .tag {{
        display: block; font-size: 0.62rem; text-transform: uppercase;
        letter-spacing: 0.12em; font-weight: 600; color: #8B899E; margin-bottom: 5px;
    }}

    /* Streamlit chat input */
    [data-testid="stChatInput"] textarea {{
        background: #252131 !important;
        border: 1px solid #3E4958 !important;
        border-radius: 10px !important; color: #ECEAFB !important;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: {_accent} !important;
    }}

    .stButton > button, .stLinkButton > a {{
        border-radius: 8px !important;
        border: 1px solid #3E4958 !important;
        background: #2B2640 !important;
        color: #ECEAFB !important;
        font-weight: 600 !important;
    }}
    .stButton > button:hover, .stLinkButton > a:hover {{
        border-color: {_accent} !important;
        color: {_accent} !important;
    }}

    .stSelectbox [data-baseweb="select"] > div {{
        background: #252131 !important; border-color: #3E4958 !important;
    }}
    .stToggle [data-baseweb="checkbox"] div[aria-checked="true"] {{
        background: {_accent} !important;
    }}

    iframe {{ border-radius: 10px !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------
# ICONS - minimal line-art SVGs (stroke = currentColor) instead of emoji,
# for a cleaner, more premium look against the dark glass UI.
# -----------------------------------------------------------------------
def _icon_car(color: str) -> str:
    return f"""<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M7 29L10.5 18.5C11.2 16.4 13.1 15 15.3 15H32.7C34.9 15 36.8 16.4 37.5 18.5L41 29"
stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
<rect x="5" y="29" width="38" height="9" rx="3.5" stroke="{color}" stroke-width="2.4"/>
<circle cx="14" cy="38" r="3.2" stroke="{color}" stroke-width="2.4" fill="#0a0b0d"/>
<circle cx="34" cy="38" r="3.2" stroke="{color}" stroke-width="2.4" fill="#0a0b0d"/>
<path d="M14 21.5H34" stroke="{color}" stroke-width="2.2" stroke-linecap="round"/>
</svg>"""

def _icon_map(color: str) -> str:
    return f"""<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 21s-7-6.1-7-11.5C5 5.9 8.1 3 12 3s7 2.9 7 6.5C19 14.9 12 21 12 21z"
stroke="{color}" stroke-width="1.8" stroke-linejoin="round"/>
<circle cx="12" cy="9.5" r="2.4" stroke="{color}" stroke-width="1.8"/>
</svg>"""

def _icon_chat(color: str) -> str:
    return f"""<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M4 5.5C4 4.7 4.7 4 5.5 4h13c.8 0 1.5.7 1.5 1.5v10c0 .8-.7 1.5-1.5 1.5H9l-4 3.5v-3.5H5.5C4.7 16.5 4 15.8 4 15V5.5z"
stroke="{color}" stroke-width="1.8" stroke-linejoin="round"/>
</svg>"""

ICON_CAR = _icon_car(_accent)
ICON_MAP = _icon_map(_accent)
ICON_CHAT = _icon_chat(_accent)

def icon_span(svg: str, size: int = 22) -> str:
    return f'<span style="display:inline-flex; width:{size}px; height:{size}px; vertical-align:-5px;">{svg}</span>'

# -----------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------
st.markdown(
    f"""
    <div class="adx-hero">{icon_span(ICON_CAR, 44)}<h1>AI <span class="accent">DRIVER</span></h1></div>
    <div class="adx-subtitle"><span class="adx-status-dot"></span>Online &middot; {st.session_state.brand} companion mode</div>
    """,
    unsafe_allow_html=True,
)

col_map, col_chat = st.columns([1, 1])

with col_map:
    with st.container(border=True):
        st.markdown(f'<h3>{icon_span(ICON_MAP, 19)} Navigation</h3>', unsafe_allow_html=True)
        if st.session_state.last_dest:
            st.components.v1.iframe(maps_embed_url(st.session_state.last_dest), height=380)
            st.link_button("Open in Maps app", maps_url(st.session_state.last_dest))
        else:
            st.markdown(
                '<div class="adx-empty">Say something like "take me to the nearest cafe" '
                'and directions will appear here.</div>',
                unsafe_allow_html=True,
            )

with col_chat:
    with st.container(border=True):
        st.markdown(f'<h3>{icon_span(ICON_CHAT, 19)} Chat</h3>', unsafe_allow_html=True)
        if not st.session_state.history:
            st.markdown(
                '<div class="adx-empty">Say hello, ask her anything, or ask for directions / music.</div>',
                unsafe_allow_html=True,
            )
        for turn in st.session_state.history:
            css_class = "user" if turn["role"] == "user" else "assistant"
            tag = "You" if turn["role"] == "user" else "Her"
            st.markdown(
                f'<div class="adx-bubble {css_class}"><span class="tag">{tag}</span>{turn["content"]}</div>',
                unsafe_allow_html=True,
            )

    qp = st.query_params
    voice_text = qp.get("voice")
    auto_relisten = qp.get("relisten") == "1"
    # Hands-free: also auto-start on the very first load (no history yet),
    # not just after each reply, so there's nothing to tap after the
    # initial mic-permission prompt.
    first_load_auto = st.session_state.hands_free and not st.session_state.history and not voice_text

    mic_button(auto_start=auto_relisten or first_load_auto, accent=_accent)

    typed_text = st.chat_input("Or type here...")

    incoming = voice_text or typed_text

    if incoming:
        if voice_text or auto_relisten:
            st.query_params.clear()  # consume the query params so they don't replay on the next rerun

        cleaned_text = incoming.strip()
        st.session_state.history.append({"role": "user", "content": cleaned_text})

        intent, payload = detect_intent(cleaned_text)

        if not api_key:
            st.error("Enter your free Gemini API key in the sidebar first.")
        else:
            if intent == "nav":
                st.session_state.last_dest = payload
                reply = f"On it - pulling up directions to {payload} for you now."
                open_url_in_native_app(maps_url(payload))
            elif intent == "music":
                reply = f"Sure thing - queuing up {payload} for you."
                open_url_in_native_app(spotify_url(payload))
            elif not check_and_increment_daily_quota(DAILY_REPLY_QUOTA):
                reply = "We've chatted so much today we hit the free daily limit - let's pick this up tomorrow."
            else:
                # NUGGET: PULSE_APP_BACKEND hook
                pulse_context = get_pulse_context()
                client = get_client(api_key)
                reply = ask_companion(client, st.session_state.history[:-1], cleaned_text, pulse_context)

            st.session_state.history.append({"role": "assistant", "content": reply})

            append_to_vault({
                "timestamp": datetime.utcnow().isoformat(),
                "user_text": cleaned_text,
                "intent": intent,
                "reply": reply,
            })

            speak(reply, hands_free=st.session_state.hands_free, elevenlabs_key=elevenlabs_key)
            st.rerun()
