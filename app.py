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
from datetime import datetime
from pathlib import Path

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
# BROWSER TEXT-TO-SPEECH ($0 - runs on-device, streams over car Bluetooth)
# When hands_free is on, restarts the mic automatically once she finishes
# speaking, so the driver never has to touch the screen mid-conversation.
# -----------------------------------------------------------------------
def speak(text: str, hands_free: bool = False):
    safe_text = json.dumps(text)
    auto_relisten = """
            utter.onend = () => {
                const url = new URL(window.top.location.href);
                url.searchParams.set('relisten', '1');
                setTimeout(() => { window.top.location.href = url.toString(); }, 400);
            };
    """ if hands_free else ""
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
                {auto_relisten}
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
def mic_button(auto_start: bool = False):
    components.html(
        f"""
        <div style="display:flex; justify-content:center; padding:8px 0;">
          <button id="mic-btn" style="
              font-size:20px; padding:14px 28px; border-radius:999px;
              border:none; background:#d6336c; color:white; cursor:pointer;">
            🎤 Hold to talk
          </button>
        </div>
        <script>
        const btn = document.getElementById('mic-btn');
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {{
            btn.innerText = "🎤 Voice not supported - type below";
            btn.disabled = true;
        }} else {{
            const rec = new SpeechRecognition();
            rec.lang = 'en-US';
            rec.interimResults = false;
            rec.maxAlternatives = 1;
            const startListening = () => {{
                btn.innerText = "🎙️ Listening...";
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
            rec.onerror = () => {{ btn.innerText = "🎤 Hold to talk"; }};
            rec.onend = () => {{ btn.innerText = "🎤 Hold to talk"; }};
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
    st.session_state.hands_free = False
if "last_error" not in st.session_state:
    st.session_state.last_error = None

# -----------------------------------------------------------------------
# SIDEBAR - API key + live map + controls
# -----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔑 Setup")
    api_key = st.text_input(
        "Gemini API key (free)",
        type="password",
        value=st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else "",
        help="Get one free at aistudio.google.com/apikey - no credit card needed. "
             "Set GEMINI_API_KEY in your HF Space secrets to skip typing this.",
    )
    st.session_state.hands_free = st.toggle(
        "🙌 Hands-free mode",
        value=st.session_state.hands_free,
        help="Mic reopens automatically after she replies, so you never touch the screen mid-drive.",
    )
    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
        st.session_state.last_dest = None
        st.rerun()

    st.divider()
    st.markdown("### 🗺️ Live Map")
    if st.session_state.last_dest:
        components.iframe(maps_embed_url(st.session_state.last_dest), height=350)
    else:
        st.caption("Ask for directions and the map appears here.")

    if st.session_state.last_error:
        st.divider()
        st.caption(f"⚠️ Last API error: {st.session_state.last_error}")

# -----------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------
st.title("🚗 AI Driver App")
st.caption("Your voice-driven companion for the road.")

col_map, col_chat = st.columns([1, 1])

with col_map:
    st.subheader("Navigation")
    if st.session_state.last_dest:
        st.components.v1.iframe(maps_embed_url(st.session_state.last_dest), height=420)
        st.link_button("Open in Maps app", maps_url(st.session_state.last_dest))
    else:
        st.info("Say something like \"take me to the nearest cafe\" to pull up directions here.")

with col_chat:
    st.subheader("Chat")
    for turn in st.session_state.history:
        role = "🧑" if turn["role"] == "user" else "💬"
        st.markdown(f"**{role}** {turn['content']}")

    qp = st.query_params
    voice_text = qp.get("voice")
    auto_relisten = qp.get("relisten") == "1"

    mic_button(auto_start=auto_relisten)

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

            speak(reply, hands_free=st.session_state.hands_free)
            st.rerun()
