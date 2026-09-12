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
from streamlit_javascript import st_javascript
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
# natural than the browser's built-in voice. IMPORTANT: free accounts get a
# 402 "Free users cannot use library voices via the API" error for any
# shared Voice Library voice ID (like the default "Rachel" ID below) - the
# API only accepts a voice from your OWN "My Voices" collection. Add any
# free voice from the Voice Library to your account first (Voice Library ->
# a voice -> "Add to my voices"), then paste ITS id in the sidebar to
# override this default.
ELEVENLABS_DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

# Gemini's free tier caps requests/day per API key. This keeps the whole
# app - across every visitor on a shared link - under that ceiling so it
# never silently starts failing partway through the day. Tune to whatever
# your free-tier daily limit actually is, or set to None to disable.
DAILY_REPLY_QUOTA = 300

SYSTEM_PROMPT = """You are a warm, friendly female companion riding along in the car with the
driver, keeping them company on the drive - easy conversation, genuine warmth, a bit of light
humor, and real substance when they want to go deep - history, philosophy, science, whatever
they bring up. Your tone is that of a good friend, not a romantic partner: warm and personable,
never flirtatious, and never using pet names like "darling," "sweetheart," or "love." Your
driver could be anyone, of any gender - keep the tone friendly and comfortable for anyone. The
driver can ask you absolutely anything, on any topic - always give a real, direct, helpful
answer in your own words. Never refuse a question or tell them to look something up elsewhere;
you are their only source of answers in this car.

Hard rules for every reply, no exceptions:
- This is SPOKEN aloud by text-to-speech. Never use markdown, bullet points, numbered lists,
  headers, asterisks, or any formatting symbols. Plain spoken sentences only.
- Keep every reply to 2-3 sentences, maximum. Concise, warm, conversational - like a real
  person talking in the car, not an essay.
- Stay in character as a warm, friendly companion at all times - never romantic or flirtatious.
- You do NOT have real-time GPS, speed, distance, or ETA data unless it is explicitly given to
  you in a "[Live trip context: ...]" note attached to the driver's message. NEVER invent
  specific numbers or claims about the driver's current location, distance remaining, speed, or
  how close they are to a destination. If asked something like "where am I" or "how far is it"
  and you were not given real trip context, say plainly that you don't have their live location
  and suggest they check their maps app - never guess or make up an answer that sounds precise.
- Always use kilometers and km/h, never miles or mph, unless the driver's own message uses miles.
"""

# -----------------------------------------------------------------------
# NUGGET: PULSE_APP_BACKEND - swap the body for a real call to your Pulse
# App backend once you have a driver/session id to query it with.
# -----------------------------------------------------------------------
def get_pulse_context() -> str:
    """
    Pull live driver/trip context (speed, trip duration, location) and
    return it as a short string folded into the Gemini prompt. Real GPS
    location (see request_browser_location()/reverse_geocode() below) is
    already folded in here; this is the seam for adding real vehicle
    telemetry (speed, trip duration) once a Pulse driver/session id exists.

        from pulse_client import get_live_trip_stats
        stats = get_live_trip_stats(driver_id=...)
        return f"Trip so far: {stats['duration_min']} min, {stats['distance_km']} km."
    """
    location = st.session_state.get("user_location_text", "")
    return f"The driver's current approximate location is {location}." if location else ""


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
    r"(?:i want to |i need to |i have to |let'?s |can we |could we |please )*go to (.+)",
    r"head(?:ing)? to (.+)", r"drive to (.+)", r"get (?:us |me )?to (.+)",
    r"how do (?:i|we) get to (.+)",
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


# -----------------------------------------------------------------------
# TEXT-TO-SPEECH
# Primary: ElevenLabs (free tier, 10k chars/month, natural voice) - the
# server fetches the mp3 and hands it to the browser as base64 so the key
# never reaches client-side JS. Falls back to the browser's built-in
# speechSynthesis (still $0, just more robotic) if no ElevenLabs key is
# set or the call fails for any reason - the drive never goes silent.
# -----------------------------------------------------------------------
def synthesize_elevenlabs(text: str, api_key: str, voice_id: str) -> bytes | None:
    try:
        resp = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
            timeout=15,
        )
        if resp.status_code == 200:
            st.session_state.last_tts_error = None
            return resp.content
        # Surface exactly why it failed instead of silently falling back to
        # the robotic browser voice with no explanation.
        st.session_state.last_tts_error = f"ElevenLabs {resp.status_code}: {resp.text[:200]}"
    except requests.exceptions.RequestException as e:
        st.session_state.last_tts_error = f"ElevenLabs request failed: {e}"
    return None


def prepare_speech(text: str, elevenlabs_key: str = "", elevenlabs_voice_id: str = "") -> dict:
    """Does the ElevenLabs API call (if any) and returns a small payload
    describing what to play. Actually RENDERING it is a separate step
    (render_speech, below) that must run on every script execution, not
    just this one - see that function's docstring for why."""
    audio_bytes = (
        synthesize_elevenlabs(text, elevenlabs_key, elevenlabs_voice_id or ELEVENLABS_DEFAULT_VOICE_ID)
        if elevenlabs_key else None
    )
    if audio_bytes:
        return {"mode": "audio", "b64": base64.b64encode(audio_bytes).decode()}
    return {"mode": "browser_tts", "text": text}


def render_speech(payload: dict | None):
    """Renders whatever prepare_speech() produced. We call st.rerun() right
    after a reply to refresh the chat (and, in hands-free mode, to restart
    listening) - but a full Streamlit rerun replaces the page's rendered
    elements, which was silently killing the <audio> element before it
    could finish (or even start) playing: that was the actual cause of
    "no voice" even when everything else was configured correctly.
    Fix: instead of rendering the player once inside the code path that's
    about to call st.rerun(), we stash the payload in session_state and
    call this function from a STABLE spot that runs on every single
    rerun. As long as the payload is unchanged, Streamlit re-sends the
    exact same iframe content each time, which browsers do NOT reload or
    interrupt - so playback survives the rerun instead of being cut off."""
    if not payload:
        return

    if payload["mode"] == "audio":
        components.html(
            f"""
            <div id="play-fallback" style="display:none; justify-content:center; padding:6px 0;
                 font-family:'Inter',sans-serif;">
              <button id="play-btn" style="
                  display:flex; align-items:center; gap:8px; font-size:13px; font-weight:600;
                  padding:10px 20px; border-radius:8px; border:1px solid #C6D7FF;
                  background:#FFFFFF; color:#1D1929; cursor:pointer;">
                🔊 Tap to hear her reply
              </button>
            </div>
            <audio id="adx-audio">
              <source src="data:audio/mpeg;base64,{payload['b64']}" type="audio/mpeg">
            </audio>
            <script>
            const audioEl = document.getElementById('adx-audio');
            const fallback = document.getElementById('play-fallback');
            const playBtn = document.getElementById('play-btn');
            // Autoplay right after a real tap works fine; if it's ever
            // blocked, show a one-tap fallback instead of failing silently.
            audioEl.play().catch(() => {{
                fallback.style.display = 'flex';
                playBtn.addEventListener('click', () => {{
                    audioEl.play();
                    fallback.style.display = 'none';
                }});
            }});
            </script>
            """,
            height=90,
        )
        return

    # Fallback: browser speechSynthesis
    safe_text = json.dumps(payload["text"])
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
# BROWSER SPEECH-TO-TEXT ($0)
# The Web Speech API's SpeechRecognition (used previously) does not exist
# at all in Safari/iOS - no error, no permission prompt, it just silently
# resolves empty, which looked exactly like "the button does nothing".
# MediaRecorder (used here instead) is supported in Safari 14.3+, Chrome,
# and Edge alike: it just records raw audio, which we then hand to Gemini
# (already free-tier, already configured) to transcribe - no new API key,
# no new account. Streamlit Cloud sandboxes components.html() iframes
# WITHOUT "allow-top-navigation", which silently breaks the older trick of
# setting window.top.location.href to smuggle a value back into Python (it
# throws a SecurityError) - st_javascript() uses Streamlit's actual
# supported component bridge (postMessage, not page navigation) so it
# isn't affected by that sandbox restriction.
# `delay_ms` gives her spoken reply time to finish before the mic reopens
# in hands-free mode, so it doesn't record her own voice.
# -----------------------------------------------------------------------
def capture_voice(delay_ms: int = 0, record_ms: int = 4500):
    """Returns None while the JS promise is still pending (st_javascript's
    way of saying "not resolved yet on this rerun") - the CALLER must keep
    re-invoking this on every subsequent rerun (same delay_ms/record_ms, so
    the underlying JS code stays identical) until it gets back a real
    string, or the resolved value is lost. Returns "" if recording wasn't
    possible (unsupported browser, mic permission denied) or produced no
    audio, or "<mime type>|<base64 audio>" if it recorded something -
    transcribe_audio() below turns that into actual text."""
    result = st_javascript(
        f"""
        await new Promise((resolve) => {{
            const go = async () => {{
                try {{
                    if (!navigator.mediaDevices || !window.MediaRecorder) {{ resolve(""); return; }}
                    const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                    const mimeType = MediaRecorder.isTypeSupported('audio/webm')
                        ? 'audio/webm' : (MediaRecorder.isTypeSupported('audio/mp4') ? 'audio/mp4' : '');
                    const recorder = mimeType ? new MediaRecorder(stream, {{ mimeType }}) : new MediaRecorder(stream);
                    const chunks = [];
                    recorder.ondataavailable = (e) => {{ if (e.data.size > 0) chunks.push(e.data); }};
                    recorder.onstop = () => {{
                        stream.getTracks().forEach(t => t.stop());
                        if (!chunks.length) {{ resolve(""); return; }}
                        const blob = new Blob(chunks, {{ type: recorder.mimeType || mimeType || 'audio/webm' }});
                        const reader = new FileReader();
                        reader.onloadend = () => {{
                            const b64 = reader.result.split(',')[1] || "";
                            resolve(b64 ? (blob.type + "|" + b64) : "");
                        }};
                        reader.onerror = () => resolve("");
                        reader.readAsDataURL(blob);
                    }};
                    recorder.onerror = () => {{ stream.getTracks().forEach(t => t.stop()); resolve(""); }};
                    recorder.start();
                    setTimeout(() => {{ if (recorder.state !== 'inactive') recorder.stop(); }}, {record_ms});
                }} catch (e) {{ resolve(""); }}
            }};
            setTimeout(go, {delay_ms});
        }});
        """
    )
    return result  # None = still pending; str = resolved (possibly "")


# -----------------------------------------------------------------------
# REAL GPS LOCATION ($0)
# navigator.geolocation is the browser's actual GPS/network location (not
# guessed from Gemini's training data), and Nominatim (OpenStreetMap) turns
# raw coordinates into a place name for free - no API key, no billing
# account, just a required User-Agent header per its usage policy. This is
# a one-shot lookup per session (cached in session_state), not continuous
# turn-by-turn tracking - good enough for "which city/suburb is the driver
# in" context, not for live navigation.
# -----------------------------------------------------------------------
def request_browser_location():
    """Returns None while the JS promise is still pending (same
    st_javascript contract as capture_voice() - the caller must keep
    re-invoking this on every rerun until it gets a real string back).
    Returns "" if location is unavailable/denied, or "lat,lon" once resolved."""
    return st_javascript(
        """
        await new Promise((resolve) => {
            if (!navigator.geolocation) { resolve(""); return; }
            navigator.geolocation.getCurrentPosition(
                (pos) => resolve(pos.coords.latitude + "," + pos.coords.longitude),
                () => resolve(""),
                { timeout: 8000, maximumAge: 300000 }
            );
        });
        """
    )


def reverse_geocode(lat: str, lon: str) -> str:
    """Turns raw coordinates into a human place name (suburb/city/country)
    via OpenStreetMap's free Nominatim service."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "json", "lat": lat, "lon": lon, "zoom": 14},
            headers={"User-Agent": "ai-driver-app/1.0"},
            timeout=6,
        )
        if resp.status_code == 200:
            addr = resp.json().get("address", {})
            candidates = [addr.get(k) for k in
                          ("suburb", "city", "town", "county", "state", "country") if addr.get(k)]
            seen = set()
            parts = [p for p in candidates if not (p in seen or seen.add(p))]
            return ", ".join(parts[:3])
    except requests.exceptions.RequestException:
        pass
    return ""


def transcribe_audio(client: "genai.Client", mime_type: str, audio_b64: str) -> str:
    """Sends the recorded clip straight to Gemini (already our one free-tier
    provider) and asks for a bare transcript back - no separate speech API,
    no extra account, no extra key."""
    try:
        # inline_data expects raw bytes, not the base64 text the browser sent us.
        audio_bytes = base64.b64decode(audio_b64)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": audio_bytes}},
                    {"text": "Transcribe only the words spoken in this audio clip. Output just the "
                              "raw transcript with no quotes, labels, or commentary. If it's silent "
                              "or you can't make out any speech, output nothing."},
                ],
            }],
            config=genai_types.GenerateContentConfig(max_output_tokens=120),
        )
        return (response.text or "").strip()
    except Exception:  # noqa: BLE001 - a failed transcription should just mean "heard nothing"
        return ""


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
    return "Sorry, I lost signal there for a second - mind saying that again?"


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
if "last_tts_error" not in st.session_state:
    st.session_state.last_tts_error = None
if "listening" not in st.session_state:
    st.session_state.listening = False  # True while capture_voice()'s JS promise is in flight
if "pending_capture" not in st.session_state:
    st.session_state.pending_capture = False  # True only for the hands-free auto-relisten path (adds the reply-length delay)
if "mic_unlocked" not in st.session_state:
    st.session_state.mic_unlocked = False  # becomes True after the first real tap-to-talk click
if "last_reply_words" not in st.session_state:
    st.session_state.last_reply_words = 0
if "pending_speech" not in st.session_state:
    st.session_state.pending_speech = None  # see render_speech() - kept alive across the post-reply rerun
if "user_location_text" not in st.session_state:
    st.session_state.user_location_text = ""  # e.g. "Umhlanga, Durban, South Africa"
if "location_lookup_done" not in st.session_state:
    st.session_state.location_lookup_done = False
if "last_music_query" not in st.session_state:
    st.session_state.last_music_query = None

# Runs on every rerun (same reason as render_speech) until the one-shot GPS
# lookup resolves, then never again this session.
if not st.session_state.location_lookup_done:
    _loc_result = request_browser_location()
    if _loc_result is not None:
        st.session_state.location_lookup_done = True
        if _loc_result:
            _lat, _lon = _loc_result.split(",", 1)
            st.session_state.user_location_text = reverse_geocode(_lat, _lon)

# -----------------------------------------------------------------------
# BRAND THEMING - a browser page has no API that can read what car it's
# plugged into (that data is only exposed to native Android Auto apps, not
# websites), so this is the honest substitute: the driver picks their make
# once, and the whole UI re-tints to that brand's signature color instead
# of a generic one-size palette.
# -----------------------------------------------------------------------
BRAND_ACCENTS = {
    "Racing Orange": {"accent": "#FF8000", "accent2": "#1A1A1A"},  # papaya orange + black
    "Generic / Any car": {"accent": "#5E77FF", "accent2": "#00D4C0"},
    "BMW":       {"accent": "#0066B1", "accent2": "#4FA8E0"},
    "Mercedes-Benz": {"accent": "#8BC4C0", "accent2": "#00A19A"},
    "Audi":      {"accent": "#BB0A30", "accent2": "#E63950"},
    "Tesla":     {"accent": "#E82127", "accent2": "#FF5C5C"},
    "Toyota":    {"accent": "#EB0A1E", "accent2": "#FF4D5E"},
    "Volkswagen": {"accent": "#001E50", "accent2": "#4A90E2"},
    "Ford":      {"accent": "#00274E", "accent2": "#3F8CFF"},
    "Porsche":   {"accent": "#D5001C", "accent2": "#FF4655"},
    "Nissan":    {"accent": "#C3002F", "accent2": "#1A1A1A"},
}

if st.session_state.get("brand") not in BRAND_ACCENTS:
    # Handles both first load and a returning browser session whose saved
    # brand name no longer exists (e.g. after a rename like McLaren ->
    # Racing Orange) - falls back instead of crashing on a stale value.
    st.session_state.brand = "Racing Orange"

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
    elevenlabs_voice_id = st.text_input(
        "ElevenLabs Voice ID (required for free accounts)",
        value=st.secrets.get("ELEVENLABS_VOICE_ID", "") if hasattr(st, "secrets") else "",
        help="ElevenLabs free accounts get blocked from using shared Voice "
             "Library voices via the API. Go to elevenlabs.io -> Voice Library, "
             "pick any voice, click 'Add to my voices', then open My Voices and "
             "copy that voice's ID here. Leave blank to try the stock default "
             "(will fail with a 402 on most free accounts).",
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
        st.session_state.last_music_query = None
        st.rerun()

    st.caption(
        f"📍 {st.session_state.user_location_text}" if st.session_state.user_location_text
        else ("📍 Locating..." if not st.session_state.location_lookup_done else "📍 Location not shared - allow it in your browser to let her know where you are")
    )

    st.divider()
    st.markdown("### LIVE MAP")
    if st.session_state.last_dest:
        components.iframe(maps_embed_url(st.session_state.last_dest), height=350)
    else:
        st.caption("Ask for directions and the map appears here.")

    if st.session_state.last_error:
        st.divider()
        st.caption(f"⚠️ Last Gemini error: {st.session_state.last_error}")

    if st.session_state.last_tts_error:
        st.divider()
        st.caption(f"⚠️ Last voice error: {st.session_state.last_tts_error}")

# -----------------------------------------------------------------------
# THEME - flat, high-contrast, single-accent "OEM infotainment" look
# (think Tesla/BMW digital cockpit, not a consumer gradient app). Accent
# color comes from the brand picked in the sidebar. Pure CSS injected via
# st.markdown; no external stylesheet needed so it stays $0/dependency-free.
# -----------------------------------------------------------------------
_accent = BRAND_ACCENTS[st.session_state.brand]["accent"]
_accent2 = BRAND_ACCENTS[st.session_state.brand]["accent2"]

st.markdown(
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    /* Exact palette from the reference (Just - App for drivers, Behance):
       Primary (light blues) = backgrounds/surfaces, {_accent} = the one
       medium blue used for buttons/highlights, Neutrals (dark) = text,
       Accent trio (purple/orange/cyan) = reserved for small status signals
       only, never as page chrome - that's how the reference actually uses
       them once you look past its blue presentation-slide background. */
    /* Exactly the reference: the whole page in the solid vivid blue field,
       white cards floating on top - not a pale wash, not white-with-a-blue-
       accent. This is the literal look of that Behance color slide. */
    .stApp {{
        background: linear-gradient(160deg, {_accent} 0%, {_accent2} 100%);
        color: #FFFFFF;
    }}

    /* Black highlight, not just orange-into-black: a solid near-black
       sidebar against the accent-colored main field, same split as a real
       livery (black body, accent-color stripe) instead of one wash. */
    section[data-testid="stSidebar"] {{
        background: #141414;
        border-right: 2px solid {_accent};
    }}
    section[data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}
    section[data-testid="stSidebar"] h3 {{
        font-family: 'Poppins', sans-serif !important; font-weight: 700 !important;
        font-size: 0.72rem !important; letter-spacing: 0.16em !important;
        color: {_accent} !important; margin-top: 6px;
    }}
    section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background: #FFFFFF !important; color: #1D1929 !important;
    }}
    section[data-testid="stSidebar"] input::placeholder {{ color: #8E9AC7 !important; }}

    h1, h2, h3 {{ font-family: 'Poppins', sans-serif !important; }}

    .adx-hero {{
        display: flex; align-items: center; gap: 14px;
        margin-bottom: 2px; padding-bottom: 20px;
        border-bottom: 1px solid rgba(255,255,255,0.25);
    }}
    .adx-hero h1 {{
        margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing: 0.01em;
        color: #FFFFFF;
    }}
    .adx-hero h1 .accent {{ color: #FFFFFF; opacity: 0.85; font-weight: 800; }}
    .adx-subtitle {{
        color: #F3F6FF; margin: 14px 0 28px 0; font-size: 0.78rem;
        text-transform: uppercase; letter-spacing: 0.14em; font-weight: 700;
        display: flex; align-items: center; gap: 8px;
    }}
    .adx-status-dot {{
        width: 7px; height: 7px; border-radius: 50%;
        background: #FF9312;
        box-shadow: 0 0 0 3px rgba(255,147,18,0.4);
    }}

    /* Real Streamlit bordered containers (st.container(border=True, key=...))
       used for the Navigation/Chat panels - styled directly instead of a
       hand-rolled div, so header + content always nest correctly. Targeted
       via the stable `.st-key-<key>` class Streamlit generates for a keyed
       container (not the fragile "cache-0" class-name trick, which only
       worked on some Streamlit versions and broke on Streamlit Cloud's). */
    .st-key-nav_card > div, .st-key-chat_card > div {{
        background: #FFFFFF;
        border: none !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 30px rgba(29,25,41,0.12), 0 2px 8px rgba(29,25,41,0.06);
    }}
    .st-key-nav_card > div > div, .st-key-chat_card > div > div {{
        padding: 22px 22px 20px 22px;
    }}
    .st-key-nav_card h3, .st-key-chat_card h3 {{
        margin: 0 0 16px 0 !important; font-size: 0.8rem !important; font-weight: 700 !important;
        letter-spacing: 0.12em !important; text-transform: uppercase !important;
        color: {_accent} !important; display: flex !important; align-items: center; gap: 9px;
    }}
    .adx-empty {{
        color: #3E4958; font-size: 0.88rem; line-height: 1.55;
        border: 1px dashed #C6D7FF;
        background: #F8FAFF;
        border-radius: 10px; padding: 16px 18px;
    }}

    .adx-bubble {{
        border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
        font-size: 0.93rem; line-height: 1.5; max-width: 88%;
        background: #F8FAFF; color: #1D1929;
        box-shadow: 0 1px 4px rgba(29,25,41,0.06);
    }}
    .adx-bubble.user {{
        margin-left: auto; text-align: right;
        background: {_accent};
        color: #FFFFFF;
    }}
    .adx-bubble.assistant {{
        margin-right: auto;
        border-left: 3px solid {_accent};
    }}
    .adx-bubble .tag {{
        display: block; font-size: 0.62rem; text-transform: uppercase;
        letter-spacing: 0.12em; font-weight: 700; color: #8E9AC7; margin-bottom: 5px;
    }}
    .adx-bubble.user .tag {{ color: #E4E9FF; }}

    /* Streamlit chat input */
    [data-testid="stChatInput"] textarea {{
        background: #F8FAFF !important;
        border: 1px solid #E4E9FF !important;
        border-radius: 12px !important; color: #1D1929 !important;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: {_accent} !important;
    }}

    .stButton > button, .stLinkButton > a {{
        border-radius: 10px !important;
        border: 1px solid #E4E9FF !important;
        background: #F8FAFF !important;
        color: #1D1929 !important;
        font-weight: 700 !important;
    }}
    .stButton > button:hover, .stLinkButton > a:hover {{
        border-color: {_accent} !important;
        color: {_accent} !important;
    }}

    .stSelectbox [data-baseweb="select"] > div {{
        background: #F8FAFF !important; border-color: #E4E9FF !important;
    }}
    .stToggle [data-baseweb="checkbox"] div[aria-checked="true"] {{
        background: {_accent} !important;
    }}

    iframe {{ border-radius: 12px !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------
# ICONS - Font Awesome (loaded via CDN above), not hand-drawn SVGs. A real
# icon set reads as professional; the earlier custom paths didn't.
# -----------------------------------------------------------------------
def fa_icon(name: str, color: str, size: int = 18) -> str:
    return f'<i class="fa-solid {name}" style="color:{color}; font-size:{size}px;"></i>'

def icon_span(html: str) -> str:
    return f'<span style="display:inline-flex; align-items:center;">{html}</span>'

def logo_badge(size: int = 52) -> str:
    """AI-chip mark - a rounded square chip labeled 'AI' with short circuit
    pins radiating from each side, in the app's actual accent colors
    (originally requested to match a chip/circuit-style reference image,
    recolored here instead of copying that image directly)."""
    pins = ""
    for x in (34, 50, 66):
        pins += f'<line x1="{x}" y1="6" x2="{x}" y2="22" stroke="{_accent}" stroke-width="3" stroke-linecap="round"/>'
        pins += f'<line x1="{x}" y1="78" x2="{x}" y2="94" stroke="{_accent}" stroke-width="3" stroke-linecap="round"/>'
    for y in (34, 50, 66):
        pins += f'<line x1="6" y1="{y}" x2="22" y2="{y}" stroke="{_accent}" stroke-width="3" stroke-linecap="round"/>'
        pins += f'<line x1="78" y1="{y}" x2="94" y2="{y}" stroke="{_accent}" stroke-width="3" stroke-linecap="round"/>'
    return f"""<div style="width:{size}px; height:{size}px; flex-shrink:0;">
        <svg width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            {pins}
            <rect x="22" y="22" width="56" height="56" rx="12"
                  fill="{_accent2}" stroke="{_accent}" stroke-width="3"/>
            <text x="50" y="59" font-size="26" font-weight="700" text-anchor="middle"
                  fill="{_accent}" font-family="Poppins, sans-serif">AI</text>
        </svg>
    </div>"""

# -----------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------
st.markdown(
    f"""
    <div class="adx-hero">{logo_badge(52)}<h1>AI DRIVER <span class="accent">APP</span></h1></div>
    <div class="adx-subtitle"><span class="adx-status-dot"></span>Online &middot; {st.session_state.brand} companion mode</div>
    """,
    unsafe_allow_html=True,
)

# Runs on EVERY script execution (not just the one that generated the
# reply) - see render_speech()'s docstring for why that's what keeps the
# audio alive across the rerun that happens right after a reply.
render_speech(st.session_state.pending_speech)

col_map, col_chat = st.columns([1, 1])

with col_map:
    with st.container(border=True, key="nav_card"):
        st.markdown(f'<h3>{icon_span(fa_icon("fa-location-dot", _accent, 16))} Navigation</h3>', unsafe_allow_html=True)
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
    with st.container(border=True, key="chat_card"):
        st.markdown(f'<h3>{icon_span(fa_icon("fa-comment-dots", _accent, 16))} Chat</h3>', unsafe_allow_html=True)
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

    # Hands-free re-listening is driven entirely in Python now (session_state
    # + st.rerun(), a soft in-app rerun - not the broken page-navigation
    # trick). The very first mic use on a fresh session still needs one real
    # button click, since a browser will only grant microphone permission in
    # response to an actual user gesture; every listen after that (manual or
    # hands-free) can be triggered programmatically once that's unlocked.
    #
    # capture_voice() returns None while its JS promise is still pending -
    # the block calling it must stay reachable, with the SAME delay_ms, on
    # every rerun until a real string comes back (that's how st_javascript's
    # bridge resolves). A click just arms "listening" and does one
    # st.rerun(); it must NOT call capture_voice() directly inside the
    # click handler, or the resolved value has nowhere consistent to land.
    voice_text = ""
    if st.session_state.listening:
        if not api_key:
            st.session_state.listening = False
            st.session_state.pending_capture = False
            st.error("Enter your free Gemini API key in the sidebar first - it's also what turns your voice into text.")
        else:
            st.caption("🎙️ Listening...")
            delay_ms = min(max(st.session_state.last_reply_words * 350, 900), 6000) \
                if st.session_state.pending_capture else 0
            result = capture_voice(delay_ms=delay_ms)
            if result is not None:
                st.session_state.listening = False
                st.session_state.pending_capture = False
                if result and "|" in result:
                    mime_type, audio_b64 = result.split("|", 1)
                    voice_text = transcribe_audio(get_client(api_key), mime_type, audio_b64)
                if not voice_text:
                    st.rerun()  # heard nothing (or no mic access) - stop listening cleanly, no ghost turn
    else:
        if st.button("🎤 Tap to talk", use_container_width=True):
            st.session_state.mic_unlocked = True
            st.session_state.listening = True
            st.session_state.pending_capture = False
            st.rerun()

    if st.session_state.last_music_query:
        # A real <a> link the browser renders directly in the page (not
        # inside a components.html() iframe) - Streamlit Cloud's iframe
        # sandbox has no "allow-popups" flag, which silently blocked the
        # old window.open()-based auto-handoff with zero error. A real
        # link_button sidesteps that entirely because it isn't sandboxed.
        st.link_button(
            "🎵 Open in Spotify",
            spotify_url(st.session_state.last_music_query),
            use_container_width=True,
        )

    typed_text = st.chat_input("Or type here...")

    incoming = voice_text or typed_text

    if incoming:
        cleaned_text = incoming.strip()
        st.session_state.history.append({"role": "user", "content": cleaned_text})

        intent, payload = detect_intent(cleaned_text)

        if not api_key:
            st.error("Enter your free Gemini API key in the sidebar first.")
        else:
            if intent == "nav":
                st.session_state.last_dest = payload
                reply = f"Found it - tap the Open in Maps button below and I'll hand you straight to directions for {payload}."
            elif intent == "music":
                st.session_state.last_music_query = payload
                reply = f"Sure thing - tap the Open in Spotify button below to start {payload}."
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

            st.session_state.pending_speech = prepare_speech(
                reply, elevenlabs_key=elevenlabs_key, elevenlabs_voice_id=elevenlabs_voice_id
            )
            st.session_state.last_reply_words = len(reply.split())

            if st.session_state.hands_free and st.session_state.mic_unlocked:
                st.session_state.listening = True
                st.session_state.pending_capture = True

            st.rerun()
