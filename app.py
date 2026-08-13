import re
import streamlit as st
from groq import Groq


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Kurdish Lyrics → English SRT",
    page_icon="🎵",
    layout="centered",
)


# ============================================================
# UI
# ============================================================

st.title("🎵 Kurdish Lyrics → English SRT")

st.caption(
    "Upload a Kurdish song → transcribe with timestamps → "
    "translate to English → download SRT"
)


# ============================================================
# HELPERS
# ============================================================

def srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp."""

    seconds = max(0.0, float(seconds))

    total = int(seconds)
    ms = int(round((seconds - total) * 1000))

    if ms >= 1000:
        total += 1
        ms = 0

    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def clean_text(text: str) -> str:
    """Clean unnecessary whitespace."""

    text = re.sub(r"\s+", " ", text or "").strip()

    return text


def make_srt(items):
    """Create an SRT file from timed subtitle items."""

    blocks = []

    subtitle_number = 1

    for item in items:

        start = item["start"]
        end = item["end"]
        text = clean_text(item["text"])

        if not text:
            continue

        blocks.append(
            f"{subtitle_number}\n"
            f"{srt_time(start)} --> {srt_time(end)}\n"
            f"{text}\n"
        )

        subtitle_number += 1

    return "\n".join(blocks)


# ============================================================
# GROQ TRANSCRIPTION
# ============================================================

def groq_transcribe(client, audio_bytes, filename):
    """
    Transcribe Kurdish audio using Groq Whisper
    and return timestamped segments.
    """

    result = client.audio.transcriptions.create(
        file=(filename, audio_bytes),

        model="whisper-large-v3",

        response_format="verbose_json",

        timestamp_granularities=["segment"],

        language="ku",

        temperature=0,
    )

    segments = []

    # --------------------------------------------------------
    # SDK object response
    # --------------------------------------------------------

    for seg in getattr(result, "segments", []) or []:

        text = clean_text(
            getattr(seg, "text", "")
        )

        if text:

            segments.append(
                {
                    "start": float(
                        getattr(seg, "start", 0)
                    ),

                    "end": float(
                        getattr(seg, "end", 0)
                    ),

                    "ku": text,
                }
            )

    # --------------------------------------------------------
    # Dictionary response fallback
    # --------------------------------------------------------

    if not segments:

        if hasattr(result, "model_dump"):

            raw = result.model_dump()

        elif isinstance(result, dict):

            raw = result

        else:

            raw = {}

        for seg in raw.get("segments", []):

            text = clean_text(
                seg.get("text", "")
            )

            if text:

                segments.append(
                    {
                        "start": float(
                            seg.get("start", 0)
                        ),

                        "end": float(
                            seg.get("end", 0)
                        ),

                        "ku": text,
                    }
                )

    return segments


# ============================================================
# TRANSLATION
# ============================================================

def translate_batch(client, texts):
    """
    Translate Kurdish Sorani lyrics into English.

    One input line = exactly one output line.
    """

    prompt = """
Translate each Kurdish Sorani lyric line into natural English.

IMPORTANT RULES:

- Return EXACTLY one English line for each numbered input.
- Keep the numbering.
- Do NOT explain anything.
- Do NOT add explanations.
- Do NOT merge lines.
- Do NOT split lines.
- Preserve names.
- Preserve the poetic meaning as naturally as possible.
- Do not add quotation marks.
- Output ONLY the numbered translations.

INPUT:
"""

    for i, text in enumerate(texts, 1):

        prompt += f"{i}. {text}\n"

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        temperature=0.15,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise Kurdish Sorani "
                    "to English lyric translator."
                ),
            },

            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    content = (
        response.choices[0]
        .message
        .content
        or ""
    )

    result = {}

    for line in content.splitlines():

        match = re.match(
            r"^\s*(\d+)\s*[\.\):-]\s*(.+?)\s*$",
            line,
        )

        if match:

            number = int(
                match.group(1)
            )

            translation = clean_text(
                match.group(2)
            )

            result[number] = translation

    return [
        result.get(i, "")
        for i in range(1, len(texts) + 1)
    ]


# ============================================================
# GET API KEY FROM STREAMLIT SECRETS
# ============================================================

try:

    api_key = st.secrets["GROQ_API_KEY"]

except Exception:

    st.error(
        "❌ GROQ_API_KEY is not configured."
    )

    st.info(
        "Go to Streamlit → Manage app → Settings → Secrets "
        "and add GROQ_API_KEY."
    )

    st.stop()


# ============================================================
# UPLOAD AUDIO
# ============================================================

st.markdown("### 📁 Upload the song")

uploaded = st.file_uploader(

    "MP3 / M4A / WAV / FLAC / OGG",

    type=[
        "mp3",
        "m4a",
        "wav",
        "flac",
        "ogg",
    ],
)


# ============================================================
# PROCESS
# ============================================================

if uploaded:

    st.audio(uploaded)

    if st.button(
        "🚀 Generate English.srt",
        type="primary",
        use_container_width=True,
    ):

        try:

            # ------------------------------------------------
            # Create Groq client
            # ------------------------------------------------

            client = Groq(
                api_key=api_key
            )

            # ------------------------------------------------
            # Read uploaded audio
            # ------------------------------------------------

            audio = uploaded.getvalue()

            # ------------------------------------------------
            # File size check
            # ------------------------------------------------

            if len(audio) > 24 * 1024 * 1024:

                st.error(
                    "❌ This file is larger than 24 MB. "
                    "Please compress the audio and try again."
                )

                st.stop()

            # ------------------------------------------------
            # Processing status
            # ------------------------------------------------

            with st.status(
                "Processing song...",
                expanded=True,
            ) as status:

                # ============================================
                # STEP 1
                # ============================================

                st.write(
                    "🎙️ Step 1/3 — "
                    "Kurdish transcription + timestamps"
                )

                segments = groq_transcribe(
                    client,
                    audio,
                    uploaded.name,
                )

                if not segments:

                    raise RuntimeError(
                        "No timestamped segments were returned."
                    )

                st.write(
                    f"✅ Found {len(segments)} "
                    "timed lyric segments."
                )

                # ============================================
                # STEP 2
                # ============================================

                st.write(
                    "🌍 Step 2/3 — "
                    "Translating Kurdish → English"
                )

                english = []

                batch_size = 25

                for start in range(
                    0,
                    len(segments),
                    batch_size,
                ):

                    batch = segments[
                        start:start + batch_size
                    ]

                    batch_texts = [
                        item["ku"]
                        for item in batch
                    ]

                    translated = translate_batch(
                        client,
                        batch_texts,
                    )

                    english.extend(
                        translated
                    )

                # ============================================
                # STEP 3
                # ============================================

                st.write(
                    "📝 Step 3/3 — "
                    "Building English.srt"
                )

                # ------------------------------------------------
                # English subtitles
                # ------------------------------------------------

                english_items = []

                for seg, en in zip(
                    segments,
                    english,
                ):

                    english_items.append(
                        {
                            "start": seg["start"],
                            "end": seg["end"],

                            "text": (
                                en
                                if en
                                else seg["ku"]
                            ),
                        }
                    )

                english_srt = make_srt(
                    english_items
                )

                # ------------------------------------------------
                # Kurdish subtitles
                # ------------------------------------------------

                kurdish_items = []

                for seg in segments:

                    kurdish_items.append(
                        {
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["ku"],
                        }
                    )

                kurdish_srt = make_srt(
                    kurdish_items
                )

                # ------------------------------------------------
                # Finish
                # ------------------------------------------------

                status.update(
                    label="✅ Finished!",
                    state="complete",
                )

            # ====================================================
            # RESULTS
            # ====================================================

            st.success(
                "✅ English and Kurdish SRT files are ready!"
            )

            # ----------------------------------------------------
            # Tabs
            # ----------------------------------------------------

            tab1, tab2 = st.tabs(
                [
                    "🇬🇧 English SRT",
                    "🟡 Kurdish SRT",
                ]
            )

            # ====================================================
            # ENGLISH
            # ====================================================

            with tab1:

                st.code(
                    english_srt,
                    language="text",
                )

                st.download_button(

                    "⬇️ Download English.srt",

                    data=english_srt.encode(
                        "utf-8"
                    ),

                    file_name="English.srt",

                    mime="application/x-subrip",

                    use_container_width=True,
                )

            # ====================================================
            # KURDISH
            # ====================================================

            with tab2:

                st.code(
                    kurdish_srt,
                    language="text",
                )

                st.download_button(

                    "⬇️ Download Kurdish.srt",

                    data=kurdish_srt.encode(
                        "utf-8"
                    ),

                    file_name="Kurdish.srt",

                    mime="application/x-subrip",

                    use_container_width=True,
                )

        except Exception as e:

            st.error(
                f"❌ Processing failed: {e}"
            )

            st.caption(
                "If the error mentions the API key, "
                "model, file size, or unsupported audio "
                "format, send me the exact error."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "☁️ Cloud-only app — nothing needs to be installed "
    "on your phone."
)
