import re
import html
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
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main {
        max-width: 900px;
        margin: auto;
    }

    .title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #999;
        font-size: 16px;
        margin-bottom: 30px;
    }

    .success-box {
        padding: 15px;
        border-radius: 10px;
        background: rgba(0, 200, 100, 0.08);
        border: 1px solid rgba(0, 200, 100, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🎵 Kurdish Lyrics → English SRT</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Upload a Kurdish song → transcribe with timestamps → '
    'translate to English → download SRT'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# API KEY
# ============================================================

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = None


if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is not configured.")

    st.info(
        """
        Go to your Streamlit app:

        **Manage app → Settings → Secrets**

        Then add:

        `GROQ_API_KEY = "your_groq_api_key"`
        """
    )

    st.stop()


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# HELPERS
# ============================================================

def clean_text(text: str) -> str:
    """Clean whitespace without destroying Kurdish characters."""
    if not text:
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp."""
    seconds = max(0.0, float(seconds))

    total_ms = int(round(seconds * 1000))

    hours = total_ms // 3_600_000
    total_ms %= 3_600_000

    minutes = total_ms // 60_000
    total_ms %= 60_000

    secs = total_ms // 1_000
    milliseconds = total_ms % 1_000

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{milliseconds:03d}"
    )


def make_srt(segments, text_key="text"):
    """Build a valid UTF-8 SRT file."""

    blocks = []
    number = 1

    for segment in segments:
        text = clean_text(segment.get(text_key, ""))

        if not text:
            continue

        start = float(segment.get("start", 0))
        end = float(segment.get("end", start + 1))

        # Make sure end is after start.
        if end <= start:
            end = start + 0.5

        blocks.append(
            f"{number}\n"
            f"{srt_time(start)} --> {srt_time(end)}\n"
            f"{text}\n"
        )

        number += 1

    return "\n".join(blocks)


# ============================================================
# TRANSCRIPTION
# ============================================================

def transcribe_kurdish(audio_bytes, filename):
    """
    Transcribe Kurdish audio.

    IMPORTANT:
    We intentionally DO NOT send language="ku".
    Groq's current supported-language list does not contain "ku".
    Whisper is allowed to detect the language automatically.
    """

    result = client.audio.transcriptions.create(
        file=(filename, audio_bytes),

        # High-accuracy multilingual Whisper model.
        model="whisper-large-v3",

        # Required for timestamps.
        response_format="verbose_json",

        # Segment timestamps.
        timestamp_granularities=["segment"],

        # Keep transcription deterministic.
        temperature=0.0,

        # Helpful context for a SONG.
        # This does not tell Whisper what language code to use.
        prompt=(
            "This is a Kurdish song with sung Kurdish lyrics. "
            "Transcribe only the words that are actually heard in the audio. "
            "Do not invent instructions, explanations, subtitles, or commentary. "
            "Preserve names and Kurdish words as accurately as possible."
        ),
    )

    segments = []

    raw_segments = getattr(result, "segments", None)

    # SDK object response
    if raw_segments:
        for seg in raw_segments:
            text = clean_text(getattr(seg, "text", ""))

            if not text:
                continue

            start = float(getattr(seg, "start", 0))
            end = float(getattr(seg, "end", start + 1))

            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )

    # Dictionary fallback
    if not segments:
        if hasattr(result, "model_dump"):
            raw = result.model_dump()
        elif isinstance(result, dict):
            raw = result
        else:
            raw = {}

        for seg in raw.get("segments", []) or []:
            text = clean_text(seg.get("text", ""))

            if not text:
                continue

            start = float(seg.get("start", 0))
            end = float(seg.get("end", start + 1))

            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )

    return segments


# ============================================================
# TRANSLATION
# ============================================================

def translate_batch(kurdish_lines):
    """
    Translate Kurdish Sorani lyrics to English.

    Returns exactly one translation for each input line.
    """

    if not kurdish_lines:
        return []

    numbered_lines = []

    for i, line in enumerate(kurdish_lines, 1):
        numbered_lines.append(f"{i}. {line}")

    input_text = "\n".join(numbered_lines)

    system_prompt = """
You are a professional Kurdish Sorani → English lyric translator.

Your task is to translate Kurdish Sorani song lyrics into natural English.

IMPORTANT RULES:

1. Translate only the supplied lyrics.
2. Do not add explanations.
3. Do not add introductions.
4. Do not add conclusions.
5. Do not write "Thank you for watching".
6. Do not write instructions.
7. Do not invent lyrics.
8. Keep exactly one English line for every numbered Kurdish line.
9. Keep the numbering exactly.
10. Preserve names and places.
11. Preserve poetic meaning.
12. Use natural English rather than word-for-word English when appropriate.
13. Never merge two numbered lines.
14. Never split one numbered line into multiple numbered lines.

OUTPUT FORMAT:

1. English translation
2. English translation
3. English translation

Nothing else.
"""

    user_prompt = (
        "Translate these Kurdish Sorani lyric lines:\n\n"
        + input_text
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    content = ""

    try:
        content = response.choices[0].message.content or ""
    except Exception:
        content = ""

    # Remove markdown code fences if model accidentally adds them.
    content = re.sub(
        r"```(?:text|plaintext)?",
        "",
        content,
        flags=re.IGNORECASE,
    )

    content = content.replace("```", "")

    translations = {}

    for line in content.splitlines():
        line = line.strip()

        if not line:
            continue

        # Accept:
        # 1. text
        # 1) text
        # 1: text
        # 1 - text
        match = re.match(
            r"^(\d+)\s*[\.\)\:\-]\s*(.+)$",
            line,
        )

        if match:
            index = int(match.group(1))
            text = clean_text(match.group(2))

            if text:
                translations[index] = text

    result = []

    for i in range(1, len(kurdish_lines) + 1):
        result.append(translations.get(i, ""))

    return result


# ============================================================
# OPTIONAL HALLUCINATION FILTER
# ============================================================

HALLUCINATION_PHRASES = [
    "thank you for watching",
    "thanks for watching",
    "subscribe",
    "like and subscribe",
    "transcribe the kurdish",
    "transcribe the kurdish sorani lyrics",
    "use kurdish sorani script when possible",
    "kurdish sorani translation and translation by",
    "translation by",
]


def looks_like_hallucination(text):
    """
    Remove common Whisper/prompt hallucinations.
    """

    normalized = clean_text(text).lower()

    if not normalized:
        return True

    for phrase in HALLUCINATION_PHRASES:
        if phrase in normalized:
            return True

    return False


# ============================================================
# FILE UPLOAD
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


if uploaded:

    st.audio(uploaded)

    file_size_mb = uploaded.size / (1024 * 1024)

    st.caption(
        f"📄 {uploaded.name} • {file_size_mb:.2f} MB"
    )

    # Groq's documented free-tier direct upload limit is 25 MB.
    if uploaded.size > 25 * 1024 * 1024:
        st.error(
            "❌ This audio file is larger than 25 MB. "
            "Please compress it to under 25 MB."
        )
        st.stop()

    st.markdown("---")

    generate = st.button(
        "🚀 Generate English SRT",
        type="primary",
        use_container_width=True,
    )

    if generate:

        audio_bytes = uploaded.getvalue()

        try:

            # ====================================================
            # STEP 1
            # ====================================================

            with st.status(
                "Processing song...",
                expanded=True,
            ) as status:

                st.write(
                    "🎙️ Step 1/3 — "
                    "Detecting language + Kurdish transcription + timestamps"
                )

                segments = transcribe_kurdish(
                    audio_bytes,
                    uploaded.name,
                )

                if not segments:
                    raise RuntimeError(
                        "Whisper returned no timestamped segments."
                    )

                st.write(
                    f"✅ Found {len(segments)} audio segments."
                )

                # ====================================================
                # FILTER BAD / HALLUCINATED SEGMENTS
                # ====================================================

                cleaned_segments = []

                for segment in segments:

                    text = segment["text"]

                    if looks_like_hallucination(text):
                        continue

                    cleaned_segments.append(segment)

                if not cleaned_segments:
                    raise RuntimeError(
                        "No usable lyric segments were found."
                    )

                segments = cleaned_segments

                st.write(
                    f"🧹 Kept {len(segments)} usable lyric segments."
                )

                # ====================================================
                # STEP 2
                # ====================================================

                st.write(
                    "🌍 Step 2/3 — "
                    "Translating Kurdish → English"
                )

                english_segments = []

                # Small batches prevent very large prompts.
                batch_size = 20

                for batch_start in range(
                    0,
                    len(segments),
                    batch_size,
                ):

                    batch = segments[
                        batch_start:
                        batch_start + batch_size
                    ]

                    kurdish_lines = [
                        x["text"]
                        for x in batch
                    ]

                    translations = translate_batch(
                        kurdish_lines
                    )

                    for segment, english in zip(
                        batch,
                        translations,
                    ):

                        english = clean_text(english)

                        # If translation failed, use Kurdish
                        # rather than creating an empty subtitle.
                        if not english:
                            english = segment["text"]

                        english_segments.append(
                            {
                                "start": segment["start"],
                                "end": segment["end"],
                                "text": english,
                            }
                        )

                # ====================================================
                # STEP 3
                # ====================================================

                st.write(
                    "📝 Step 3/3 — "
                    "Building SRT files"
                )

                kurdish_srt = make_srt(
                    segments,
                    text_key="text",
                )

                english_srt = make_srt(
                    english_segments,
                    text_key="text",
                )

                status.update(
                    label="✅ Finished!",
                    state="complete",
                )

            # ========================================================
            # RESULTS
            # ========================================================

            st.success(
                "🎉 Your subtitle files are ready!"
            )

            tab1, tab2 = st.tabs(
                [
                    "🇬🇧 English SRT",
                    "☀️ Kurdish SRT",
                ]
            )

            # ========================================================
            # ENGLISH
            # ========================================================

            with tab1:

                st.download_button(
                    label="⬇️ Download English.srt",
                    data=english_srt.encode("utf-8-sig"),
                    file_name="English.srt",
                    mime="application/x-subrip",
                    use_container_width=True,
                )

                st.text_area(
                    "English SRT",
                    english_srt,
                    height=400,
                )

            # ========================================================
            # KURDISH
            # ========================================================

            with tab2:

                st.download_button(
                    label="⬇️ Download Kurdish.srt",
                    data=kurdish_srt.encode("utf-8-sig"),
                    file_name="Kurdish.srt",
                    mime="application/x-subrip",
                    use_container_width=True,
                )

                st.text_area(
                    "Kurdish SRT",
                    kurdish_srt,
                    height=400,
                )

        except Exception as e:

            st.error(
                f"❌ Processing failed:\n\n{e}"
            )

            st.info(
                """
If this error appears again, send me the **exact error message**.

The most important things to check are:
- GROQ_API_KEY
- audio format
- audio size
- Groq model availability
- API rate limit
"""
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "☁️ Cloud-only app — nothing needs to be installed on your phone."
)
