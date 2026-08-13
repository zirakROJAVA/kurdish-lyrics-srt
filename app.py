import re
import streamlit as st
from groq import Groq


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Kurdish Lyrics → English SRT",
    page_icon="🎵",
    layout="centered"
)

TRANSCRIPTION_MODEL = "whisper-large-v3"
TRANSLATION_MODEL = "llama-3.3-70b-versatile"

MAX_FILE_SIZE = 25 * 1024 * 1024


# ============================================================
# UI
# ============================================================

st.title("🎵 Kurdish Lyrics → English SRT")

st.caption(
    "Kurdish Sorani song → transcription → English translation → SRT"
)


# ============================================================
# API KEY
# ============================================================

try:
    API_KEY = st.secrets["GROQ_API_KEY"]

except Exception:
    st.error("❌ GROQ_API_KEY is not configured.")

    st.info(
        "Streamlit → Manage app → Settings → Secrets → "
        "GROQ_API_KEY"
    )

    st.stop()


client = Groq(api_key=API_KEY)


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = str(text)

    text = text.replace("\n", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def srt_time(seconds):

    seconds = max(0.0, float(seconds))

    milliseconds = int(round(seconds * 1000))

    hours = milliseconds // 3600000
    milliseconds %= 3600000

    minutes = milliseconds // 60000
    milliseconds %= 60000

    seconds_int = milliseconds // 1000
    milliseconds %= 1000

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds_int:02d},"
        f"{milliseconds:03d}"
    )


def make_srt(items):

    output = []

    number = 1

    for item in items:

        start = float(item["start"])
        end = float(item["end"])

        text = clean_text(item["text"])

        if not text:
            continue

        if end <= start:
            end = start + 1

        output.append(
            f"{number}\n"
            f"{srt_time(start)} --> {srt_time(end)}\n"
            f"{text}\n"
        )

        number += 1

    return "\n".join(output)


# ============================================================
# HALLUCINATION DETECTION
# ============================================================

BAD_PHRASES = [
    "thank you for watching",
    "thanks for watching",
    "subscribe to the channel",
    "subscribe to my channel",
    "like and subscribe",
    "welcome to my channel",
    "we will start",
    "let's start",
    "in this video",
    "the following video",
    "english lyrics",
    "lyrics are not",
    "preserve names",
    "use kurdish sorani",
    "new generation",
    "thank you",
]


def looks_like_hallucination(text):

    low = text.lower()

    for phrase in BAD_PHRASES:

        if phrase in low:
            return True

    return False


def remove_obvious_hallucinations(segments):

    cleaned = []

    for segment in segments:

        text = clean_text(segment["text"])

        if not text:
            continue

        if looks_like_hallucination(text):
            continue

        cleaned.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": text
        })

    return cleaned


# ============================================================
# TRANSCRIPTION
# ============================================================

def transcribe_song(audio_bytes, filename):

    prompt = (
        "Kurdish Sorani song lyrics. "
        "Central Kurdish language. "
        "The audio is a Kurdish song, not English. "
        "Transcribe only the words actually sung. "
        "Do not translate. "
        "Do not add explanations. "
        "Do not add introductions. "
        "Do not write YouTube phrases. "
        "Do not write English instructions. "
        "Use Kurdish Sorani Arabic-based script when possible. "
        "Preserve Kurdish names and words."
    )

    result = client.audio.transcriptions.create(

        file=(filename, audio_bytes),

        model=TRANSCRIPTION_MODEL,

        prompt=prompt,

        response_format="verbose_json",

        timestamp_granularities=[
            "segment"
        ],

        temperature=0
    )

    segments = []

    raw_segments = getattr(
        result,
        "segments",
        None
    )

    if raw_segments:

        for seg in raw_segments:

            text = clean_text(
                getattr(seg, "text", "")
            )

            if not text:
                continue

            start = float(
                getattr(seg, "start", 0) or 0
            )

            end = float(
                getattr(seg, "end", 0) or 0
            )

            if end <= start:
                end = start + 1

            segments.append({
                "start": start,
                "end": end,
                "text": text
            })

    # --------------------------------------------------------
    # Dictionary fallback
    # --------------------------------------------------------

    if not segments:

        try:

            raw = result.model_dump()

        except Exception:

            raw = result

        if isinstance(raw, dict):

            for seg in raw.get(
                "segments",
                []
            ):

                text = clean_text(
                    seg.get("text", "")
                )

                if not text:
                    continue

                start = float(
                    seg.get("start", 0)
                )

                end = float(
                    seg.get("end", 0)
                )

                if end <= start:
                    end = start + 1

                segments.append({
                    "start": start,
                    "end": end,
                    "text": text
                })

    return remove_obvious_hallucinations(
        segments
    )


# ============================================================
# TRANSLATION
# ============================================================

def translate_lines(texts):

    if not texts:
        return []

    prompt = """
You are an expert Kurdish Sorani song translator.

Translate the following Kurdish Sorani lyrics into natural English.

STRICT RULES:

- Return exactly one translation for every numbered line.
- Keep the numbering.
- Never merge lines.
- Never split lines.
- Do not invent lyrics.
- Do not add explanations.
- Do not add comments.
- Do not add introductions.
- Preserve names.
- Preserve poetic meaning.
- If a word is unclear, preserve the original Kurdish word.
- Do NOT create text that is not present in the Kurdish input.

INPUT:

"""

    for i, text in enumerate(texts, 1):

        prompt += f"{i}. {text}\n"

    response = client.chat.completions.create(

        model=TRANSLATION_MODEL,

        temperature=0,

        messages=[
            {
                "role": "system",
                "content": (
                    "You translate Kurdish Sorani lyrics "
                    "to English accurately."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = (
        response
        .choices[0]
        .message
        .content
        or ""
    )

    translations = {}

    for line in content.splitlines():

        match = re.match(
            r"^\s*(\d+)\s*[\.\):-]\s*(.+?)\s*$",
            line
        )

        if not match:
            continue

        number = int(
            match.group(1)
        )

        text = clean_text(
            match.group(2)
        )

        if text:

            translations[number] = text

    return [
        translations.get(
            i,
            ""
        )
        for i in range(
            1,
            len(texts) + 1
        )
    ]


# ============================================================
# UPLOAD
# ============================================================

st.markdown("## 📁 Upload the song")

uploaded = st.file_uploader(
    "MP3 / M4A / WAV / FLAC / OGG",
    type=[
        "mp3",
        "m4a",
        "wav",
        "flac",
        "ogg"
    ]
)


# ============================================================
# PROCESS
# ============================================================

if uploaded:

    st.audio(uploaded)

    audio = uploaded.getvalue()

    size_mb = len(audio) / (
        1024 * 1024
    )

    st.caption(
        f"📦 File size: {size_mb:.2f} MB"
    )

    if len(audio) > MAX_FILE_SIZE:

        st.error(
            "❌ File is larger than 25 MB. "
            "Please compress it first."
        )

        st.stop()

    if st.button(
        "🚀 Generate SRT",
        type="primary",
        use_container_width=True
    ):

        try:

            # ==================================================
            # STEP 1
            # ==================================================

            with st.status(
                "🎙️ Processing song...",
                expanded=True
            ) as status:

                st.write(
                    "1️⃣ Transcribing Kurdish Sorani..."
                )

                segments = transcribe_song(
                    audio,
                    uploaded.name
                )

                if not segments:

                    raise RuntimeError(
                        "No reliable transcription "
                        "segments were returned."
                    )

                st.write(
                    f"✅ {len(segments)} "
                    f"segments detected."
                )

                # ==============================================
                # STEP 2
                # ==============================================

                st.write(
                    "2️⃣ Translating Kurdish → English..."
                )

                english = []

                batch_size = 20

                for i in range(
                    0,
                    len(segments),
                    batch_size
                ):

                    batch = segments[
                        i:i + batch_size
                    ]

                    texts = [
                        x["text"]
                        for x in batch
                    ]

                    result = translate_lines(
                        texts
                    )

                    english.extend(
                        result
                    )

                # ==============================================
                # STEP 3
                # ==============================================

                st.write(
                    "3️⃣ Creating SRT files..."
                )

                english_items = []
                kurdish_items = []

                for i, segment in enumerate(
                    segments
                ):

                    kurdish_text = segment[
                        "text"
                    ]

                    english_text = ""

                    if i < len(english):

                        english_text = clean_text(
                            english[i]
                        )

                    # Kurdish
                    kurdish_items.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": kurdish_text
                    })

                    # English
                    english_items.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": (
                            english_text
                            if english_text
                            else kurdish_text
                        )
                    })

                kurdish_srt = make_srt(
                    kurdish_items
                )

                english_srt = make_srt(
                    english_items
                )

                status.update(
                    label="✅ Finished successfully",
                    state="complete"
                )

            # ==================================================
            # RESULTS
            # ==================================================

            st.success(
                "🎉 SRT files are ready!"
            )

            tab1, tab2 = st.tabs(
                [
                    "🇬🇧 English",
                    "🟡 Kurdish"
                ]
            )

            # --------------------------------------------------
            # ENGLISH
            # --------------------------------------------------

            with tab1:

                st.code(
                    english_srt,
                    language="text"
                )

                st.download_button(
                    "⬇️ Download English.srt",
                    data=english_srt.encode(
                        "utf-8"
                    ),
                    file_name="English.srt",
                    mime="application/x-subrip",
                    use_container_width=True
                )

            # --------------------------------------------------
            # KURDISH
            # --------------------------------------------------

            with tab2:

                st.code(
                    kurdish_srt,
                    language="text"
                )

                st.download_button(
                    "⬇️ Download Kurdish.srt",
                    data=kurdish_srt.encode(
                        "utf-8"
                    ),
                    file_name="Kurdish.srt",
                    mime="application/x-subrip",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                "❌ Processing failed"
            )

            st.code(
                str(e)
            )

            st.info(
                "Send me this exact error if it happens."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "☁️ Cloud-only • Kurdish Sorani • English SRT"
)
