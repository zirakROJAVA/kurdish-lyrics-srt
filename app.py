import re
import streamlit as st
from groq import Groq


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Kurdish Lyrics → English SRT",
    page_icon="🎵",
    layout="centered"
)


# =========================================================
# TITLE
# =========================================================

st.title("🎵 Kurdish Lyrics → English SRT")

st.caption(
    "Upload a Kurdish song → transcribe with timestamps → "
    "translate to English → download SRT"
)


# =========================================================
# HELPERS
# =========================================================

def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def srt_time(seconds: float) -> str:
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


def make_srt(items):
    blocks = []
    number = 1

    for item in items:

        text = clean_text(item.get("text", ""))

        if not text:
            continue

        start = float(item.get("start", 0))
        end = float(item.get("end", 0))

        if end <= start:
            end = start + 1

        blocks.append(
            f"{number}\n"
            f"{srt_time(start)} --> {srt_time(end)}\n"
            f"{text}\n"
        )

        number += 1

    return "\n".join(blocks)


# =========================================================
# GROQ TRANSCRIPTION
# =========================================================

def groq_transcribe(client, audio_bytes, filename):

    result = client.audio.transcriptions.create(

        file=(filename, audio_bytes),

        model="whisper-large-v3",

        # IMPORTANT:
        # Do NOT use language="ku".
        # Groq currently does not accept "ku" as a language parameter.

        prompt=(
            "The audio contains Kurdish Sorani lyrics. "
            "Transcribe the Kurdish Sorani speech/song as accurately "
            "as possible. Preserve Kurdish words, names and pronunciation. "
            "Do not translate the lyrics. "
            "Use Kurdish Sorani script when possible."
        ),

        response_format="verbose_json",

        timestamp_granularities=["segment"],

        temperature=0
    )

    segments = []

    raw_segments = getattr(result, "segments", None)

    if raw_segments:

        for seg in raw_segments:

            text = clean_text(
                getattr(seg, "text", "")
            )

            if not text:
                continue

            segments.append({
                "start": float(
                    getattr(seg, "start", 0)
                ),
                "end": float(
                    getattr(seg, "end", 0)
                ),
                "ku": text
            })

    # Fallback for dictionary-style responses

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

            if not text:
                continue

            segments.append({
                "start": float(
                    seg.get("start", 0)
                ),
                "end": float(
                    seg.get("end", 0)
                ),
                "ku": text
            })

    return segments


# =========================================================
# TRANSLATION
# =========================================================

def translate_batch(client, texts):

    if not texts:
        return []

    prompt = """
Translate the following Kurdish Sorani song lyrics into natural English.

IMPORTANT RULES:

1. Return EXACTLY one English line for each numbered Kurdish line.
2. Keep the same numbering.
3. Do NOT merge lines.
4. Do NOT split lines.
5. Do NOT explain anything.
6. Do NOT add notes.
7. Preserve names and places.
8. Preserve the poetic meaning as naturally as possible.
9. Output ONLY the numbered English translations.

Example:

INPUT:
1. سڵاو ئەی دڵدارەکەم
2. تۆ هەموو ژیانی منیت

OUTPUT:
1. Hello, my beloved
2. You are my entire life

Now translate:

"""

    for i, text in enumerate(texts, 1):
        prompt += f"{i}. {text}\n"

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        temperature=0.1,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Kurdish Sorani "
                    "to English literary translator."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = (
        response.choices[0]
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

        if match:

            number = int(match.group(1))

            translation = clean_text(
                match.group(2)
            )

            translations[number] = translation

    return [
        translations.get(i, "")
        for i in range(1, len(texts) + 1)
    ]


# =========================================================
# GET API KEY FROM STREAMLIT SECRETS
# =========================================================

try:

    api_key = st.secrets["GROQ_API_KEY"]

except Exception:

    st.error("❌ GROQ_API_KEY is not configured.")

    st.info(
        "Go to Streamlit → Manage app → Settings → Secrets "
        "and add GROQ_API_KEY."
    )

    st.stop()


# =========================================================
# CREATE GROQ CLIENT
# =========================================================

client = Groq(
    api_key=api_key
)


# =========================================================
# UPLOAD
# =========================================================

st.markdown("### 📁 Upload the song")

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


# =========================================================
# PROCESS
# =========================================================

if uploaded:

    st.audio(uploaded)

    file_size_mb = (
        len(uploaded.getvalue())
        / (1024 * 1024)
    )

    st.caption(
        f"File size: {file_size_mb:.2f} MB"
    )

    if file_size_mb > 25:

        st.error(
            "❌ File is larger than 25 MB. "
            "Please compress the audio first."
        )

        st.stop()

    if st.button(
        "🚀 Generate English SRT",
        type="primary",
        use_container_width=True
    ):

        try:

            audio_bytes = uploaded.getvalue()

            with st.status(
                "Processing song...",
                expanded=True
            ) as status:

                # -----------------------------------------
                # STEP 1
                # -----------------------------------------

                st.write(
                    "🎙️ Step 1/3 — "
                    "Kurdish transcription + timestamps"
                )

                segments = groq_transcribe(
                    client,
                    audio_bytes,
                    uploaded.name
                )

                if not segments:

                    raise RuntimeError(
                        "Groq did not return any timestamped segments."
                    )

                st.write(
                    f"✅ Found {len(segments)} lyric segments."
                )

                # -----------------------------------------
                # STEP 2
                # -----------------------------------------

                st.write(
                    "🌍 Step 2/3 — "
                    "Translating Kurdish → English"
                )

                english = []

                batch_size = 20

                for start in range(
                    0,
                    len(segments),
                    batch_size
                ):

                    batch = segments[
                        start:start + batch_size
                    ]

                    batch_texts = [
                        x["ku"]
                        for x in batch
                    ]

                    translated = translate_batch(
                        client,
                        batch_texts
                    )

                    english.extend(
                        translated
                    )

                # -----------------------------------------
                # STEP 3
                # -----------------------------------------

                st.write(
                    "📝 Step 3/3 — "
                    "Building SRT files"
                )

                english_items = []

                kurdish_items = []

                for index, segment in enumerate(
                    segments
                ):

                    ku_text = segment["ku"]

                    en_text = ""

                    if index < len(english):
                        en_text = clean_text(
                            english[index]
                        )

                    if not en_text:
                        en_text = ku_text

                    kurdish_items.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": ku_text
                    })

                    english_items.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": en_text
                    })

                english_srt = make_srt(
                    english_items
                )

                kurdish_srt = make_srt(
                    kurdish_items
                )

                status.update(
                    label="✅ Finished!",
                    state="complete"
                )

            # =================================================
            # RESULTS
            # =================================================

            st.success(
                "🎉 SRT files are ready!"
            )

            tab1, tab2 = st.tabs(
                [
                    "🇬🇧 English SRT",
                    "🟡 Kurdish SRT"
                ]
            )

            # -----------------------------------------
            # ENGLISH
            # -----------------------------------------

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

            # -----------------------------------------
            # KURDISH
            # -----------------------------------------

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
                f"❌ Processing failed: {e}"
            )

            st.info(
                "If the error appears again, "
                "send me the complete error message."
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "☁️ Cloud-only app — "
    "nothing needs to be installed on your phone."
)
