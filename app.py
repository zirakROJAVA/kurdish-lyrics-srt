import io
import re
import html
import requests
import streamlit as st
from groq import Groq

st.set_page_config(page_title="Kurdish Lyrics → English SRT", page_icon="🎵", layout="centered")

st.title("🎵 Kurdish Lyrics → English SRT")
st.caption("Upload a Kurdish song → transcribe with timestamps → translate to English → download SRT")

# -----------------------------
# Helpers
# -----------------------------
def srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    if ms >= 1000:
        total += 1
        ms = 0
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text

def make_srt(items):
    blocks = []
    for i, item in enumerate(items, 1):
        start = item["start"]
        end = item["end"]
        text = clean_text(item["text"])
        if not text:
            continue
        blocks.append(
            f"{i}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n"
        )
    return "\n".join(blocks)

def groq_transcribe(client, audio_bytes, filename):
    # Whisper large-v3 returns segment timestamps in verbose_json.
    result = client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3",
        response_format="verbose_json",
        timestamp_granularities=["segment"],
        language="ku",
        temperature=0,
    )
    segments = []
    for seg in getattr(result, "segments", []) or []:
        text = clean_text(getattr(seg, "text", ""))
        if text:
            segments.append({
                "start": float(getattr(seg, "start", 0)),
                "end": float(getattr(seg, "end", 0)),
                "ku": text,
            })
    if not segments:
        # Some SDK/API versions return dictionaries.
        raw = result.model_dump() if hasattr(result, "model_dump") else result
        for seg in raw.get("segments", []):
            text = clean_text(seg.get("text", ""))
            if text:
                segments.append({
                    "start": float(seg.get("start", 0)),
                    "end": float(seg.get("end", 0)),
                    "ku": text,
                })
    return segments

def translate_batch(client, texts):
    prompt = """Translate each Kurdish Sorani lyric line into natural English.
Rules:
- Return EXACTLY one English line for each numbered input.
- Keep the numbering.
- Do not explain anything.
- Do not merge or split lines.
- Preserve names and poetic meaning as naturally as possible.
- Output only the translations.

INPUT:
"""
    for i, t in enumerate(texts, 1):
        prompt += f"{i}. {t}\n"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.15,
        messages=[
            {"role": "system", "content": "You are a precise Kurdish Sorani to English lyric translator."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or ""

    result = {}
    for line in content.splitlines():
        m = re.match(r"^\s*(\d+)\s*[\.\):-]\s*(.+?)\s*$", line)
        if m:
            result[int(m.group(1))] = clean_text(m.group(2))

    return [result.get(i, "") for i in range(1, len(texts) + 1)]

# -----------------------------
# UI
# -----------------------------
st.markdown("### 1) Groq API key")
api_key = st.text_input(
    "Paste your Groq API key",
    type="password",
    help="Create/copy your key from Groq. It is used only for this session."
)

if not api_key:
    st.info("Paste your Groq API key, then upload the song.")
    st.stop()

st.markdown("### 2) Upload the song")
uploaded = st.file_uploader(
    "MP3 / M4A / WAV / FLAC / OGG",
    type=["mp3", "m4a", "wav", "flac", "ogg"]
)

if uploaded:
    st.audio(uploaded)

    if st.button("🚀 Generate English.srt", type="primary", use_container_width=True):
        client = Groq(api_key=api_key)

        audio = uploaded.getvalue()

        # Groq's upload limit can vary by account/model. Warn early.
        if len(audio) > 24 * 1024 * 1024:
            st.error("This file is larger than 24 MB. Please use a smaller/compressed audio file.")
            st.stop()

        try:
            with st.status("Processing song…", expanded=True) as status:
                st.write("🎙️ Step 1/3 — Kurdish transcription + timestamps")
                segments = groq_transcribe(client, audio, uploaded.name)

                if not segments:
                    raise RuntimeError("No timestamped segments were returned.")

                st.write(f"Found {len(segments)} timed lyric segments.")

                st.write("🌍 Step 2/3 — Translating Kurdish → English")
                # Keep requests manageable.
                english = []
                batch_size = 25
                for start in range(0, len(segments), batch_size):
                    batch = segments[start:start + batch_size]
                    english.extend(translate_batch(client, [x["ku"] for x in batch]))

                st.write("📝 Step 3/3 — Building English.srt")
                items = []
                for seg, en in zip(segments, english):
                    items.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": en if en else seg["ku"]
                    })

                english_srt = make_srt(items)

                # Also create Kurdish SRT so the original transcription is available.
                kurdish_srt = make_srt([
                    {"start": x["start"], "end": x["end"], "text": x["ku"]}
                    for x in segments
                ])

                status.update(label="✅ Finished!", state="complete")

            st.success("English SRT is ready.")

            tab1, tab2 = st.tabs(["English SRT", "Kurdish SRT"])
            with tab1:
                st.code(english_srt, language="text")
                st.download_button(
                    "⬇️ Download English.srt",
                    data=english_srt.encode("utf-8"),
                    file_name="English.srt",
                    mime="application/x-subrip",
                    use_container_width=True,
                )

            with tab2:
                st.code(kurdish_srt, language="text")
                st.download_button(
                    "⬇️ Download Kurdish.srt",
                    data=kurdish_srt.encode("utf-8"),
                    file_name="Kurdish.srt",
                    mime="application/x-subrip",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.caption("If the error mentions the API key, model, file size, or unsupported audio format, send me the exact error text.")

st.divider()
st.caption("Cloud-only app: nothing needs to be installed on your phone.")
