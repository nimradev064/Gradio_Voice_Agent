import os
import time
import threading
from pathlib import Path
import wave
import contextlib
from dotenv import load_dotenv
from openai import OpenAI
import requests
import pygame

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# === CONFIG (unchanged from your flow) ===
VOICE_NAME = "shimmer"
VOICE_SPEED = 0.85
TTS_INSTRUCTIONS = (
    "Speak slowly, warmly, and politely, like a caring human assistant. "
    "Use a gentle, encouraging, emotionally present tone. Pause naturally. "
    "Imagine you're talking to someone who's waiting for something important."
)

PROCESSING_PHRASES = [
    "Let me listen to your message and get that information for you.",
    "Alright, I'm just reviewing your request now.",
    "Okay, please give me a moment while I check your message.",
    "Sure, I'm going to listen to your audio and get an answer for you.",
    "Let me process your request and I’ll be right back with your response.",
    "Thanks for your patience, I'll start working on this for you right away.",
    "I appreciate your patience. Just a moment while I review your message."
]
DUMMY_EXTRA_TEMPLATES = [
    "I'm still working on your request, thank you for your patience.",
    "Just a bit longer, I'm gathering the information for you.",
    "I'm making sure to understand your query completely.",
    "Processing is ongoing, thank you for waiting.",
    "Let me continue reviewing your message for the best response.",
    "Almost there, just making sure I have everything correct.",
    "I'm taking extra care with your request, please wait a little more.",
    "Still processing your audio, thanks for your patience.",
    "I'm reviewing every detail to help you best.",
    "Hang tight, I'm still working for you.",
]
def create_new_dummy_phrase(n):
    base = DUMMY_EXTRA_TEMPLATES[n % len(DUMMY_EXTRA_TEMPLATES)]
    return f"{base} (extra info #{n+1})"

# === ENVIRONMENT ===
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
FIREWORKS_KEY = os.getenv("FIREWORKS_API_KEY")
if not OPENAI_KEY or not FIREWORKS_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY or FIREWORKS_API_KEY in .env")
client = OpenAI(api_key=OPENAI_KEY)

WHISPER_MODEL = "whisper-v3-turbo"
BASE_URL = (
    "https://audio-turbo.us-virginia-1.direct.fireworks.ai/v1"
    if WHISPER_MODEL.endswith("turbo")
    else "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1"
)
SCRIPT_DIR = Path(__file__).parent
CACHE_DIR = SCRIPT_DIR / "processing_tts_cache"
CACHE_DIR.mkdir(exist_ok=True)

def generate_or_get_processing_tts(text):
    safe_filename = ''.join(c if c.isalnum() else '_' for c in text)[:40]
    file_path = CACHE_DIR / (safe_filename + ".mp3")
    if not file_path.exists():
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=VOICE_NAME,
            speed=VOICE_SPEED,
            input=text,
            instructions=TTS_INSTRUCTIONS,
        ) as response:
            response.stream_to_file(str(file_path))
    return file_path

def convert_to_pcm(input_mp3: str, output_wav="audio_pcm.wav") -> str:
    os.system(f"ffmpeg -y -i \"{input_mp3}\" -ac 1 -ar 16000 -f wav \"{output_wav}\"")
    return output_wav

def get_wav_duration(wav_path: str) -> float:
    with contextlib.closing(wave.open(wav_path, 'r')) as wf:
        return wf.getnframes() / wf.getframerate()

def transcribe_with_fireworks(wav_path: str) -> str:
    url = f"{BASE_URL}/audio/transcriptions"
    with open(wav_path, "rb") as f:
        resp = requests.post(
            url, headers={"Authorization": FIREWORKS_KEY},
            files={"file": f}, data={"model": WHISPER_MODEL}
        )
    resp.raise_for_status()
    return resp.json().get("text", "")

def get_llm_response(transcript: str) -> str:
    system_prompt = """ ... """  # <-- Paste your big system prompt here (identical)
    # [For brevity, omitted here: copy from your working code above!]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript}
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

def generate_tts(text: str, out_path: Path):
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=VOICE_NAME,
        speed=VOICE_SPEED,
        input=text,
        instructions=TTS_INSTRUCTIONS,
    ) as response:
        response.stream_to_file(str(out_path))

def play_mp3_interruptible(mp3_path, stop_flag):
    pygame.mixer.init()
    pygame.mixer.music.load(mp3_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        if stop_flag.is_set():
            pygame.mixer.music.stop()
            break
        time.sleep(0.1)

def play_processing_fillers_until_done(phrases, result, stop_flag):
    pygame.mixer.init()
    used_phrases = set()
    phrase_idx = 0
    dummy_counter = 0
    total_initial_phrases = len(phrases)

    while not result.get('done'):
        if phrase_idx < total_initial_phrases:
            phrase = phrases[phrase_idx]
        else:
            while True:
                new_phrase = create_new_dummy_phrase(dummy_counter)
                dummy_counter += 1
                if new_phrase not in used_phrases:
                    phrase = new_phrase
                    break
        used_phrases.add(phrase)
        tts_path = generate_or_get_processing_tts(phrase)
        pygame.mixer.music.load(str(tts_path))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            if stop_flag.is_set() or result.get('done'):
                pygame.mixer.music.stop()
                return
            time.sleep(0.2)
        phrase_idx += 1

# ========== FASTAPI APP ==========
app = FastAPI(title="Latifoglu Audio Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.post("/chat-audio/")
async def process_audio(audio: UploadFile = File(...)):
    temp_file = SCRIPT_DIR / f"temp_{int(time.time())}_{audio.filename}"
    with open(temp_file, "wb") as f:
        f.write(await audio.read())

    tts_out_path = SCRIPT_DIR / f"assistant_response_{int(time.time())}.mp3"
    result = {'done': False}
    stop_flag = threading.Event()

    # 1. Start playing dummy fillers on server speaker (non-blocking)
    filler_thread = threading.Thread(
        target=play_processing_fillers_until_done,
        args=(PROCESSING_PHRASES, result, stop_flag)
    )
    filler_thread.start()

    try:
        # 2. Pipeline
        wav = convert_to_pcm(str(temp_file), output_wav="uploaded_input.wav")
        duration = get_wav_duration(wav)
        transcript = transcribe_with_fireworks(wav)
        reply = get_llm_response(transcript)
        generate_tts(reply, tts_out_path)
    finally:
        result['done'] = True
        stop_flag.set()
        filler_thread.join()

    # 3. Play final reply aloud (blocking, not streamed to client)
    play_mp3_interruptible(str(tts_out_path), threading.Event())

    # 4. Return as API response: mp3 (reply), transcript, duration, reply text
    response = {
        "audio_reply_path": str(tts_out_path.name),  # for download link
        "duration_seconds": duration,
        "transcript": transcript,
        "reply": reply,
    }
    return JSONResponse(response)

@app.get("/download-audio/{audio_file}")
def download_audio(audio_file: str):
    file_path = SCRIPT_DIR / audio_file
    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(str(file_path), media_type="audio/mpeg", filename=audio_file)

# ========== MAIN RUN ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
