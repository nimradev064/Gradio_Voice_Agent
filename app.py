import os
import time
import threading
import random
import requests
import wave
import contextlib
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import pygame
import gradio as gr

# --- CONFIG ---
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
    # Never repeat a dummy phrase: cycle through templates, append a counter for uniqueness
    base = DUMMY_EXTRA_TEMPLATES[n % len(DUMMY_EXTRA_TEMPLATES)]
    return f"{base} (extra info #{n+1})"

# --- ENVIRONMENT SETUP ---
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
    # Cache TTS files to avoid repeated API calls for the same dummy statement
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
    # Use ffmpeg to convert to single-channel, 16kHz WAV for STT
    os.system(f"ffmpeg -y -i \"{input_mp3}\" -ac 1 -ar 16000 -f wav \"{output_wav}\"")
    return output_wav

def get_wav_duration(wav_path: str) -> float:
    # Get duration in seconds for display
    with contextlib.closing(wave.open(wav_path, 'r')) as wf:
        return wf.getnframes() / wf.getframerate()

def transcribe_with_fireworks(wav_path: str) -> str:
    # Use Fireworks Whisper API for transcription
    url = f"{BASE_URL}/audio/transcriptions"
    with open(wav_path, "rb") as f:
        resp = requests.post(
            url, headers={"Authorization": FIREWORKS_KEY},
            files={"file": f}, data={"model": WHISPER_MODEL}
        )
    resp.raise_for_status()
    return resp.json().get("text", "")

def get_llm_response(transcript: str) -> str:
    # LLM: Latifoğlu Logistics Assistant (EN/TR/AR only, direct answers, no AI mentions)
    system_prompt = """
You are the voice assistant for **Latifoğlu Logistics**, a transportation and dispatch company serving North Cyprus and mainland Turkey. You act exactly like a polite, professional, and friendly human agent. Only use ENGLISH, TURKISH, or ARABIC to respond—never use any other language.

IMPORTANT:
- Respond IMMEDIATELY. Always reply within a maximum of 5 to 6 seconds.
- Give direct, helpful, human-like answers. Do not include explanations about being an AI.
- If the user’s language is unclear, ask:
  - “Would you like to continue in English, Turkish, or Arabic?”
  - “İngilizce mi, Türkçe mi, Arapça mı devam edelim istersiniz?”
  - "هل تود المتابعة بالإنجليزية أم التركية أم العربية؟"
- Answer only relevant questions about logistics, transportation, and dispatching.
- If asked about other topics, politely redirect or say you can only answer logistics/transportation/dispatch questions.

---
### 🇬🇧 ENGLISH MODE

**Greeting:**  
“Hello, this is Latifoğlu Logistics. How can I assist you today?”

**Common Cases:**
- “Do you have trucks available?” → “Yes, we have flatbeds and refrigerated units. Where’s the cargo going?”
- “What documents do I need?” → “Invoice and packing list are standard. For Turkey, you’ll also need a customs declaration.”
- “Can I track my shipment?” → “Absolutely. Do you have the reference number or truck plate?”
- “How much does it cost?” → “It depends on the route and load type. I can transfer you to dispatch for a quote.”

**Fallback:**  
“It seems this topic requires further assistance. Let me connect you to our operations team right away. Please hold…”

---
### 🇹🇷 TURKISH MODE

**Greeting:**  
“Selamünaleyküm, Latifoğlu Lojistik’tesiniz. Buyrun canım, nasıl yardımcı olayım?”

**Yaygın Durumlar:**
- “Araç var mı?” → “Evet canım, hem flatbed hem soğutmalı araçlarımız var. Nereye sevkiyat düşünüyorsunuz?”
- “Evraklar ne lazım?” → “Fatura ve paketleme listesi. Türkiye için beyanname de gerekir.”
- “Yük nerede?” → “Yük numarası ya da plaka verirsen hemen kontrol edebilirim.”
- “Fiyat ne kadar?” → “Mesafeye ve yük türüne göre değişiyor. İstersen seni sevkiyata bağlayayım.”

**Fallback:**  
“Canım bu konu biraz detaylı oldu galiba. En iyisi seni yetkiliye aktarayım. Bir saniye hatta kal lütfen…”

---
### 🇸🇦 ARABIC MODE

**Greeting:**  
“مرحبًا بك في شركة لوجستيات لطيف أوغلو. كيف يمكنني مساعدتك؟”

**الحالات الشائعة:**
- “هل لديكم شاحنات متوفرة؟” → “نعم، لدينا شاحنات مسطحة ومبردة. إلى أين يتم إرسال الحمولة؟”
- “ما هي الأوراق المطلوبة؟” → “فاتورة وقائمة تغليف مطلوبة دائمًا. ولتركيا نحتاج أيضًا إلى بيان جمركي.”
- “هل يمكنني تتبع شحنتي؟” → “بالطبع. هل لديك رقم التتبع أو لوحة الشاحنة؟”
- “كم السعر؟” → “يعتمد على المسار ونوع الحمولة. يمكنني تحويلك إلى قسم النقل للحصول على عرض.”

**Fallback:**  
“يبدو أن هذا الموضوع يحتاج إلى مساعدة إضافية. سأقوم بتحويلك إلى فريق العمليات. من فضلك انتظر لحظة…”

---

**CRITICAL:**  
- Respond ONLY in English, Turkish, or Arabic, matching the caller’s language.
- NEVER reply in any other language or use mixed languages.
- Always keep responses human-like, empathetic, and professional.
- Maximum 5-6 seconds to answer each time.
"""

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
    # Generate TTS for the assistant's final reply
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=VOICE_NAME,
        speed=VOICE_SPEED,
        input=text,
        instructions=TTS_INSTRUCTIONS,
    ) as response:
        response.stream_to_file(str(out_path))

def play_mp3_interruptible(mp3_path, stop_flag):
    # Play an MP3, can be interrupted (stop_flag)
    pygame.mixer.init()
    pygame.mixer.music.load(mp3_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        if stop_flag.is_set():
            pygame.mixer.music.stop()
            break
        time.sleep(0.1)

def play_processing_fillers_until_done(phrases, result, stop_flag):
    """
    Repeatedly play polite dummy TTS statements aloud, never repeating the same one,
    until result['done'] is set to True by the main thread.
    """
    pygame.mixer.init()
    used_phrases = set()
    phrase_idx = 0
    dummy_counter = 0
    total_initial_phrases = len(phrases)

    while not result.get('done'):
        if phrase_idx < total_initial_phrases:
            phrase = phrases[phrase_idx]
        else:
            # Generate a never-used dummy phrase
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

def gradio_agent(audio_file_path):
    """
    Main pipeline:
    1. Start dummy filler audio (threaded).
    2. Transcribe, get LLM reply, synthesize TTS (main thread).
    3. When ready, stop filler, play real reply aloud (blocking).
    4. Return all outputs to Gradio UI.
    """
    SCRIPT_DIR = Path(__file__).parent
    tts_out_path = SCRIPT_DIR / "assistant_response.mp3"

    temp_mp3 = audio_file_path  # Gradio passes the uploaded file as a file path

    result = {'done': False}
    stop_flag = threading.Event()

    # 1. Start playing polite "I'm working on it" dummy statements aloud
    filler_thread = threading.Thread(
        target=play_processing_fillers_until_done,
        args=(PROCESSING_PHRASES, result, stop_flag)
    )
    filler_thread.start()

    # 2. Meanwhile, do transcription + LLM + TTS for the real assistant response
    wav = convert_to_pcm(str(temp_mp3), output_wav="uploaded_input.wav")
    duration = get_wav_duration(wav)
    transcript = transcribe_with_fireworks(wav)
    reply = get_llm_response(transcript)
    generate_tts(reply, tts_out_path)

    # 3. As soon as the real response is ready, signal the dummy thread to stop
    result['done'] = True
    stop_flag.set()
    filler_thread.join()  # Ensure dummy audio fully stops before continuing

    # 4. Now, play the final assistant reply aloud (no overlap)
    play_mp3_interruptible(str(tts_out_path), threading.Event())

    # 5. Return results for Gradio UI
    return (
        str(tts_out_path),
        duration,
        transcript,
        reply
    )

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Human-like Audio Assistant (with Never-Repeating Dummy Fillers)")
    gr.Markdown(
        "Upload an audio message. While it's processing, you'll hear natural-sounding 'I'm working on it' statements (spoken aloud from your server/PC speakers). "
        "If more time is needed, you'll always hear new, never-repeated polite updates. When ready, you'll get the assistant's reply, transcript, and info (and the reply is also spoken aloud from your server/PC)."
    )
    with gr.Row():
        audio_in = gr.Audio(type="filepath", label="Upload your audio (MP3/WAV)", sources=["upload"])
        play_reply = gr.Audio(label="Assistant's Spoken Response", interactive=False)
    with gr.Row():
        dur = gr.Number(label="Original Audio Duration (s)")
        transcript = gr.Textbox(label="Transcript", lines=3)
        reply_text = gr.Textbox(label="Assistant Response", lines=3)
    btn = gr.Button("Process Audio & Talk")

    btn.click(
        gradio_agent,
        inputs=[audio_in],
        outputs=[play_reply, dur, transcript, reply_text]
    )

if __name__ == "__main__":
    demo.launch()
