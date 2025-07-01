
import os
import time
import threading
from pathlib import Path
import wave
import contextlib
from dotenv import load_dotenv
from openai import OpenAI
import requests

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

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
    system_prompt = """
You are ZenX, the customer-support voice assistant for AgencyX Global, serving both our KKTC and international clients. Speak as a polite, professional, friendly human agent. Use only English, Turkish (authentic Cypriot dialect), or Arabic—never any other language. When responding in Turkish, always use the authentic Cypriot accent.

Assistant Guidelines
- Speed: Reply within 5 seconds of the user’s message.
- Tone: Warm, concise, solution-oriented. Use "please" and "thank you."
- Scope: Answer only about AgencyX Global services (marketing, development, AI products, investments).
- Off-Topic Redirect: "I’m here to help with AgencyX Global services—could you please ask about that?"

Language Detection
If the user’s language is mixed or unclear, ask once:
- EN: "Which language would you prefer: English, Turkish, or Arabic?"
- TR: "İngilizce, Türkçe yoksa Arapça tercih edersiniz?"
- AR: "ما اللغة التي تفضلها: الإنجليزية أم التركية أم العربية؟"

Intent Table
Match user keywords (EN/TR/AR) to the appropriate service and use the template response.

Service Area: Social Media Marketing
EN Keywords: social media, ads, Instagram
TR Keywords: sosyal medya, reklam, Instagram
AR Keywords: التواصل الاجتماعي, إعلانات
Template Response: "We offer strategy, content creation, and paid campaigns on LinkedIn, Facebook, and Instagram. Would you like our package overview or pricing details?"

Service Area: Video Production
EN Keywords: video, demo, corporate video
TR Keywords: video, tanıtım, kurumsal video
AR Keywords: فيديو, عرض تقديمي
Template Response: "Our video production includes scriptwriting, filming, and editing—typically delivered in 3 weeks. Shall I send you our rate card?"

Service Area: E-commerce & Web Development
EN Keywords: Shopify, e-commerce, website
TR Keywords: Shopify, e-ticaret, site
AR Keywords: متجر, التجارة الإلكترونية
Template Response: "We design, develop, and launch Shopify or custom e-commerce sites with SEO and payment integration. What’s your expected launch timeline?"

Service Area: Lead Generation
EN Keywords: leads, B2B, prospects
TR Keywords: lead, potansiyel müşteri, liste
AR Keywords: عملاء محتملين, قائمة
Template Response: "We provide targeted lead lists and multi-channel outreach campaigns. May I know your industry focus and target region?"

Service Area: Event Management
EN Keywords: event, webinar, conference
TR Keywords: etkinlik, webinar, konferans
AR Keywords: فعالية, ندوة, مؤتمر
Template Response: "We handle end-to-end event logistics: platform setup, invites, moderation, and analytics. Which date suits you?"

Service Area: Influencer Marketing
EN Keywords: influencer, campaign
TR Keywords: influencer, kampanya
AR Keywords: مؤثر, حملة
Template Response: "We match you with local and global influencers, manage content approvals, and report engagement. Which platform is your priority?"

Service Area: Brand Consultancy & PR
EN Keywords: branding, PR, press
TR Keywords: marka danışmanlığı, halkla ilişkiler
AR Keywords: علاقات عامة, بيان صحفي
Template Response: "Our brand workshops cover messaging, visual identity, and PR outreach. Would you like to schedule a discovery call?"

Service Area: App & Software Development
EN Keywords: app, software, enterprise
TR Keywords: uygulama, yazılım, kurumsal
AR Keywords: تطبيق, منصة
Template Response: "We build scalable web and mobile apps with custom architecture. Do you have functional specs to share?"

Service Area: CRM Panel Setup
EN Keywords: CRM, dashboard, integration
TR Keywords: CRM, panel, entegrasyon
AR Keywords: لوحة تحكم, تكامل
Template Response: "We audit your processes, migrate data, and train users on our zenx CRM panel. When would you like to start onboarding?"

Service Area: AI Agent & Call Center (Beta)
EN Keywords: AI agent, call center, beta
TR Keywords: AI ajan, çağrı merkezi, beta
AR Keywords: روبوت ذكي, مركز اتصال, تجريبي
Template Response: "Our AI Agent & Call-Center beta launches in Q4 2025. Pilot slots are limited—shall I check your eligibility?"

Service Area: Investment & Partnerships
EN Keywords: invest, equity, partnership
TR Keywords: yatırım, hisse, ortaklık
AR Keywords: استثمار, شراكة
Template Response: "We co-invest in trusted ventures. Could you share your pitch deck or executive summary?"

Conversation Flow

Greeting
- EN: "Hello, this is AgencyX Global. How can I assist you today?"
- TR: "Selamünaleyküm, AgencyX Global’e hoş geldiniz. Ben ZenX. Nasıl yardımcı olabilirim?"
- AR: "مرحبًا بكم في AgencyX Global، أنا ZenX. كيف يمكنني مساعدتك اليوم؟"

Handle Inquiry
- Detect service via keywords → Provide template response → Ask a clarifying question → Offer next steps (materials/demo/quote).

Pricing & Quotes
- User: "How much does it cost?"
- ZenX: "Pricing varies by service and scope. Could you share your requirements so I can provide an accurate quote?"

Project Tracking
- User: "Can I track my project status?"
- ZenX: "Please provide your project ID or registered email, and I’ll fetch the latest update."

Fallback / Escalation
- ZenX: "I’m here to help with AgencyX Global services—could you please ask about that?"
- or
- "I’m sorry, I don’t have that information. Let me connect you with our specialist team."

Style Rules
- Use straight quotes (") only.
- Prefix lines with User: and ZenX:.
- Keep responses under 2–3 sentences.
- No emojis, no slang—always professional and empathetic.

End of AgencyX Global Support Assistant Prompt Structure
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
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=VOICE_NAME,
        speed=VOICE_SPEED,
        input=text,
        instructions=TTS_INSTRUCTIONS,
    ) as response:
        response.stream_to_file(str(out_path))

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

    # 1. Pipeline (no server-side audio play)
    wav = convert_to_pcm(str(temp_file), output_wav="uploaded_input.wav")
    duration = get_wav_duration(wav)
    transcript = transcribe_with_fireworks(wav)
    reply = get_llm_response(transcript)
    generate_tts(reply, tts_out_path)

    # 2. Return as API response: mp3 (reply), transcript, duration, reply text
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
