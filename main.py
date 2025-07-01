# # # # import os
# # # # import asyncio
# # # # import json
# # # # import base64
# # # # import requests
# # # # import websockets
# # # # from dotenv import load_dotenv

# # # # load_dotenv()
# # # # API_KEY = os.getenv("ELEVENLABS_API_KEY")
# # # # MODEL_ID = "eleven_flash_v2_5"

# # # # def get_default_voice_id() -> str:
# # # #     url = "https://api.elevenlabs.io/v1/voices"
# # # #     headers = {"Accept": "application/json", "xi-api-key": API_KEY}
# # # #     resp = requests.get(url, headers=headers)
# # # #     resp.raise_for_status()
# # # #     for v in resp.json().get("voices", []):
# # # #         if not v.get("is_legacy", False):
# # # #             return v["voice_id"]
# # # #     raise RuntimeError("No default voice found")

# # # # def handle_audio_chunk(chunk_bytes: bytes, idx=[0]):
# # # #     filename = f"chunk_{idx[0]:03d}.mp3"
# # # #     with open(filename, "wb") as f:
# # # #         f.write(chunk_bytes)
# # # #     print(f"📥 Saved {filename}")
# # # #     idx[0] += 1

# # # # async def tts_ws_stream(text: str):
# # # #     voice_id = get_default_voice_id()
# # # #     print(f"🔊 Using voice_id: {voice_id}")
# # # #     uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={MODEL_ID}"
# # # #     headers = {"xi-api-key": API_KEY}

# # # #     async with websockets.connect(uri, additional_headers=headers) as ws:
# # # #         # INITIAL HANDSHAKE
# # # #         await ws.send(json.dumps({
# # # #             "text": " ",
# # # #             "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
# # # #             "generation_config": {"chunk_length_schedule": [50, 120, 160]}
# # # #         }))

# # # #         async def listener():
# # # #             while True:
# # # #                 msg = await ws.recv()
# # # #                 data = json.loads(msg)

# # # #                 # Only decode if 'audio' exists and is not None
# # # #                 audio_b64 = data.get("audio")
# # # #                 if audio_b64:
# # # #                     try:
# # # #                         chunk = base64.b64decode(audio_b64)
# # # #                         handle_audio_chunk(chunk)
# # # #                     except Exception as e:
# # # #                         print("⚠️ Error decoding audio chunk:", e)

# # # #                 # Final signal: stop loop
# # # #                 if data.get("isFinal"):
# # # #                     print("✅ Finished streaming")
# # # #                     break

# # # #         listen_task = asyncio.create_task(listener())

# # # #         # STREAM TEXT IN CHUNKS
# # # #         for part in text.split(". "):
# # # #             await ws.send(json.dumps({"text": part.strip() + ". ", "try_trigger_generation": True}))

# # # #         # FLUSH buffer & wait for listener to finish
# # # #         await ws.send(json.dumps({"text": ""}))
# # # #         await listen_task

# # # # if __name__ == "__main__":
# # # #     sample_text = (
# # # #         "Hello world. This is streamed with alignment using your default ElevenLabs voice."
# # # #     )
# # # #     asyncio.run(tts_ws_stream(sample_text))

# # # # import os
# # # # import asyncio
# # # # import json
# # # # import base64
# # # # import requests
# # # # import websockets
# # # # import httpx
# # # # from dotenv import load_dotenv

# # # # load_dotenv()
# # # # ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# # # # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# # # # MODEL_ID = "eleven_flash_v2_5"

# # # # def get_default_voice_id():
# # # #     url = "https://api.elevenlabs.io/v1/voices"
# # # #     headers = {"Accept": "application/json", "xi-api-key": ELEVEN_API_KEY}
# # # #     resp = requests.get(url, headers=headers)
# # # #     resp.raise_for_status()
# # # #     for v in resp.json().get("voices", []):
# # # #         if not v.get("is_legacy", False):
# # # #             return v["voice_id"]
# # # #     raise RuntimeError("No default voice found")

# # # # async def transcribe_with_openai(chunk_bytes, chunk_name):
# # # #     url = "https://api.openai.com/v1/audio/transcriptions"
# # # #     headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
# # # #     files = {
# # # #         "file": (chunk_name, chunk_bytes, "audio/mp3"),
# # # #         "model": (None, "whisper-1"),
# # # #         "response_format": (None, "text")
# # # #     }
# # # #     async with httpx.AsyncClient() as client:
# # # #         resp = await client.post(url, headers=headers, files=files, timeout=60)
# # # #         resp.raise_for_status()
# # # #         return resp.text.strip()

# # # # async def handle_audio_chunk_and_transcribe(chunk_bytes, idx):
# # # #     chunk_name = f"chunk_{idx:03d}.mp3"
# # # #     # Save audio chunk to disk (for debugging, can be removed)
# # # #     with open(chunk_name, "wb") as f:
# # # #         f.write(chunk_bytes)
# # # #     print(f"📥 Saved {chunk_name}")
# # # #     transcript = await transcribe_with_openai(chunk_bytes, chunk_name)
# # # #     print(f"📝 Transcript {chunk_name}: {transcript}")
# # # #     return transcript

# # # # async def tts_ws_stream_and_transcribe(text):
# # # #     voice_id = get_default_voice_id()
# # # #     print(f"🔊 Using voice_id: {voice_id}")
# # # #     uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={MODEL_ID}"
# # # #     headers = {"xi-api-key": ELEVEN_API_KEY}
# # # #     transcripts = []

# # # #     async with websockets.connect(uri, additional_headers=headers) as ws:
# # # #         await ws.send(json.dumps({
# # # #             "text": " ",
# # # #             "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
# # # #             "generation_config": {"chunk_length_schedule": [50, 120, 160]}
# # # #         }))
# # # #         idx = 0
# # # #         finished = False

# # # #         async def listener():
# # # #             nonlocal idx, finished
# # # #             while True:
# # # #                 msg = await ws.recv()
# # # #                 data = json.loads(msg)
# # # #                 audio_b64 = data.get("audio")
# # # #                 if audio_b64:
# # # #                     try:
# # # #                         chunk = base64.b64decode(audio_b64)
# # # #                         # Launch transcription in background
# # # #                         t = asyncio.create_task(handle_audio_chunk_and_transcribe(chunk, idx))
# # # #                         transcripts.append(t)
# # # #                         idx += 1
# # # #                     except Exception as e:
# # # #                         print("⚠️ Error decoding/transcribing audio chunk:", e)
# # # #                 if data.get("isFinal"):
# # # #                     finished = True
# # # #                     print("✅ Finished streaming")
# # # #                     break

# # # #         listen_task = asyncio.create_task(listener())

# # # #         # Stream text in sentences
# # # #         for part in text.split(". "):
# # # #             await ws.send(json.dumps({"text": part.strip() + ". ", "try_trigger_generation": True}))

# # # #         # End stream and wait for all to finish
# # # #         await ws.send(json.dumps({"text": ""}))
# # # #         await listen_task

# # # #         # Gather all transcripts
# # # #         results = await asyncio.gather(*transcripts)
# # # #         print("\n====== All Transcripts ======\n", "\n".join(results))
# # # #         return results

# # # # if __name__ == "__main__":
# # # #     sample_text = (
# # # #         "Hello world. This is streamed with alignment using your default ElevenLabs voice. "
# # # #         "You can modify this text to test longer sentences."
# # # #     )
# # # #     asyncio.run(tts_ws_stream_and_transcribe(sample_text))


# # # # stt

# # # import os
# # # import requests
# # # from pydub import AudioSegment
# # # from dotenv import load_dotenv

# # # load_dotenv()
# # # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# # # WHISPER_MODEL = "whisper-1"

# # # # =========== CONFIG ===============
# # # INPUT_MP3 = "audio.mp3"   # Path to your source mp3
# # # CHUNK_LENGTH_MS = 10_000  # 10 seconds in milliseconds
# # # TMP_DIR = "chunks"
# # # os.makedirs(TMP_DIR, exist_ok=True)

# # # def split_audio_to_chunks(audio_path, chunk_length_ms):
# # #     print("🔪 Splitting audio...")
# # #     audio = AudioSegment.from_mp3(audio_path)
# # #     chunks = []
# # #     for i, start in enumerate(range(0, len(audio), chunk_length_ms)):
# # #         end = min(start + chunk_length_ms, len(audio))
# # #         chunk = audio[start:end]
# # #         filename = os.path.join(TMP_DIR, f"chunk_{i:03d}.mp3")
# # #         chunk.export(filename, format="mp3")
# # #         chunks.append(filename)
# # #     print(f"✅ Created {len(chunks)} chunks")
# # #     return chunks

# # # # def transcribe_with_whisper(chunk_path):
# # # #     url = "https://api.openai.com/v1/audio/transcriptions"
# # # #     headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
# # # #     files = {
# # # #         "file": (os.path.basename(chunk_path), open(chunk_path, "rb"), "audio/mp3")
# # # #     }
# # # #     data = {"model": WHISPER_MODEL, "response_format": "text"}
# # # #     try:
# # # #         resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
# # # #         resp.raise_for_status()
# # # #         transcript = resp.text.strip()
# # # #         print(f"📝 Transcript ({os.path.basename(chunk_path)}): {transcript}")
# # # #         return transcript
# # # #     except Exception as e:
# # # #         print(f"⚠️ Whisper transcription error: {e}")
# # # #         return None


# # # def transcribe_with_whisper(chunk_path: str):
# # #     audio = AudioSegment.from_mp3(chunk_path)
# # #     if audio.duration_seconds < 0.1:
# # #         print(f"⚠️ Skipping short chunk: {chunk_path}")
# # #         return None

# # #     url = "https://api.openai.com/v1/audio/transcriptions"
# # #     headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
# # #     files = {
# # #         "file": (os.path.basename(chunk_path), open(chunk_path, "rb"), "audio/mp3")
# # #     }
# # #     data = {"model": "whisper-1", "response_format": "text"}

# # #     resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
# # #     resp.raise_for_status()
# # #     return resp.text.strip()


# # # if __name__ == "__main__":
# # #     # 1. Split audio
# # #     chunks = split_audio_to_chunks(INPUT_MP3, CHUNK_LENGTH_MS)

# # #     # 2. Transcribe each chunk
# # #     transcripts = []
# # #     for chunk_file in chunks:
# # #         text = transcribe_with_whisper(chunk_file)
# # #         transcripts.append(text)

# # #     # 3. (Optional) Combine all and print
# # #     print("\n=== Full Transcript ===")
# # #     print(" ".join([t for t in transcripts if t]))


# # # import asyncio
# # # from openai import AsyncOpenAI
# # # from openai.helpers import LocalAudioPlayer
# # # from dotenv import load_dotenv
# # # load_dotenv()

# # # openai = AsyncOpenAI()

# # # # -------- TTS (Text-to-Speech) --------
# # # # async def tts():
# # # #     async with openai.audio.speech.with_streaming_response.create(
# # # #         model="gpt-4o-mini-tts",
# # # #         voice="coral",
# # # #         input="Today is a wonderful day to build something people love!",
# # # #         instructions="Speak in a cheerful and positive tone.",
# # # #         response_format="pcm",
# # # #     ) as response:
# # # #         await LocalAudioPlayer().play(response)

# # # # -------- STT (Speech-to-Text) --------
# # # async def stt(audio_path):
# # #     with open(audio_path, "rb") as audio_file:
# # #         response = await openai.audio.transcriptions.create(
# # #             file=audio_file,
# # #             model="whisper-1",          # OpenAI's STT model
# # #             response_format="text",     # Returns only the transcribed text
# # #         )
# # #         print("Transcription:", response)

# # # # -------- Main Logic --------
# # # async def main():
# # #     # TTS: Play speech from text
# # #     # await tts()
    
# # #     # STT: Transcribe audio file
# # #     # Replace 'your_audio_file.wav' with your real audio file path
# # #     await stt("audio.mp3")

# # # if __name__ == "__main__":
# # #     asyncio.run(main())


# # import asyncio
# # import time
# # from openai import AsyncOpenAI
# # from openai.helpers import LocalAudioPlayer
# # from dotenv import load_dotenv
# # from pydub import AudioSegment

# # load_dotenv()

# # openai = AsyncOpenAI()

# # def get_audio_duration(audio_path):
# #     audio = AudioSegment.from_file(audio_path)
# #     return audio.duration_seconds

# # # -------- STT (Speech-to-Text) --------
# # async def stt(audio_path):
# #     duration = get_audio_duration(audio_path)
# #     print(f"Audio Duration: {duration:.2f} seconds")

# #     start_time = time.time()
# #     with open(audio_path, "rb") as audio_file:
# #         response = await openai.audio.transcriptions.create(
# #             file=audio_file,
# #             model="whisper-1",          
# #             response_format="text",     
# #         )
# #     end_time = time.time()
# #     elapsed = end_time - start_time

# #     print("Transcription:", response)
# #     print(f"Transcription Time: {elapsed:.2f} seconds")

# # # -------- Main Logic --------
# # async def main():
# #     await stt("audio.mp3")

# # if __name__ == "__main__":
# #     asyncio.run(main())
#  ### Testing 



# # import os
# # import time
# # import subprocess
# # import requests
# # import openai
# # from dotenv import load_dotenv
# # openai.api_key = os.getenv("OPENAI_API_KEY")


# # load_dotenv()
# # API_KEY = os.getenv("FIREWORKS_API_KEY")
# # assert API_KEY, "Set FIREWORKS_API_KEY in .env"

# # # Choose your model and corresponding endpoint
# # MODEL = "whisper-v3-turbo"  # or "whisper-v3"
# # BASE_URL = "https://audio-turbo.us-virginia-1.direct.fireworks.ai/v1" if MODEL.endswith("turbo") else "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1"

# # def convert_to_pcm(audio_in, audio_out="audio_pcm.wav"):
# #     subprocess.run([
# #         "ffmpeg", "-y", "-i", audio_in,
# #         "-ac", "1", "-ar", "16000",
# #         "-f", "wav", audio_out
# #     ], check=True)
# #     return audio_out

# # def transcribe_async(pcm_path):
# #     url = f"{BASE_URL}/audio/transcriptions"
# #     with open(pcm_path, "rb") as f:
# #         files = {"file": f}
# #         data = {"model": MODEL}
# #         headers = {"Authorization": API_KEY}
# #         start = time.time()
# #         resp = requests.post(url, headers=headers, files=files, data=data)
# #         elapsed = time.time() - start

# #     if resp.status_code != 200:
# #         print("🛑 API Error:", resp.status_code, resp.text)
# #         resp.raise_for_status()

# #     text = resp.json().get("text", "")
# #     print("\n📄 Transcript:\n", text)
# #     print(f"🕒 Took {elapsed:.2f}s total using model `{MODEL}`")

# # def llm_respond(transcript: str):
# #     prompt = (
# #         "You are a helpful, human-like assistant. "
# #         "Here is the transcript of a client speaking:\n\n"
# #         f"{transcript}\n\n"
# #         "Please respond in a friendly, relevant way, addressing any questions or concerns implied in the transcript."
# #     )
# #     resp = openai.ChatCompletion.create(
# #         model="gpt-4o-mini",
# #         messages=[{"role": "user", "content": prompt}],
# #         temperature=0.7
# #     )
# #     return resp.choices[0].message.content.strip()

# # def main():
# #     mp3 = input("Enter path to MP3 file: ").strip()
# #     if not os.path.isfile(mp3):
# #         print("❗ File not found:", mp3)
# #         return

# #     print("🔁 Converting to PCM WAV...")
# #     pcm = convert_to_pcm(mp3)

# #     print("🔄 Transcribing audio...")
# #     transcript = transcribe_async(pcm)
# #     print("\n📄 Transcript:\n")
# #     print(transcript)

# #     print("\n🤖 Sending to LLM for assistant response...")
# #     response = llm_respond(transcript)
# #     print("\n📝 Assistant Response:\n")
# #     print(response)

# #     os.remove(pcm)

# # if __name__ == "__main__":
# #     main()



# # import os
# # import time
# # import subprocess
# # import requests
# # import openai
# # import wave
# # import contextlib
# # from dotenv import load_dotenv

# # load_dotenv()

# # # Set your OpenAI key
# # openai.api_key = os.getenv("OPENAI_API_KEY")
# # if not openai.api_key:
# #     raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")

# # # Fireworks transcription settings
# # MODEL = "whisper-v3-turbo"  # or "whisper-v3"
# # BASE_URL = (
# #     "https://audio-turbo.us-virginia-1.direct.fireworks.ai/v1"
# #     if MODEL.endswith("turbo")
# #     else "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1"
# # )

# # def convert_to_pcm(input_mp3: str, output_wav: str = "audio_pcm.wav") -> str:
# #     subprocess.run([
# #         "ffmpeg", "-y",
# #         "-i", input_mp3,
# #         "-ac", "1", "-ar", "16000",
# #         "-f", "wav", output_wav
# #     ], check=True)
# #     return output_wav

# # def get_wav_duration(wav_path: str) -> float:
# #     """Return duration in seconds of a WAV file."""
# #     with contextlib.closing(wave.open(wav_path, 'r')) as wf:
# #         frames = wf.getnframes()
# #         rate = wf.getframerate()
# #         return frames / float(rate)  # duration in seconds :contentReference[oaicite:1]{index=1}

# # def transcribe_with_fireworks(pcm_path: str) -> str:
# #     url = f"{BASE_URL}/audio/transcriptions"
# #     with open(pcm_path, "rb") as f:
# #         headers = {"Authorization": os.getenv("FIREWORKS_API_KEY") or ""}
# #         if not headers["Authorization"]:
# #             raise RuntimeError("Missing FIREWORKS_API_KEY in environment")
# #         files = {"file": f}
# #         data = {"model": MODEL}
# #         start = time.time()
# #         resp = requests.post(url, headers=headers, files=files, data=data)
# #         elapsed = time.time() - start

# #     if resp.status_code != 200:
# #         resp.raise_for_status()

# #     print(f"✅ Transcribed in {elapsed:.2f}s.")
# #     return resp.json().get("text", "")

# # def get_llm_response(transcript: str) -> str:
# #     resp = openai.ChatCompletion.create(
# #         model="gpt-4o-mini",
# #         messages=[{"role": "user", "content": transcript}],
# #         temperature=0.7
# #     )
# #     return resp.choices[0].message.content.strip()

# # def format_duration(secs: float) -> str:
# #     mins = int(secs // 60)
# #     rem = secs % 60
# #     return f"{mins}m {rem:.1f}s" if mins else f"{rem:.1f}s"

# # def main():
# #     mp3_path = input("Enter path to MP3 file: ").strip()
# #     if not os.path.isfile(mp3_path):
# #         print("❗ File not found:", mp3_path)
# #         return

# #     print("🔁 Converting to WAV (mono, 16 kHz)...")
# #     pcm = convert_to_pcm(mp3_path)

# #     duration_s = get_wav_duration(pcm)
# #     print("⏱ Audio duration:", format_duration(duration_s))

# #     print("🔄 Sending to Fireworks for transcription...")
# #     transcript = transcribe_with_fireworks(pcm)
# #     print("\n📄 Transcript:\n", transcript)

# #     print("\n🤖 Generating response via OpenAI...")
# #     response = get_llm_response(transcript)
# #     print("\n📝 Assistant Response:\n", response)

# #     try:
# #         os.remove(pcm)
# #     except OSError:
# #         pass

# # if __name__ == "__main__":
# #     main()



# #-----Final Code ----


# # import os
# # import time
# # import subprocess
# # import requests
# # from pathlib import Path
# # from dotenv import load_dotenv
# # from playsound import playsound
# # import openai 
# # import wave
# # import contextlib
# # from openai import OpenAI


# # load_dotenv()

# # # API keys
# # openai.api_key = os.getenv("OPENAI_API_KEY")
# # fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
# # if not openai.api_key:
# #     raise RuntimeError("Missing OPENAI_API_KEY in env or .env")
# # if not fireworks_api_key:
# #     raise RuntimeError("Missing FIREWORKS_API_KEY in env or .env")

# # # Setup clients and endpoints
# # MODEL = "whisper-v3-turbo"
# # BASE_URL = (
# #     "https://audio-turbo.us-virginia-1.direct.fireworks.ai/v1"
# #     if MODEL.endswith("turbo") else
# #     "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1"
# # )

# # def convert_to_pcm(input_mp3: str, output_wav: str = "audio_pcm.wav") -> str:
# #     subprocess.run([
# #         "ffmpeg", "-y", "-i", input_mp3,
# #         "-ac", "1", "-ar", "16000", "-f", "wav", output_wav
# #     ], check=True)
# #     return output_wav

# # def get_wav_duration(wav_path: str) -> float:
# #     with contextlib.closing(wave.open(wav_path, 'r')) as wf:
# #         return wf.getnframes() / float(wf.getframerate())

# # def transcribe_with_fireworks(pcm_path: str) -> str:
# #     url = f"{BASE_URL}/audio/transcriptions"
# #     with open(pcm_path, "rb") as f:
# #         headers = {"Authorization": fireworks_api_key}
# #         start = time.time()
# #         resp = requests.post(url, headers=headers, files={"file": f}, data={"model": MODEL})
# #         resp.raise_for_status()
# #         elapsed = time.time() - start
# #     print(f"✅ Transcribed in {elapsed:.2f}s.")
# #     return resp.json().get("text", "")

# # # def get_llm_response(transcript: str) -> str:
# # #     resp = openai.ChatCompletion.create(
# # #         model="gpt-4o-mini",
# # #         messages=[{"role": "user", "content": transcript}],
# # #         temperature=0.7
# # #     )
# # #     return resp.choices[0].message.content.strip()


# # def get_llm_response(transcript):
# #     client = OpenAI()  # Automatically picks up OPENAI_API_KEY

# #     resp = client.chat.completions.create(
# #         model="gpt-4o-mini",
# #         messages=[{"role": "user", "content": transcript}],
# #         temperature=0.7,
# #     )
# #     return resp.choices[0].message.content.strip()

# # # def generate_and_play_tts(text: str, out_path: Path):
# # #     try:
# # #         with openai.audio.speech.with_streaming_response.create(
# # #             model="gpt-4o-mini-tts",
# # #             voice="coral",
# # #             input=text,
# # #             instructions="Speak in a cheerful and positive tone."
# # #         ) as stream:
# # #             stream.stream_to_file(str(out_path))
# # #     except Exception as e:
# # #         raise RuntimeError(f"TTS generation error: {e}")

# # #     if not out_path.exists():
# # #         raise FileNotFoundError(f"Generated audio not found: {out_path}")
# # #     try:
# # #         playsound(str(out_path))
# # #     except Exception as e:
# # #         raise RuntimeError(f"Audio playback error: {e}")



# # def generate_and_play_tts(text: str, out_path: Path):
# #     try:
# #         client = OpenAI()
# #         with client.audio.speech.with_streaming_response.create(
# #             model="gpt‑4o‑mini‑tts",
# #             voice="coral",
# #             input=text,
# #             instructions="Speak in a cheerful and positive tone."
# #         ) as stream:
# #             stream.stream_to_file(str(out_path))
# #     except Exception as e:
# #         raise RuntimeError(f"TTS generation error: {e}")

# #     if not out_path.exists():
# #         raise FileNotFoundError(f"Generated audio not found: {out_path}")
# #     try:
# #         playsound(str(out_path))
# #     except Exception as e:
# #         raise RuntimeError(f"Audio playback error: {e}")

# # def main():
# #     mp3_path = input("Enter path to MP3 file: ").strip()
# #     if not os.path.isfile(mp3_path):
# #         print("❗ File not found:", mp3_path)
# #         return

# #     print("🔁 Converting to WAV (mono, 16 kHz)…")
# #     pcm = convert_to_pcm(mp3_path)
# #     duration = get_wav_duration(pcm)
# #     print(f"⏱ Duration: {int(duration//60)}m {duration%60:.1f}s")

# #     print("🔄 Transcribing audio…")
# #     transcript = transcribe_with_fireworks(pcm)
# #     print("\n📄 Transcript:\n", transcript)

# #     print("\n🤖 Generating GPT response…")
# #     response = get_llm_response(transcript)
# #     print("\n📝 Assistant Response:\n", response)

# #     # TTS output
# #     tts_path = Path(__file__).parent / "assistant_response.mp3"
# #     print("\n🎙 Generating and playing TTS…")
# #     generate_and_play_tts(response, tts_path)
# #     print("✅ Done.")

# #     try:
# #         os.remove(pcm)
# #     except OSError:
# #         pass

# # if __name__ == "__main__":
# #     main()

# #working

# # import os
# # import time
# # import subprocess
# # import requests
# # import wave
# # import contextlib
# # from pathlib import Path
# # from dotenv import load_dotenv
# # from openai import OpenAI
# # from pydub import AudioSegment
# # from pydub.playback import play

# # load_dotenv()

# # OPENAI_KEY = os.getenv("OPENAI_API_KEY")
# # FIREWORKS_KEY = os.getenv("FIREWORKS_API_KEY")

# # if not OPENAI_KEY or not FIREWORKS_KEY:
# #     raise RuntimeError("Missing OPENAI_API_KEY or FIREWORKS_API_KEY in .env")

# # # Instantiate OpenAI client
# # client = OpenAI(api_key=OPENAI_KEY)

# # # Fireworks transcription setup
# # WHISPER_MODEL = "whisper-v3-turbo"
# # BASE_URL = (
# #     "https://audio-turbo.us-virginia-1.direct.fireworks.ai/v1"
# #     if WHISPER_MODEL.endswith("turbo")
# #     else "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1"
# # )

# # def convert_to_pcm(input_mp3: str, output_wav="audio_pcm.wav") -> str:
# #     subprocess.run([
# #         "ffmpeg", "-y", "-i", input_mp3,
# #         "-ac", "1", "-ar", "16000", "-f", "wav", output_wav
# #     ], check=True)
# #     return output_wav

# # def get_wav_duration(wav_path: str) -> float:
# #     with contextlib.closing(wave.open(wav_path, 'r')) as wf:
# #         return wf.getnframes() / wf.getframerate()

# # def transcribe_with_fireworks(wav_path: str) -> str:
# #     url = f"{BASE_URL}/audio/transcriptions"
# #     with open(wav_path, "rb") as f:
# #         start = time.time()
# #         resp = requests.post(
# #             url, headers={"Authorization": FIREWORKS_KEY},
# #             files={"file": f}, data={"model": WHISPER_MODEL}
# #         )
# #     resp.raise_for_status()
# #     print(f"✅ Transcribed in {time.time() - start:.2f}s")
# #     return resp.json().get("text", "")

# # def get_llm_response(transcript: str) -> str:
# #     resp = client.chat.completions.create(
# #         model="gpt-4o-mini",
# #         messages=[{"role": "user", "content": transcript}],
# #         temperature=0.7,
# #     )
# #     return resp.choices[0].message.content.strip()

# # def generate_and_play_tts(text: str, out_path: Path):
# #     with client.audio.speech.with_streaming_response.create(
# #         model="gpt-4o-mini-tts",
# #         voice="coral",
# #         input=text,
# #         instructions="Speak in a cheerful and positive tone."
# #     ) as response:
# #         response.stream_to_file(str(out_path))

# #     audio = AudioSegment.from_file(out_path)
# #     play(audio)
# #     end = time.time()

# # def main():
# #     mp3_path = input("Enter path to MP3 file: ").strip()
# #     if not os.path.isfile(mp3_path):
# #         print("❗ File not found:", mp3_path)
# #         return

# #     print("🔁 Converting to WAV…")
# #     wav = convert_to_pcm(mp3_path)
# #     dur = get_wav_duration(wav)
# #     print(f"⏱ Duration: {int(dur//60)}m {dur % 60:.1f}s")
# #     start = time.time()  # Start timer before TTS generation

# #     print("🔄 Transcribing audio…")
# #     text = transcribe_with_fireworks(wav)
# #     print("\n📄 Transcript:\n", text)

# #     print("🤖 Generating GPT response…")
# #     reply = get_llm_response(text)
# #     print("\n📝 Assistant Response:\n", reply)
    
# #     end = time.time()  # End timer after playback
# #     print(f"🕑 Total response time (TTS generation + playback): {end - start:.2f} seconds")
# #     tts_file = Path(__file__).parent / "assistant_response.mp3"
# #     print("🎙 Generating and playing response…")
# #     generate_and_play_tts(reply, tts_file)
# #     print("✅ All done!")

# #     try:
# #         os.remove(wav)
# #     except OSError:
# #         pass

# # if __name__ == "__main__":
# #     main()


# import os
# import time
# import threading
# import requests
# import wave
# import contextlib
# from pathlib import Path
# from dotenv import load_dotenv
# from openai import OpenAI
# import pygame

# # --- Setup ---
# load_dotenv()
# OPENAI_KEY = os.getenv("OPENAI_API_KEY")
# FIREWORKS_KEY = os.getenv("FIREWORKS_API_KEY")
# if not OPENAI_KEY or not FIREWORKS_KEY:
#     raise RuntimeError("Missing OPENAI_API_KEY or FIREWORKS_API_KEY in .env")

# client = OpenAI(api_key=OPENAI_KEY)

# WHISPER_MODEL = "whisper-v3-turbo"
# BASE_URL = (
#     "https://audio-turbo.us-virginia-1.direct.fireworks.ai/v1"
#     if WHISPER_MODEL.endswith("turbo")
#     else "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1"
# )

# SCRIPT_DIR = Path(__file__).parent
# WAITING_MP3 = SCRIPT_DIR / "waiting.mp3"

# # --- Helpers ---
# def create_waiting_mp3(path):
#     print("🛠️  Generating waiting.mp3 (OpenAI TTS)…")
#     with client.audio.speech.with_streaming_response.create(
#         model="gpt-4o-mini-tts",
#         voice="coral",
#         input="Please wait while I process your request.",
#         instructions="Speak in a cheerful and positive tone."
#     ) as response:
#         response.stream_to_file(str(path))
#     print("✅ waiting.mp3 created!")

# def ensure_waiting_mp3():
#     if not WAITING_MP3.exists():
#         create_waiting_mp3(WAITING_MP3)

# def convert_to_pcm(input_mp3: str, output_wav="audio_pcm.wav") -> str:
#     os.system(f"ffmpeg -y -i \"{input_mp3}\" -ac 1 -ar 16000 -f wav \"{output_wav}\"")
#     return output_wav

# def get_wav_duration(wav_path: str) -> float:
#     with contextlib.closing(wave.open(wav_path, 'r')) as wf:
#         return wf.getnframes() / wf.getframerate()

# def transcribe_with_fireworks(wav_path: str) -> str:
#     url = f"{BASE_URL}/audio/transcriptions"
#     with open(wav_path, "rb") as f:
#         resp = requests.post(
#             url, headers={"Authorization": FIREWORKS_KEY},
#             files={"file": f}, data={"model": WHISPER_MODEL}
#         )
#     resp.raise_for_status()
#     return resp.json().get("text", "")

# def get_llm_response(transcript: str) -> str:
#     resp = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "Act like a human assistant. Respond to the transcript."},
#             {"role": "user", "content": transcript}
#         ],
#         temperature=0.7,
#     )
#     return resp.choices[0].message.content.strip()

# def generate_tts(text: str, out_path: Path):
#     with client.audio.speech.with_streaming_response.create(
#         model="gpt-4o-mini-tts",
#         voice="coral",
#         input=text,
#         instructions="Speak in a cheerful and positive tone."
#     ) as response:
#         response.stream_to_file(str(out_path))

# # --- Audio playback using pygame ---
# def play_mp3(mp3_path, stop_flag=None):
#     pygame.mixer.init()
#     pygame.mixer.music.load(mp3_path)
#     pygame.mixer.music.play()
#     while pygame.mixer.music.get_busy():
#         if stop_flag and stop_flag.is_set():
#             pygame.mixer.music.stop()
#             break
#         time.sleep(0.1)

# # --- Main flow in threads ---
# def background_pipeline(mp3_path, tts_out_path, result):
#     wav = convert_to_pcm(mp3_path)
#     result['duration'] = get_wav_duration(wav)
#     text = transcribe_with_fireworks(wav)
#     result['transcript'] = text
#     reply = get_llm_response(text)
#     result['reply'] = reply
#     generate_tts(reply, tts_out_path)
#     result['done'] = True
#     try:
#         os.remove(wav)
#     except OSError:
#         pass

# def play_dummy_until_done(dummy_audio_path, result, stop_flag):
#     # Play the dummy audio in a loop until the real answer is ready
#     while not result.get('done'):
#         play_mp3(dummy_audio_path, stop_flag)
#         if result.get('done'):
#             break

# def main():
#     ensure_waiting_mp3()
#     mp3_path = input("Enter path to MP3 file: ").strip()
#     if not os.path.isfile(mp3_path):
#         print("❗ File not found:", mp3_path)
#         return

#     tts_out_path = SCRIPT_DIR / "assistant_response.mp3"
#     result = {'done': False}
#     stop_flag = threading.Event()

#     pipeline_thread = threading.Thread(
#         target=background_pipeline, args=(mp3_path, tts_out_path, result)
#     )
#     dummy_thread = threading.Thread(
#         target=play_dummy_until_done, args=(str(WAITING_MP3), result, stop_flag)
#     )

#     pipeline_thread.start()
#     print("🎤 Speaking: 'Please wait while I process your request…'")
#     dummy_thread.start()

#     pipeline_thread.join()
#     result['done'] = True
#     stop_flag.set()  # Ask dummy thread to stop after this audio round
#     dummy_thread.join()  # Wait for dummy thread to exit after main thread sets done

#     # Now play the real response
#     print("🎙 Playing assistant's response…")
#     play_mp3(str(tts_out_path))

#     print("\n---")
#     print(f"⏱ Audio Duration: {result.get('duration', 0):.1f}s")
#     print("📄 Transcript:", result.get('transcript', ''))
#     print("📝 Assistant Response:", result.get('reply', ''))
#     print("✅ Done!")

# if __name__ == "__main__":
#     main()




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

# --- TTS filler generation/cache ---
def generate_or_get_processing_tts(text):
    safe_filename = ''.join(c if c.isalnum() else '_' for c in text)[:40]
    file_path = CACHE_DIR / (safe_filename + ".mp3")
    if not file_path.exists():
        print(f"🛠️  Generating processing TTS: {text}")
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
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Act like a human assistant. Respond to the transcript."},
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

# --- Background pipeline for transcript + response ---
def background_pipeline(mp3_path, tts_out_path, result):
    wav = convert_to_pcm(mp3_path)
    result['duration'] = get_wav_duration(wav)
    text = transcribe_with_fireworks(wav)
    result['transcript'] = text
    reply = get_llm_response(text)
    result['reply'] = reply
    generate_tts(reply, tts_out_path)
    result['done'] = True
    try:
        os.remove(wav)
    except OSError:
        pass

# --- Thread: play a series of human-like filler utterances, interrupt instantly ---
def play_processing_fillers_until_done(phrases, result, stop_flag):
    last_idx = None
    while not result.get('done'):
        # Pick a random phrase not same as last
        idxs = list(range(len(phrases)))
        if last_idx is not None and len(idxs) > 1:
            idxs.remove(last_idx)
        idx = random.choice(idxs)
        last_idx = idx
        phrase = phrases[idx]
        tts_path = generate_or_get_processing_tts(phrase)
        play_mp3_interruptible(str(tts_path), stop_flag)
        if result.get('done'):
            break

def main():
    mp3_path = input("Enter path to MP3 file: ").strip()
    if not os.path.isfile(mp3_path):
        print("❗ File not found:", mp3_path)
        return

    tts_out_path = SCRIPT_DIR / "assistant_response.mp3"
    result = {'done': False}
    stop_flag = threading.Event()

    pipeline_thread = threading.Thread(
        target=background_pipeline, args=(mp3_path, tts_out_path, result)
    )
    filler_thread = threading.Thread(
        target=play_processing_fillers_until_done, args=(PROCESSING_PHRASES, result, stop_flag)
    )

    pipeline_thread.start()
    print("🎤 Speaking: natural processing statements until reply is ready…")
    filler_thread.start()

    pipeline_thread.join()
    result['done'] = True
    stop_flag.set()  # Instantly interrupts dummy/filler speech
    filler_thread.join()

    # Play the real assistant reply
    print("🎙 Playing assistant's response…")
    play_mp3_interruptible(str(tts_out_path), threading.Event())

    print("\n---")
    print(f"⏱ Audio Duration: {result.get('duration', 0):.1f}s")
    print("📄 Transcript:", result.get('transcript', ''))
    print("📝 Assistant Response:", result.get('reply', ''))
    print("✅ Done!")

if __name__ == "__main__":
    main()
