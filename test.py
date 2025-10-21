from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import os
import io
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

input_text = ".february"
input_text = input_text.strip().replace(".", "")

# Jessica - Voice ID: cgSgspJ2msm6clMCkdW9
# Alice - Voice ID: Xb7hH8MSUJpSbSDYk0k2
audio = client.text_to_speech.convert(
    text=input_text,
    voice_id="Xb7hH8MSUJpSbSDYk0k2",
    model_id="eleven_flash_v2_5"
)
play(audio)

# Thu thập tất cả chunks vào memory
audio_chunks = []
for chunk in audio:
    if chunk:
        audio_chunks.append(chunk)

# Ghi một lần duy nhất
with open(f"./audio_files/elevenlabs/{input_text}.mp3", "wb") as f:
    f.write(b''.join(audio_chunks))


print(f"Audio đã được lưu thành {input_text}.mp3!")