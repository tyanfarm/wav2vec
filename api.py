from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from scipy.io.wavfile import write as write_wav
import numpy as np
import librosa
import tempfile
import os
import uvicorn

MODEL_NAME = "KoelLabs/xlsr-english-01"

class PhonemeExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Using device: {self.device}")
    
    def process_audio(self, audio_path: str):
        audio, _ = librosa.load(audio_path, sr=16000)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            phonemes = self.processor.batch_decode(predicted_ids)[0]
        
        # Clear tensor cache to free up memory
        torch.cuda.empty_cache()
        
        return phonemes

# Load model once at startup
print("Loading model...")
extractor = PhonemeExtractor()
print("Model loaded!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server starting up, beginning GPU warm-up...")
    
    # 1. Tạo một file audio im lặng tạm thời (ví dụ: 1 giây, 16kHz)
    samplerate = 16000
    duration = 1.0
    silence = np.zeros(int(samplerate * duration), dtype=np.int16)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        write_wav(tmp.name, samplerate, silence)
        warmup_file_path = tmp.name

    # 2. Chạy xử lý trên file audio tạm để kích hoạt kernel GPU
    try:
        print(f"Running warm-up with dummy audio: {warmup_file_path}")
        _ = extractor.process_audio(warmup_file_path)
        print("✅ GPU is warm and ready to go!")
    finally:
        # 3. Dọn dẹp file tạm sau khi khởi động xong
        os.unlink(warmup_file_path)
        print(f"Cleaned up temporary file: {warmup_file_path}")
        
    yield
    
    print("🔌 Server shutting down.")
    
app = FastAPI(lifespan=lifespan)

@app.post("/extract-phonemes")
async def extract_phonemes(file: UploadFile = File(...)):   # ... means required
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name 
        print(f"Saved uploaded file to {tmp_path}")
    
    try:
        phonemes = extractor.process_audio(tmp_path)
        return {"phonemes": phonemes}
    finally:
        os.unlink(tmp_path)
        
@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)