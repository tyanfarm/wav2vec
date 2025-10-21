from contextlib import asynccontextmanager
import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from scipy.io.wavfile import write as write_wav
from app.services.phoneme_extractor import PhonemeExtractor
from app.services.utils import save_upload_to_temp
from app.services.config import SAMPLE_RATE
import numpy as np

extractor = None 

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup
    print("Loading model...")
    global extractor 
    extractor = PhonemeExtractor()
    print("Model loaded!")

    # silent wav 1s
    silence = np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.int16)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        write_wav(tmp.name, SAMPLE_RATE, silence)
        warmup_file_path = tmp.name

    try:
        print(f"Running warm-up with dummy audio: {warmup_file_path}")
        _ = extractor.process(warmup_file_path)
        print("✅ Device is warm and ready to go!")
    finally:
        # 3. Dọn dẹp file tạm sau khi khởi động xong
        os.unlink(warmup_file_path)
        print(f"Cleaned up temporary file: {warmup_file_path}")
        
    yield
    
    print("🔌 Server shutting down.")

app = FastAPI(lifespan=lifespan)

@app.post("/phonemes")
async def extract_phonemes(file: UploadFile = File(...)):   # ... means required
    tmp_path = await save_upload_to_temp(file)

    try:
        phonemes = extractor.process(tmp_path)

        return {"phonemes": phonemes}
    finally:
        os.unlink(tmp_path)

@app.get("/health")
def health():
    return {"status": "ok"}
