from contextlib import asynccontextmanager
import os
import tempfile
import numpy as np
from fastapi import FastAPI, APIRouter, UploadFile, File, Body
from scipy.io.wavfile import write as write_wav
from app.services.phoneme_extractor import PhonemeExtractor
from app.services.utils import save_upload_to_temp
from app.services.config import SAMPLE_RATE
from app.services.model import ComparePhonemes

extractor = None 

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup
    print("Loading model...")
    global extractor
    extractor = PhonemeExtractor()
    print("Model loaded!")

    print("🚀 Server starting up, beginning warm-up...")

    # silent wav 1s
    silence = np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.int16)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        write_wav(tmp.name, SAMPLE_RATE, silence)
        warmup_file_path = tmp.name

    try:
        print(f"Running warm-up with dummy audio: {warmup_file_path}")
        _ = extractor.extract_phonemes(warmup_file_path)
        print("✅ Device is warm and ready to go!")
    finally:
        # Clear up temp file after warm-up
        os.unlink(warmup_file_path)
        print(f"Cleaned up temporary file: {warmup_file_path}")

    yield

    print("🔌 Server shutting down.")

app = FastAPI(lifespan=lifespan)
v1 = APIRouter(prefix="/v1")

@v1.post("/phonemes")
async def extract_phonemes(file: UploadFile = File(...)):   # ... means required
    tmp_path = await save_upload_to_temp(file)

    try:
        phonemes = extractor.extract_phonemes(tmp_path)

        return {"phonemes": phonemes}
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.unlink(tmp_path)

@v1.post("/phonemes/test")
async def test_extract_phonemes(payload: ComparePhonemes = Body(...)):
    correct_phoneme = payload.correct
    test_phoneme = payload.test

    try:
        result = extractor.compare_phonemes(correct_phoneme, test_phoneme)
        return {"result": result}
    
    except Exception as e:
        return {"error": str(e)}

@v1.get("/health")
def health():
    return {"status": "ok"}

app.include_router(v1)