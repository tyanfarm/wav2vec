import torch
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from app.services.config import SAMPLE_RATE, MODEL_NAME

class PhonemeExtractor:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(self.device).eval()
        print(f"[PhonemeExtractor] Using device: {self.device}")

    def process(self, audio_path: str) -> str:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            phonemes = self.processor.batch_decode(predicted_ids)[0]

        # clear cache nếu có CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # chuẩn hóa ký tự hiếm nếu muốn
        return phonemes.replace("ʉ", "u")