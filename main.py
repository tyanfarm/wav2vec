import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
from typing import Tuple
import warnings
import sys
warnings.filterwarnings('ignore')

class PhonemeExtractor:
    def __init__(self, model_name: str = "bookbot/wav2vec2-ljspeech-gruut", device: str = None):
        print(f"Loading model {model_name}...")
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def load_audio(self, audio_path: str, sr: int = 16000) -> np.ndarray:
        audio, _ = librosa.load(audio_path, sr=sr)
        return audio
    
    def get_phoneme_embeddings(self, audio: np.ndarray) -> Tuple[torch.Tensor, str]:
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            phonemes = self.processor.batch_decode(predicted_ids)[0]
        
        return hidden_states.squeeze(0).cpu(), phonemes
    
    def process_file(self, audio_path: str) -> Tuple[torch.Tensor, str]:
        audio = self.load_audio(audio_path)
        return self.get_phoneme_embeddings(audio)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: phoneme_extractor <audio_file_path>")
        sys.exit(1)
    
    # Load model once
    extractor = PhonemeExtractor(device='cpu')
    
    # Process audio file
    audio_path = sys.argv[1]
    embeddings, phonemes = extractor.process_file(audio_path)
    
    print(f"\nPhonemes: {phonemes}")
    print(f"Embedding shape: {embeddings.shape}")