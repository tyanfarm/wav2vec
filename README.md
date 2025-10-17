# wav2vec

winget install ffmpeg

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

## Compile binary
- pip install pyinstaller
- pyinstaller --onefile --name phoneme_extractor main.py  
- dist\phoneme_extractor.exe audio_files\word_february.mp3

## Compile api binary
- pip install pyinstaller
- pyinstaller --onefile --name phoneme_uvicorn api.py  
- dist\phoneme_phoneme_uvicorn.exe