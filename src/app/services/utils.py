import os
import tempfile
from typing import Optional
from fastapi import UploadFile

async def save_upload_to_temp(file: UploadFile, suffix: Optional[str] = None) -> str:
    """Save temporary uploaded file and return its path."""
    if suffix is None:
        _, ext = os.path.splitext(file.filename or "")
        suffix = ext or ".bin"

    CHUNK = 1024 * 1024  # 1MB
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        while True:
            chunk = await file.read(CHUNK)
            if not chunk:
                break
            tmp.write(chunk)
        return tmp.name