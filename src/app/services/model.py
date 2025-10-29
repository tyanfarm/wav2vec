from pydantic import BaseModel
from enum import Enum

class PronounciationStatus(Enum):
    MATCH = "match"
    SIMILAR = "similar"
    MISMATCH = "mismatch"

class ComparePhonemes(BaseModel):
    correct: str
    test: str
    letter_phoneme_map: list[dict[str, str]] | None = None