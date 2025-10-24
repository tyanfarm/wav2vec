MODEL_NAME = "KoelLabs/xlsr-english-01"
SAMPLE_RATE = 16000
HOST = "0.0.0.0"
PORT = 8000
SIMILAR_PHONEME = [
    # === STOPS (PLOSIVES) ===
    ["p", "pʰ"],
    ["d", "ɾ", "ɾ̃"],
    ["k", "kʰ"],
    ["ʔ"],

    # === AFFRICATES ===
    ["tʃ", "dʒ"],

    # === FRICATIVES BY PLACE ===
    ["θ", "θʰ", "ð"],
    ["s", "sʰ"],
    ["x", "ɣ"],
    ["h", "ɦ"],

    # === NASALS (place-matched + syllabic) ===
    ["m", "m̩"],
    ["n", "n̩"],
    ["ŋ", "ŋ̍"],

    # === LATERALS / RHOTICS / GLIDES ===
    ["l", "l̩"],
    ["ɹ", "r"],
    
    # === FRONT VOWELS ===
    ["i", "ĩ", "ĩ", "ɪ"],
    ["e", "ɛ", "ɛ̃"],
    ["æ", "æ̃", "a"],

    # === CENTRAL VOWELS ===
    ["ə", "ə̃", "ə̥", "ɜ", "ʌ", "ɨ"],
    ["ɚ", "ɝ"],

    # === BACK VOWELS ===
    ["u", "ʊ", "ʊ̃", "ʉ"],
    ["o", "ɔ", "ɒ"],
    ["ɑ", "ɑ̃"],

    # === DIPHTHONGS ===
    ["oʊ", "oʊ̃", "əʊ", "aʊ"],
]

