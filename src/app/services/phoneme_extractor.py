import torch
import librosa
import difflib
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from app.services.model import PronounciationStatus

from app.services.config import SAMPLE_RATE, MODEL_NAME, SIMILAR_PHONEME

class PhonemeExtractor:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(self.device).eval()
        self.phoneme_groups = {}
        for group in SIMILAR_PHONEME:
            for phoneme in group:
                self.phoneme_groups[phoneme] = group
                
        default_tokens = set([s for group in SIMILAR_PHONEME for s in group])
        print(f"Default tokens: {default_tokens}")
        print(f"Lenghth of default tokens: {len(default_tokens)}")

        print(f"[PhonemeExtractor] Using device: {self.device}")

    def extract_phonemes(self, audio_path: str) -> str:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            phonemes = self.processor.batch_decode(predicted_ids)[0]

        # Clear cache torch.cuda
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Normalize specific phoneme characters
        return phonemes.replace("ʉ", "u").replace("ʔ", "").strip()
    
    """
    -- Problem: opcodes from difflib.SequenceMatcher will have case: ['dʒ', 'æ'] -> ['tʃ', 'ɛ']
    -> Solution: need to further split to 1-1 if possible: ['dʒ'->'tʃ'], ['æ'->'ɛ']
    """
    def compare_phonemes(self, correct_phonemes: str, test_phonemes: str, letter_phoneme_map: list[dict] = None) -> list[dict]:
        """
        Compare two phoneme strings and return a list of differences.
        """
        correct_tokens = self._tokenize_ipa(correct_phonemes)
        test_tokens = self._tokenize_ipa(test_phonemes)

        opcodes = difflib.SequenceMatcher(a=correct_tokens, b=test_tokens).get_opcodes()
        
        refined_opcodes = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'replace':
                refined_opcodes.extend(self._normalize_opcodes_per_token(correct_tokens, test_tokens, i1, i2, j1, j2))
            else:
                refined_opcodes.append((tag, i1, i2, j1, j2))

        opcodes = refined_opcodes
        result = []
        phoneme_statuses = []
        
        GREEN = '\033[92m'   # đúng
        YELLOW = '\033[93m'  # lỗi nhẹ
        RED = '\033[91m'     # lỗi nặng
        RESET = '\033[0m'
        colored_output = ""
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                current_segment = "".join(correct_tokens[i1:i2])
                current_status = PronounciationStatus.MATCH.value

                for idx in range(i1, i2):
                    phoneme_statuses.append((correct_tokens[idx], current_status))

                colored_output += GREEN + current_segment + RESET
                print(f"- Giống nhau: {correct_tokens[i1:i2]}")

            elif tag == 'replace':
                current_segment = "".join(correct_tokens[i1:i2])

                # Case: ['æ'] -> ['ɛ']
                if (i2 - i1 == 1) and (j2 - j1 == 1) and self._are_phonemes_similar(correct_tokens[i1], test_tokens[j1]):
                    current_status = PronounciationStatus.SIMILAR.value

                    colored_output += YELLOW + current_segment + RESET
                    print(f"- TƯƠNG TỰ (YELLOW): '{correct_tokens[i1]}' -> '{test_tokens[j1]}' (cùng nhóm)")
                else:
                    current_status = PronounciationStatus.MISMATCH.value

                    colored_output += RED + current_segment + RESET
                    print(f"- THAY THẾ (RED): {correct_tokens[i1:i2]} -> {test_tokens[j1:j2]}")

                for idx in range(i1, i2):
                    phoneme_statuses.append((correct_tokens[idx], current_status))

            elif tag == 'delete':
                # Tokens missing from correct_tokens & only in test_tokens
                for idx in range(i1, i2):
                    current_segment = correct_tokens[idx]
                    if self._is_semivowel(current_segment, idx):
                        current_status = PronounciationStatus.SIMILAR.value

                        colored_output += YELLOW + current_segment + RESET
                        print(f"- THIẾU SEMIVOWEL (YELLOW): '{current_segment}' tại vị trí token {idx}")
                    else:
                        current_status = PronounciationStatus.MISMATCH.value

                        colored_output += RED + current_segment + RESET
                        print(f"- XÓA (RED): '{current_segment}' tại vị trí token {idx}")

                    phoneme_statuses.append((current_segment, current_status))

            elif tag == 'insert':
                # inserted token in test_tokens
                current_segment = "".join(test_tokens[j1:j2])
                # Check is_semivowel insertion
                if (j2 - j1 == 1) and self._is_semivowel(test_tokens[j1], i1):
                    current_status = PronounciationStatus.SIMILAR.value

                    colored_output += YELLOW + current_segment + RESET
                    print(f"- THÊM SEMIVOWEL (YELLOW): '{current_segment}' tại vị trí (theo correct) {i1}")
                else:
                    current_status = PronounciationStatus.MISMATCH.value

                    colored_output += RED + current_segment + RESET
                    print(f"- CHÈN (RED): {test_tokens[j1:j2]} tại vị trí (theo correct) {i1}")
                # For insert: we don't have corresponding phoneme in correct_tokens
                # Skip adding to phoneme_statuses

            result.append({
                "phonemes": current_segment,
                "status": current_status
            })

        print(colored_output)

        # Print colored word if letter_phoneme_map is provided
        if letter_phoneme_map:
            self._print_colored_word(letter_phoneme_map, phoneme_statuses)

        return result
    
    def _print_colored_word(self, letter_phoneme_map: list[dict], phoneme_statuses: list[tuple]):
        """
        Print the word with letters colored based on their corresponding phoneme status.
        Handles cases where one letter maps to multiple phonemes.
        
        Args:
            letter_phoneme_map: List of dicts with 'phoneme' and 'letter' keys
            phoneme_statuses: List of (phoneme, status) tuples from comparison
        """
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        RESET = '\033[0m'
        
        # Build a mapping from phoneme index to status
        phoneme_status_map = {}
        phoneme_idx = 0
        for phoneme, status in phoneme_statuses:
            phoneme_status_map[phoneme_idx] = status
            phoneme_idx += 1
        
        colored_word = ""
        word = ""
        
        # Track current position in phoneme_statuses (skip spaces)
        current_phoneme_idx = 0
        prev_letter = None
        letter_statuses = []  # Collect statuses for current letter group
        
        # Iterate through letter_phoneme_map
        for i, mapping in enumerate(letter_phoneme_map):
            letter = mapping.get('letter', '')
            phoneme = mapping.get('phoneme', '')
            
            # Handle space between words (both phoneme and letter are empty)
            if not letter and not phoneme:
                # Flush any pending letter
                if prev_letter and letter_statuses:
                    word += prev_letter
                    colored_word += self._get_colored_letter(prev_letter, letter_statuses, GREEN, YELLOW, RED, RESET)
                    letter_statuses = []
                
                colored_word += " "
                word += " "
                prev_letter = None
                continue
            
            # Skip entries with no letter but has phoneme
            if not letter:
                current_phoneme_idx += 1
                continue
            
            # Get status for this phoneme position
            status = phoneme_status_map.get(current_phoneme_idx, PronounciationStatus.MATCH.value)
            
            # Check if this is a new letter or continuation of previous letter
            if letter != prev_letter:
                # Flush previous letter if exists
                if prev_letter and letter_statuses:
                    word += prev_letter
                    colored_word += self._get_colored_letter(prev_letter, letter_statuses, GREEN, YELLOW, RED, RESET)
                
                # Start new letter group
                prev_letter = letter
                letter_statuses = [status]
            else:
                # Same letter, accumulate status
                letter_statuses.append(status)
            
            current_phoneme_idx += 1
        
        # Flush last letter
        if prev_letter and letter_statuses:
            word += prev_letter
            colored_word += self._get_colored_letter(prev_letter, letter_statuses, GREEN, YELLOW, RED, RESET)
        
        print(f"\n📝 Word với letter highlight: {colored_word}")
        print(f"   Original word: {word}")

    def _get_colored_letter(self, letter: str, statuses: list, GREEN: str, YELLOW: str, RED: str, RESET: str) -> str:
        """
        Determine the color for a letter based on all phoneme statuses associated with it.
        Priority: MISMATCH > SIMILAR > MATCH
        """
        # If any phoneme is MISMATCH, color the whole letter RED
        if PronounciationStatus.MISMATCH.value in statuses:
            return RED + letter + RESET
        # If any phoneme is SIMILAR, color the letter YELLOW
        elif PronounciationStatus.SIMILAR.value in statuses:
            return YELLOW + letter + RESET
        # All phonemes are MATCH
        else:
            return GREEN + letter + RESET
    
    def _tokenize_ipa(self, s: str) -> list[str]:
        """Tokenize IPA string into list of phoneme tokens."""
        default_tokens = set([s for group in SIMILAR_PHONEME for s in group])
        tokens = []
        i = 0
        # Longest match: 3 -> 2 -> 1
        while i < len(s):
            match = None
            # Check 3 characters
            if i + 3 <= len(s) and s[i:i+3] in default_tokens:
                match = s[i:i+3]; i += 3
            # Check 2 characters
            elif i + 2 <= len(s) and s[i:i+2] in default_tokens:
                match = s[i:i+2]; i += 2
            else:
                # 1 character: if not in known tokens, still take as single token
                match = s[i]; i += 1
            tokens.append(match)
        return tokens
    
    def _normalize_opcodes_per_token(self, correct_tokens, test_tokens, i1, i2, j1, j2) -> list[tuple]:
        """Normalize opcodes for a block by splitting into 1–1 if possible."""
        opcodes = []
        len_correct = i2 - i1
        len_test = j2 - j1
        if len_correct == len_test:
            # Split to 1–1
            for k in range(len_correct):
                ci = i1 + k
                tj = j1 + k
                if correct_tokens[ci] == test_tokens[tj]:
                    opcodes.append(('equal', ci, ci+1, tj, tj+1))
                else:
                    opcodes.append(('replace', ci, ci+1, tj, tj+1))
        else:
            # Cannot split, keep original
            opcodes.append(('replace', i1, i2, j1, j2))
        return opcodes
        
    def _are_phonemes_similar(self, phoneme1, phoneme2) -> bool:
        """Check if two phonemes are similar based on SIMILAR_PHONEME groups."""
        if phoneme1 == phoneme2:
            return True
        return (phoneme1 in self.phoneme_groups) and (phoneme2 in self.phoneme_groups) and (self.phoneme_groups[phoneme1] is self.phoneme_groups[phoneme2])
    
    def _is_semivowel(self, token, token_index) -> bool:
        """Check if a token is a semivowel and not at the start."""
        semivowels = ["j", "w"]
        return (token in semivowels) and (token_index > 0)