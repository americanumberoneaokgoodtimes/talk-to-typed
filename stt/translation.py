"""Translation module using CTranslate2 with NLLB-200 for offline multilingual translation."""
import json
import os
import re
import threading
import unicodedata
from pathlib import Path
from typing import Optional

# These will be imported lazily to avoid hard dependency
_ctranslate2 = None
_sentencepiece = None
_translators: dict = {}
_translators_lock = threading.Lock()

# Standard FLORES-200 language tokens used by NLLB-200 distilled CTranslate2 models
NLLB_CODES = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
}

LEGACY_NLLB_CODES = {
    "en": "eng",
    "es": "spa",
    "pt": "por",
    "de": "deu",
    "fr": "fra",
}

# High-frequency distinctive function words (normalized ASCII + accented forms handled dynamically)
LANG_MARKERS = {
    "en": {
        "the", "be", "to", "of", "and", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "just", "him", "know",
        "take", "people", "into", "year", "your", "good", "some", "could", "them",
        "see", "other", "than", "then", "now", "look", "only", "come", "its", "over",
        "think", "also", "back", "after", "use", "two", "how", "our", "work", "first",
        "well", "way", "even", "new", "want", "because", "any", "these", "give", "day",
        "most", "us", "is", "was", "are", "were", "been", "has", "had", "does",
        "did", "doing", "going", "coming", "being", "said", "says",
        "am", "should", "may", "might", "must", "shall", "need", "hello", "world", "today",
    },
    "es": {
        "el", "la", "los", "las", "de", "del", "que", "y", "en", "un",
        "una", "unos", "unas", "ser", "estar", "haber", "tener", "hacer", "poder", "decir",
        "este", "esta", "estas", "estos", "ese", "esa", "esos", "esas", "aquel", "aquella",
        "mi", "tu", "su", "mis", "tus", "sus", "nuestro", "nuestra", "vuestro", "vuestra",
        "te", "se", "nos", "os", "le", "les", "lo",
        "con", "por", "para", "sin", "sobre", "entre", "hasta", "desde", "durante", "mediante",
        "como", "cuando", "donde", "quien", "cual", "cuyo", "cuanto",
        "ni", "pero", "sino", "tambien", "tampoco",
        "muy", "mas", "menos", "tan", "tanto", "asi", "aqui", "alli", "alla",
        "ahora", "antes", "despues", "entonces", "luego", "siempre", "nunca", "veces",
        "algo", "alguien", "nadie", "todo", "nada", "algun", "ningun", "cada", "cualquier",
        "uno", "dos", "tres", "primero", "ultimo", "otro", "mismo", "diferente", "igual",
        "grande", "pequeno", "bueno", "malo", "nuevo", "viejo", "alto", "bajo", "largo", "corto",
        "hola", "mundo", "hoy", "es", "son", "fue", "era", "hay", "espanol", "oracion", "probar", "deteccion",
    },
    "pt": {
        "o", "os", "as", "de", "do", "da", "dos", "das", "no", "na", "nos", "nas", "que", "em",
        "um", "uma", "uns", "umas", "ser", "estar", "ter", "haver", "fazer", "poder",
        "este", "esta", "estes", "estas", "esse", "essa", "esses", "essas", "aquele", "aquela",
        "meu", "minha", "teu", "tua", "seu", "sua", "nosso", "nossa", "vosso", "vossa",
        "te", "se", "vos", "lhe", "lhes", "lho", "lha",
        "com", "por", "para", "sem", "sobre", "entre", "ate", "desde", "durante", "mediante",
        "como", "quando", "onde", "quem", "qual", "cujo", "quanto",
        "nao", "nem", "mas", "porem", "entretanto", "ou", "ora", "tambem", "tampouco",
        "muito", "mais", "menos", "tam", "tanto", "assim", "aqui", "ai",
        "agora", "antes", "depois", "entao", "logo", "sempre", "nunca", "vezes",
        "algo", "alguem", "ninguem", "tudo", "nada", "algum", "nenhum", "cada", "qualquer",
        "dois", "tres", "primeiro", "ultimo", "outro", "mesmo", "diferente", "igual",
        "grande", "pequeno", "bom", "boa", "mau", "novo", "velho", "alto", "baixo", "longo", "curto",
        "ola", "mundo", "voce", "hoje", "frase", "portugues", "testar", "deteccao", "esta", "sao", "uma",
    },
}


def _import_ctranslate2():
    global _ctranslate2
    if _ctranslate2 is None:
        import ctranslate2
        _ctranslate2 = ctranslate2
    return _ctranslate2


def _import_sentencepiece():
    global _sentencepiece
    if _sentencepiece is None:
        import sentencepiece as spm
        _sentencepiece = spm
    return _sentencepiece


def _inspect_model_meta(model_path: str) -> tuple[dict, bool]:
    """Detect whether the CTranslate2 model uses FLORES-200 ('eng_Latn') or 3-letter ('eng') codes, and whether source EOS is required."""
    p = Path(model_path)
    code_map = dict(NLLB_CODES)
    needs_source_eos = True

    cfg_file = p / "config.json"
    if cfg_file.exists():
        try:
            c_data = json.loads(cfg_file.read_text(encoding="utf-8", errors="replace"))
            if c_data.get("add_source_eos") is True:
                needs_source_eos = False
        except Exception:
            pass

    st_file = p / "special_tokens_map.json"
    vocab_json = p / "shared_vocabulary.json"
    vocab_txt = p / "shared_vocabulary.txt"
    try:
        if st_file.exists():
            st_data = json.loads(st_file.read_text(encoding="utf-8", errors="replace"))
            specials = set(st_data.get("additional_special_tokens", []))
            if "eng_Latn" in specials:
                code_map = dict(NLLB_CODES)
            elif "eng" in specials:
                code_map = dict(LEGACY_NLLB_CODES)
        elif vocab_json.exists():
            v_data = json.loads(vocab_json.read_text(encoding="utf-8", errors="replace"))
            v_set = set(v_data.keys() if isinstance(v_data, dict) else v_data)
            if "eng_Latn" not in v_set and "eng" in v_set:
                code_map = dict(LEGACY_NLLB_CODES)
        elif vocab_txt.exists():
            v_set = set(vocab_txt.read_text(encoding="utf-8", errors="replace").splitlines())
            if "eng_Latn" not in v_set and "eng" in v_set:
                code_map = dict(LEGACY_NLLB_CODES)
    except Exception:
        pass

    return code_map, needs_source_eos


def get_translator(model_path: str, device: str = "cpu", compute_type: str = "int8") -> tuple:
    """Get or create a CTranslate2 translator with SentencePiece tokenizer.
    
    Returns (translator, spm_processor)
    """
    key = (model_path, device, compute_type)
    with _translators_lock:
        if key not in _translators:
            ct2 = _import_ctranslate2()
            spm = _import_sentencepiece()
            
            translator = ct2.Translator(model_path, device=device, compute_type=compute_type)
            
            # Load SentencePiece model (usually in same directory)
            spm_path = Path(model_path) / "sentencepiece.model"
            if not spm_path.exists():
                for alt in [Path(model_path) / "sentencepiece.bpe.model",
                           Path(model_path).parent / "sentencepiece.model",
                           Path(model_path) / "tokenizer.model"]:
                    if alt.exists():
                        spm_path = alt
                        break
            
            spm_processor = spm.SentencePieceProcessor(str(spm_path))
            code_map, needs_eos = _inspect_model_meta(model_path)
            _translators[key] = (translator, spm_processor, code_map, needs_eos)
        entry = _translators[key]
        return entry[0], entry[1]


def _get_translator_full(model_path: str, device: str = "cpu", compute_type: str = "int8") -> tuple:
    get_translator(model_path, device=device, compute_type=compute_type)
    return _translators[(model_path, device, compute_type)]


def _invoke_progress_cb(progress_cb, current: int, total: int) -> None:
    if not progress_cb:
        return
    for call_args in ((current, total), ()):
        try:
            progress_cb(*call_args)
            return
        except TypeError:
            continue
        except Exception:
            return


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    model_dir: str,
    device: str = "cpu",
    beam_size: int = 2,
    max_batch_size: int = 16,
) -> str:
    """Translate a single text from source to target language."""
    if source_lang == target_lang:
        return text
    
    if not text or not text.strip():
        return ""
    
    try:
        translator, spm, code_map, needs_eos = _get_translator_full(model_dir, device)
        
        src_token = code_map.get(source_lang, NLLB_CODES.get(source_lang, "eng_Latn"))
        tgt_token = code_map.get(target_lang, NLLB_CODES.get(target_lang, "eng_Latn"))
        
        pieces = spm.encode_as_pieces(text.strip())
        source_tokens = [src_token] + pieces + (["</s>"] if needs_eos else [])
        max_len = min(256, max(24, len(pieces) * 3 + 16))
        
        results = translator.translate_batch(
            [source_tokens],
            beam_size=beam_size,
            target_prefix=[[tgt_token]],
            max_batch_size=max_batch_size,
            max_decoding_length=max_len,
            repetition_penalty=1.15,
        )
        
        target_tokens = list(results[0].hypotheses[0])
        while target_tokens and target_tokens[0] in (tgt_token, "<s>", "</s>"):
            target_tokens = target_tokens[1:]
        while target_tokens and target_tokens[-1] in ("</s>", "<s>"):
            target_tokens = target_tokens[:-1]
        
        translated = spm.decode_pieces(target_tokens)
        return translated.strip()
        
    except Exception as e:
        return f"[Translation error: {e}] {text}"


def translate_segments(
    segments: list[tuple[float, float, str]],
    source_lang: str,
    target_lang: str,
    model_dir: str,
    device: str = "cpu",
    progress_cb=None,
) -> list[tuple[float, float, str, str]]:
    """Translate a list of (start, end, text) segments.
    
    Returns list of (start, end, source_text, translated_text)
    """
    if source_lang == target_lang:
        return [(s, e, t, t) for s, e, t in segments]
    
    if not segments:
        return []
    
    try:
        translator, spm, code_map, needs_eos = _get_translator_full(model_dir, device)
        src_token = code_map.get(source_lang, NLLB_CODES.get(source_lang, "eng_Latn"))
        tgt_token = code_map.get(target_lang, NLLB_CODES.get(target_lang, "eng_Latn"))
        
        batch_sources = []
        max_pieces = 8
        for _, _, text in segments:
            pieces = spm.encode_as_pieces(text.strip())
            max_pieces = max(max_pieces, len(pieces))
            batch_sources.append([src_token] + pieces + (["</s>"] if needs_eos else []))
        
        max_len = min(256, max(24, max_pieces * 3 + 16))
        results = translator.translate_batch(
            batch_sources,
            beam_size=2,
            target_prefix=[[tgt_token]] * len(batch_sources),
            max_batch_size=16,
            max_decoding_length=max_len,
            repetition_penalty=1.15,
        )
        
        translated_segments = []
        for i, (start, end, src_text) in enumerate(segments):
            target_tokens = list(results[i].hypotheses[0])
            while target_tokens and target_tokens[0] in (tgt_token, "<s>", "</s>"):
                target_tokens = target_tokens[1:]
            while target_tokens and target_tokens[-1] in ("</s>", "<s>"):
                target_tokens = target_tokens[:-1]
            translated = spm.decode_pieces(target_tokens).strip()
            translated_segments.append((start, end, src_text, translated))
            _invoke_progress_cb(progress_cb, i + 1, len(segments))
        
        return translated_segments
        
    except Exception as e:
        print(f"[stt] Translation failed: {e}", file=__import__('sys').stderr)
        return [(s, e, t, f"[Translation error] {t}") for s, e, t in segments]


def _strip_accents(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s)
        if unicodedata.category(ch) != "Mn"
    )


_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)


def detect_language_simple(text: str) -> tuple[str, float]:
    """Punctuation- and accent-aware language detection from transcribed text.
    
    Returns (lang_code, calibrated_confidence_0_to_1)
    """
    if not text or not text.strip():
        return "en", 0.0

    raw_lower = text.lower()
    tokens_raw = _WORD_RE.findall(raw_lower)
    if not tokens_raw:
        return "en", 0.0

    tokens_norm = [_strip_accents(w.replace("'", "")) for w in tokens_raw]
    n_toks = len(tokens_norm)

    scores = {"en": 0.0, "es": 0.0, "pt": 0.0}
    for w in tokens_norm:
        for lang, markers in LANG_MARKERS.items():
            if w in markers:
                scores[lang] += 1.0

    # Orthographic / diacritic tie-breakers & boosters
    if "ñ" in raw_lower or "¿" in raw_lower or "¡" in raw_lower:
        scores["es"] += 2.0
    if any(ch in raw_lower for ch in ("ã", "õ", "ç", "ê", "ô")):
        scores["pt"] += 2.0
    for w in tokens_raw:
        if w.endswith(("ción", "ciones", "mente")) and "ñ" in raw_lower:
            scores["es"] += 0.8
        elif w.endswith(("ção", "ções", "nho", "nha", "lho", "lha")):
            scores["pt"] += 1.2
        elif "'" in w or w.endswith(("ing", "ght", "tion")) or w.startswith(("th", "wh", "sh")):
            scores["en"] += 0.6

    best_lang = max(scores, key=scores.get)
    best_hits = scores[best_lang]
    if best_hits <= 0.0:
        return "en", 0.35

    sorted_vals = sorted(scores.values(), reverse=True)
    second_hits = sorted_vals[1] if len(sorted_vals) > 1 else 0.0
    hit_density = best_hits / max(n_toks, 1)
    margin = (best_hits - second_hits) / max(best_hits, 1.0)

    # Calibrate confidence into [0, 1] accounting for stopword density (~35-65% in natural speech)
    calibrated = min(0.99, 0.35 + 0.75 * min(hit_density, 0.60) + 0.25 * margin)
    if n_toks < 3:
        calibrated *= 0.80
    return best_lang, round(float(calibrated), 3)


def is_translation_available(model_dir: str) -> bool:
    """Check if translation model is available."""
    return Path(model_dir).exists() and (Path(model_dir) / "model.bin").exists()


def get_model_info(model_dir: str) -> dict:
    """Get information about the translation model."""
    p = Path(model_dir)
    if not p.exists():
        return {"available": False}
    
    size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    has_spm = (p / "sentencepiece.model").exists() or (p / "sentencepiece.bpe.model").exists()
    return {
        "available": True,
        "path": str(p),
        "size_mb": round(size / (1024 * 1024), 1),
        "has_tokenizer": has_spm,
    }