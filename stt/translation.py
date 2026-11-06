"""Translation module using CTranslate2 with NLLB-200 for offline multilingual translation."""
import os
import threading
from pathlib import Path
from typing import Optional

# These will be imported lazily to avoid hard dependency
_ctranslate2 = None
_sentencepiece = None
_translators: dict = {}
_translators_lock = threading.Lock()

# NLLB language codes (model-specific format)
# The downloaded NLLB-200 distilled 600M CTranslate2 model uses 3-letter codes: eng, spa, por
NLLB_CODES = {
    "en": "eng",
    "es": "spa",
    "pt": "por",
}

# Language detection markers (high-frequency function words)
# These are designed to be distinctive between languages
LANG_MARKERS = {
    "en": {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know",
        "take", "people", "into", "year", "your", "good", "some", "could", "them",
        "see", "other", "than", "then", "now", "look", "only", "come", "its", "over",
        "think", "also", "back", "after", "use", "two", "how", "our", "work", "first",
        "well", "way", "even", "new", "want", "because", "any", "these", "give", "day",
        "most", "us", "is", "was", "are", "were", "been", "has", "had", "does",
        "did", "doing", "going", "coming", "being", "said", "says",
        "am", "is", "are", "was", "were", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "should", "could", "may", "might", "must",
        "shall", "can", "need", "dare", "ought", "used",
    },
    "es": {
        "el", "la", "los", "las", "de", "que", "y", "a", "en", "un",
        "una", "unos", "unas", "ser", "estar", "haber", "tener", "hacer", "poder", "decir",
        "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "aquel", "aquella",
        "mi", "tu", "su", "mis", "tus", "sus", "nuestro", "nuestra", "vuestro", "vuestra",
        "me", "te", "se", "nos", "os", "le", "les", "lo", "la", "los", "las",
        "con", "por", "para", "sin", "sobre", "entre", "hasta", "desde", "durante", "mediante",
        "como", "cuando", "donde", "quien", "cual", "cuyo", "cuanto", "que", "quien",
        "no", "ni", "pero", "sino", "o", "u", "y", "e", "ni", "tambien", "tampoco",
        "muy", "mas", "menos", "tan", "tanto", "asi", "asi", "aqui", "alli", "alla",
        "ahora", "antes", "despues", "entonces", "luego", "siempre", "nunca", "a veces",
        "algo", "alguien", "nadie", "todo", "nada", "algun", "ningun", "cada", "cualquier",
        "uno", "dos", "tres", "primero", "ultimo", "otro", "mismo", "diferente", "igual",
        "grande", "pequeno", "bueno", "malo", "nuevo", "viejo", "alto", "bajo", "largo", "corto",
    },
    "pt": {
        "o", "a", "os", "as", "de", "que", "e", "do", "da", "em",
        "um", "uma", "uns", "umas", "ser", "estar", "ter", "haver", "fazer", "poder",
        "este", "esta", "estes", "estas", "esse", "essa", "esses", "essas", "aquele", "aquela",
        "meu", "minha", "teu", "tua", "seu", "sua", "nosso", "nossa", "vosso", "vossa",
        "me", "te", "se", "nos", "vos", "lhe", "lhes", "o", "a", "os", "as", "lho", "lha",
        "com", "por", "para", "sem", "sobre", "entre", "ate", "desde", "durante", "mediante",
        "como", "quando", "onde", "quem", "qual", "cujo", "quanto", "que", "quem",
        "nao", "nem", "mas", "porem", "entretanto", "ou", "ora", "nem", "tambem", "tampouco",
        "muito", "mais", "menos", "tam", "tanto", "assim", "aqui", "ai", "la", "la",
        "agora", "antes", "depois", "entao", "logo", "sempre", "nunca", "as vezes",
        "algo", "alguem", "ninguem", "tudo", "nada", "algum", "nenhum", "cada", "qualquer",
        "um", "dois", "tres", "primeiro", "ultimo", "outro", "mesmo", "diferente", "igual",
        "grande", "pequeno", "bom", "mau", "novo", "velho", "alto", "baixo", "longo", "curto",
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
                # Try common locations
                for alt in [Path(model_path) / "sentencepiece.bpe.model",
                           Path(model_path).parent / "sentencepiece.model",
                           Path(model_path) / "tokenizer.model"]:
                    if alt.exists():
                        spm_path = alt
                        break
            
            spm_processor = spm.SentencePieceProcessor(str(spm_path))
            
            _translators[key] = (translator, spm_processor)
        return _translators[key]


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    model_dir: str,
    device: str = "cpu",
    beam_size: int = 4,
    max_batch_size: int = 16,
) -> str:
    """Translate a single text from source to target language.
    
    Args:
        text: Source text to translate
        source_lang: Source language code (en, es, pt)
        target_lang: Target language code (en, es, pt)
        model_dir: Path to CTranslate2 NLLB model directory
        device: "cpu" or "cuda"
        beam_size: Beam search size
        max_batch_size: Max batch size for translation
        
    Returns:
        Translated text
    """
    if source_lang == target_lang:
        return text
    
    if not text or not text.strip():
        return ""
    
    try:
        translator, spm = get_translator(model_dir, device)
        
        # NLLB expects language tokens as prefixes
        src_token = NLLB_CODES.get(source_lang, "eng_Latn")
        tgt_token = NLLB_CODES.get(target_lang, "eng_Latn")
        
        # Tokenize with source language prefix
        source_tokens = spm.encode_as_pieces(text)
        source_tokens = [src_token] + source_tokens
        
        # Translate
        results = translator.translate_batch(
            [source_tokens],
            beam_size=beam_size,
            target_prefix=[[tgt_token]],
            max_batch_size=max_batch_size,
        )
        
        # Detokenize (remove target language token)
        target_tokens = results[0].hypotheses[0]
        if target_tokens and target_tokens[0] == tgt_token:
            target_tokens = target_tokens[1:]
        
        translated = spm.decode_pieces(target_tokens)
        return translated.strip()
        
    except Exception as e:
        # Fallback: return original text with error prefix
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
        translator, spm = get_translator(model_dir, device)
        src_token = NLLB_CODES.get(source_lang, "eng_Latn")
        tgt_token = NLLB_CODES.get(target_lang, "eng_Latn")
        
        # Prepare batch
        batch_sources = []
        for _, _, text in segments:
            tokens = spm.encode_as_pieces(text)
            batch_sources.append([src_token] + tokens)
        
        # Translate in batches
        results = translator.translate_batch(
            batch_sources,
            beam_size=4,
            target_prefix=[[tgt_token]] * len(batch_sources),
            max_batch_size=16,
        )
        
        # Process results
        translated_segments = []
        for i, (start, end, src_text) in enumerate(segments):
            target_tokens = results[i].hypotheses[0]
            if target_tokens and target_tokens[0] == tgt_token:
                target_tokens = target_tokens[1:]
            translated = spm.decode_pieces(target_tokens).strip()
            translated_segments.append((start, end, src_text, translated))
            
            if progress_cb:
                progress_cb(i + 1, len(segments))
        
        return translated_segments
        
    except Exception as e:
        # Return original text on error
        print(f"[stt] Translation failed: {e}", file=__import__('sys').stderr)
        return [(s, e, t, f"[Translation error] {t}") for s, e, t in segments]


def detect_language_simple(text: str) -> tuple[str, float]:
    """Simple heuristic language detection from text.
    
    Returns (lang_code, confidence)
    """
    words = set(text.lower().split())
    if not words:
        return "en", 0.5
    
    scores = {}
    for lang, lang_words in LANG_MARKERS.items():
        matches = words & lang_words
        scores[lang] = len(matches) / max(len(words), 1)
    
    if not scores or max(scores.values()) < 0.05:
        return "en", 0.5  # Default to English
    
    best_lang = max(scores, key=scores.get)
    return best_lang, min(scores[best_lang], 1.0)


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