"""
Evolution Edge – Local ONNX Inference Engine
=============================================
Runs the quantised student model locally on CPU via ONNX Runtime.
No GPU or NPU required (AMD Ryzen AI NPU auto-detected if available).

Architecture note:
  - Primary:  ORTModelForCausalLM  (ONNX Runtime via HF Optimum)
  - Fallback: AutoModelForCausalLM (pure transformers, CPU mode)

Confidence scoring:
  Uses a multi-signal heuristic combining:
    1. Token-probability entropy (from generation scores)
    2. Answer coherence signals (repetition, length, hedging phrases)
  → Confidence ∈ [0, 1].  Below 0.65 → router escalates to cloud.
"""

import os, re, json, logging, math
from typing import Optional
import numpy as np

log = logging.getLogger(__name__)

# ── Lazy singletons ──────────────────────────────────────────────────────────
_model     = None
_tokenizer = None
_backend   = "none"   # "ort" | "transformers"


def _load_model():
    """Load the local model once; pick best available backend."""
    global _model, _tokenizer, _backend

    if _model is not None:
        return

    from config import ONNX_MODEL_DIR, TOKENIZER_DIR, LOCAL_MODEL_NAME, ONNX_PROVIDER

    # ── Try ONNX Runtime via Optimum ─────────────────────────────────────────
    if os.path.exists(os.path.join(ONNX_MODEL_DIR, "model.onnx")):
        try:
            from optimum.onnxruntime import ORTModelForCausalLM
            from transformers import AutoTokenizer
            log.info("Loading ONNX model via Optimum (CPUExecutionProvider)…")
            _model = ORTModelForCausalLM.from_pretrained(
                ONNX_MODEL_DIR, provider=ONNX_PROVIDER
            )
            _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            _backend = "ort"
            log.info("✅ ONNX Runtime backend ready")
            return
        except Exception as e:
            log.warning(f"ORT load failed ({e}), falling back to transformers.")

    # ── Fallback: vanilla transformers (CPU) ─────────────────────────────────
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        log.info(f"Loading {LOCAL_MODEL_NAME} via transformers (CPU)…")
        _tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model     = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME)
        _model.eval()
        _backend   = "transformers"
        log.info("✅ Transformers (CPU) backend ready")
    except Exception as e:
        log.error(f"Model load failed: {e}")
        raise RuntimeError("Could not load local model. Run setup_local.py first.") from e


def reload_model():
    """Hot-reload the model (called after a knowledge packet is applied)."""
    global _model, _tokenizer, _backend
    _model, _tokenizer, _backend = None, None, "none"
    _load_model()
    log.info("✅ Model reloaded after packet application")


# ── Knowledge Base helpers ────────────────────────────────────────────────────

def _load_knowledge_base() -> dict:
    from config import KNOWLEDGE_BASE_PATH
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        return {"examples": []}
    with open(KNOWLEDGE_BASE_PATH) as f:
        return json.load(f)


def _build_few_shot_context(query: str, kb: dict, max_examples: int = 4) -> str:
    """
    Find the most relevant few-shot examples from the knowledge base
    and prepend them to the query using ModelManager's Jaccard logic.
    """
    # Import locally to avoid circular dependencies
    from local_model.model_manager import ModelManager
    manager = ModelManager()
    
    # Retrieve highest similarity packets matching the query
    top = manager.get_relevant_examples(query, top_k=max_examples)

    if not top:
        return ""

    lines = []
    for ex in top:
        lines.append(f"<|user|>\n{ex['q']}\n<|assistant|>\n{ex['a']}")
    return "\n\n".join(lines) + "\n\n"


def _build_prompt(query: str, few_shot_ctx: str) -> str:
    if few_shot_ctx:
        return f"<|system|>\nYou are Evolution Edge, a helpful, precise AI assistant.</s>\n{few_shot_ctx}<|user|>\n{query}</s>\n<|assistant|>\n"
    return f"<|system|>\nYou are Evolution Edge, a helpful, precise AI assistant.</s>\n<|user|>\n{query}</s>\n<|assistant|>\n"


# ── Confidence Scoring ────────────────────────────────────────────────────────

def compute_confidence(generated_text: str, query: str,
                        scores: Optional[list] = None) -> float:
    """
    Multi-signal confidence score in [0.0, 1.0].

    Signals:
      1. Score-entropy  – from generation token probabilities (if available)
      2. Length bonus   – very short answers are less reliable
      3. Repetition penalty – GPT loops when confused
      4. Hedging penalty    – "I'm not sure", "maybe", "I think"
    """
    import torch, torch.nn.functional as F

    signals = []

    # ── Signal 1: token probability entropy ──────────────────────────────────
    if scores:
        entropies = []
        for score_tensor in scores:
            probs   = F.softmax(score_tensor.float(), dim=-1).squeeze(0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            vocab   = score_tensor.shape[-1]
            max_ent = math.log(vocab)
            entropies.append(entropy / max_ent)
        avg_entropy    = float(np.mean(entropies))
        conf_entropy   = 1.0 - avg_entropy
        signals.append(("entropy", conf_entropy, 0.55))   # weight 55%
    
    # ── Signal 2: length bonus ────────────────────────────────────────────────
    word_count  = len(generated_text.split())
    len_score   = min(1.0, word_count / 40.0)             # saturates at 40 words
    signals.append(("length", len_score, 0.15))

    # ── Signal 3: repetition penalty ─────────────────────────────────────────
    words         = generated_text.lower().split()
    unique_ratio  = len(set(words)) / max(len(words), 1)
    signals.append(("unique", unique_ratio, 0.15))

    # ── Signal 4: hedging-phrase penalty ─────────────────────────────────────
    hedges = ["i'm not sure", "i don't know", "unclear", "uncertain",
              "might be", "i think", "perhaps", "possibly", "not certain"]
    hedge_count = sum(1 for h in hedges if h in generated_text.lower())
    hedge_score = max(0.0, 1.0 - hedge_count * 0.25)
    signals.append(("hedge", hedge_score, 0.15))

    # ── Weighted average ──────────────────────────────────────────────────────
    total_w  = sum(w for _, _, w in signals)
    conf     = sum(s * w for _, s, w in signals) / total_w

    log.debug(f"Confidence signals: {[(n, round(s,3)) for n,s,_ in signals]} → {conf:.3f}")
    return float(np.clip(conf, 0.0, 1.0))


# ── Main inference entry point ────────────────────────────────────────────────

def local_inference(query: str) -> tuple[str, float, float]:
    """
    Generate an answer for `query` using the local ONNX model on CPU.

    Returns:
        (answer_text: str, confidence: float, elapsed_ms: float)
    """
    import torch
    import time

    _load_model()

    kb       = _load_knowledge_base()
    few_shot = _build_few_shot_context(query, kb)
    prompt   = _build_prompt(query, few_shot)

    log.info(f"[Local CPU] Inference | few-shot examples: {len(kb.get('examples', []))}")

    from config import MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY

    inputs    = _tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=512)
    input_len = inputs["input_ids"].shape[1]

    start_time = time.time()
    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens       = MAX_NEW_TOKENS,
            do_sample            = True,
            temperature          = TEMPERATURE,
            top_p                = TOP_P,
            repetition_penalty   = REPETITION_PENALTY,
            pad_token_id         = _tokenizer.eos_token_id,
            output_scores        = True,
            return_dict_in_generate = True,
        )
    elapsed_ms = (time.time() - start_time) * 1000.0

    # Decode only the newly generated tokens
    gen_ids      = output.sequences[0, input_len:]
    answer_text  = _tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Clean up the answer
    answer_text  = _clean_answer(answer_text, query)

    # Compute confidence
    scores = list(output.scores) if hasattr(output, "scores") and output.scores else None
    conf   = compute_confidence(answer_text, query, scores)

    log.info(f"[Local CPU] Answer length: {len(answer_text.split())} words | Confidence: {conf:.3f} | Time: {elapsed_ms:.1f}ms")
    return answer_text, conf, elapsed_ms


def _clean_answer(text: str, query: str) -> str:
    """Remove prompt echoes, truncate at sentence boundary, etc."""
    # Remove leading tags
    text = re.sub(r"^(<\|assistant\|>|<\|user\|>|Q:|A:|Question:|Answer:)\s*", "", text, flags=re.IGNORECASE)
    # Truncate after the first trailing token (model started asking another Q)
    cut = re.search(r"\n\s*(<\|user\|>|Q:)", text)
    if cut:
        text = text[:cut.start()]
    # Ensure it ends at a sentence boundary
    last_period = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_period > len(text) // 2:
        text = text[:last_period + 1]
    return text.strip() or "I need more context to answer that accurately."
