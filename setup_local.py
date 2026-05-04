"""
Evolution Edge – Local Model Setup
===================================
Run ONCE before launching app.py.

What this does:
  1. Downloads distilgpt2 from Hugging Face (~350 MB)
  2. Exports it to ONNX format via Optimum (CPUExecutionProvider)
  3. Saves tokenizer alongside the ONNX model
  4. Initialises an empty knowledge base (grows as model evolves)
  5. Initialises the model version tracker

Usage:
  python setup_local.py

Hardware:
  Any CPU — no GPU, no NPU required.
  (AMD Ryzen AI NPU will be auto-detected and used if available via DirectML)
"""

import os, sys, json, shutil, logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format=" %(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("setup")


def banner():
    print("""
+======================================================+
|         EVOLUTION EDGE -- LOCAL SETUP                |
|   Self-Evolving Neural Bridge | AMD Hackathon 2026   |
+======================================================+
""")


def check_python():
    major, minor = sys.version_info[:2]
    if major < 3 or minor < 10:
        log.error(f"Python 3.10+ required. Found {major}.{minor}")
        sys.exit(1)
    log.info(f"✅ Python {major}.{minor}")


def check_packages():
    missing = []
    for pkg, import_name in [
        ("transformers", "transformers"),
        ("optimum",      "optimum"),
        ("onnxruntime",  "onnxruntime"),
        ("torch",        "torch"),
        ("gradio",       "gradio"),
    ]:
        try:
            __import__(import_name)
            log.info(f"✅ {pkg} installed")
        except ImportError:
            log.warning(f"❌ {pkg} NOT found")
            missing.append(pkg)
    if missing:
        log.error("Run:  pip install -r requirements_local.txt")
        sys.exit(1)


def export_onnx_model():
    """Download distilgpt2 and export to ONNX via HF Optimum."""
    from config import LOCAL_MODEL_NAME, ONNX_MODEL_DIR, TOKENIZER_DIR, ONNX_PROVIDER

    if os.path.exists(os.path.join(ONNX_MODEL_DIR, "model.onnx")):
        log.info("✅ ONNX model already exists — skipping export")
        return

    log.info(f"📥 Downloading {LOCAL_MODEL_NAME} (may take a few minutes)…")

    try:
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer

        log.info("🔄 Exporting to ONNX (CPUExecutionProvider)…")
        model = ORTModelForCausalLM.from_pretrained(
            LOCAL_MODEL_NAME,
            export=True,
            provider=ONNX_PROVIDER,
        )
        model.save_pretrained(ONNX_MODEL_DIR)
        log.info(f"✅ ONNX model saved → {ONNX_MODEL_DIR}")

        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(TOKENIZER_DIR)
        log.info(f"✅ Tokenizer saved → {TOKENIZER_DIR}")

    except Exception as e:
        log.warning(f"Optimum ONNX export failed ({e}). Falling back to raw transformers.")
        _fallback_download(LOCAL_MODEL_NAME, TOKENIZER_DIR)


def _fallback_download(model_name: str, tokenizer_dir: str):
    """Fallback: just save tokenizer; inference.py will use transformers directly."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    log.info("📥 Using fallback: downloading via transformers…")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(tokenizer_dir)
    # Cache the model weights for offline use
    AutoModelForCausalLM.from_pretrained(model_name)
    log.info("✅ Fallback download complete. Inference will use transformers backend.")


def init_knowledge_base():
    from config import KNOWLEDGE_BASE_PATH
    if os.path.exists(KNOWLEDGE_BASE_PATH):
        log.info("✅ Knowledge base already exists")
        return
    kb = {
        "version":  "1.0.0",
        "created":  datetime.now().isoformat(),
        "examples": [],
        "topics":   [],
        "stats": {
            "packets_applied": 0,
            "total_queries":   0,
            "cloud_queries":   0,
        },
    }
    with open(KNOWLEDGE_BASE_PATH, "w") as f:
        json.dump(kb, f, indent=2)
    log.info(f"✅ Knowledge base initialised → {KNOWLEDGE_BASE_PATH}")


def init_version_tracker():
    from config import MODEL_VERSION_FILE, LOCAL_MODEL_NAME
    if os.path.exists(MODEL_VERSION_FILE):
        log.info("✅ Version tracker already exists")
        return
    info = {
        "version":         "1.0.0",
        "base_model":      LOCAL_MODEL_NAME,
        "onnx_exported":   True,
        "packets_applied": 0,
        "improvements":    [],
        "created":         datetime.now().isoformat(),
        "last_updated":    datetime.now().isoformat(),
    }
    with open(MODEL_VERSION_FILE, "w") as f:
        json.dump(info, f, indent=2)
    log.info(f"✅ Version tracker initialised → {MODEL_VERSION_FILE}")


def main():
    banner()
    check_python()
    check_packages()
    init_knowledge_base()
    init_version_tracker()
    export_onnx_model()

    print("""
+======================================================+
|  [OK] SETUP COMPLETE!                               |
|                                                     |
|  Next step:  python app.py                          |
|  Then open:  http://localhost:7862                  |
+======================================================+
""")


if __name__ == "__main__":
    main()
