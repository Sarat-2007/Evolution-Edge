"""
Evolution Edge – Central Configuration
=======================================
Self-Evolving Neural Bridge | AMD Hackathon 2026
Team: Evolution Edge | Member: Venkata Durga Sai Sarat Chandra

HARDWARE NOTE:
    Local side:  Lenovo LOQ 15ARP9 — runs on CPU (no NPU required)
    Cloud side:  AMD Instinct MI300X — ROCm 6.x (swap USE_MOCK_CLOUD below)
"""

import os

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR              = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR            = os.path.join(BASE_DIR, "models")
ONNX_MODEL_DIR        = os.path.join(MODELS_DIR, "onnx")
TOKENIZER_DIR         = os.path.join(MODELS_DIR, "tokenizer")
KNOWLEDGE_PACKETS_DIR = os.path.join(BASE_DIR, "knowledge_packets")
EVOLUTION_LOG_DIR     = os.path.join(BASE_DIR, "evolution_log")

# ─── Model Files ─────────────────────────────────────────────────────────────
KNOWLEDGE_BASE_PATH   = os.path.join(MODELS_DIR, "knowledge_base.json")
MODEL_VERSION_FILE    = os.path.join(MODELS_DIR, "model_version.json")

# ─── Local Model (CPU / ONNX Runtime) ─────────────────────────────────────────
# CPU fallback — no GPU or NPU required on local device
LOCAL_MODEL_NAME      = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ONNX_PROVIDER        = "CPUExecutionProvider"  # DirectML or DmlExecutionProvider for Windows GPU

# ─── Cloud Teacher Model ──────────────────────────────────────────────────────
# Mock (CPU demo): gpt2-medium
# Real AMD cloud: meta-llama/Meta-Llama-3-8B-Instruct on MI300X
CLOUD_TEACHER_MODEL   = "gpt2-medium"  # swap to Llama-3-8B on real AMD cloud

# ─── Confidence Router ───────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.65   # below → escalate to cloud
ESCALATION_KEYWORDS   = [
    "explain", "why", "how does", "how do", "what is", "what are",
    "difference between", "compare", "elaborate", "describe",
    "mechanism", "theory", "concept", "advanced", "deep dive",
    "in detail", "pros and cons", "advantages", "disadvantages",
]

# ─── Generation Parameters ───────────────────────────────────────────────────
MAX_NEW_TOKENS      = 350
TEMPERATURE         = 0.75
TOP_P               = 0.92
REPETITION_PENALTY  = 1.15

# ─── Cloud Configuration ─────────────────────────────────────────────────────
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  TO SWITCH TO REAL AMD INSTINCT MI300X CLOUD:                           │
# │  1. Set USE_MOCK_CLOUD = False                                           │
# │  2. Set CLOUD_ENDPOINT to your AMD Developer Cloud URL                  │
# │  3. Set CLOUD_API_KEY  to your AMD cloud API key                        │
# │  That's it — no other code changes needed.                              │
# └─────────────────────────────────────────────────────────────────────────┘
USE_MOCK_CLOUD  = True
CLOUD_ENDPOINT  = os.getenv("CLOUD_ENDPOINT", "http://your-amd-mi300x-endpoint:8080")
CLOUD_API_KEY   = os.getenv("CLOUD_API_KEY", "")  # Set via environment variable for security
CLOUD_TIMEOUT   = 60   # seconds

# ─── Evolution / Knowledge Packet Settings ───────────────────────────────────
MAX_TEACHING_EXAMPLES = 8
MOCK_PACKET_DELAY_SEC = 3.0    # simulates network + cloud compute latency
MIN_PACKET_SIZE_MB    = 4.0
MAX_PACKET_SIZE_MB    = 14.0

# ─── Ensure Dirs Exist ───────────────────────────────────────────────────────
for _dir in [MODELS_DIR, ONNX_MODEL_DIR, TOKENIZER_DIR,
             KNOWLEDGE_PACKETS_DIR, EVOLUTION_LOG_DIR]:
    os.makedirs(_dir, exist_ok=True)
