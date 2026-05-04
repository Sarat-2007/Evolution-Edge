"""
Evolution Edge – Real AMD Cloud Agent (FastAPI Server)
=======================================================
DEPLOY THIS ON: AMD Developer Cloud (AMD Instinct MI300X + ROCm 6.x)

How to deploy:
  1. SSH into your AMD Developer Cloud instance
  2. pip install -r requirements_cloud.txt
  3. pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
  4. python cloud_agent.py

Then in config.py on your local machine:
  USE_MOCK_CLOUD = False
  CLOUD_ENDPOINT = "http://<your-amd-cloud-ip>:8080"

What this does:
  - Receives anonymised queries from the edge device
  - Runs Llama-3-8B-Instruct on MI300X for high-quality answers
  - Generates 8 synthetic teaching examples (knowledge distillation)
  - Optionally LoRA fine-tunes TinyLlama and exports as ONNX packet
  - Returns packet to the edge device

Author: Evolution Edge | AMD Hackathon 2026
"""

import os, json, time, uuid, logging, traceback
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("cloud_agent")

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ── PyTorch + ROCm ────────────────────────────────────────────────────────────
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# ── Config ────────────────────────────────────────────────────────────────────
TEACHER_MODEL  = os.getenv("TEACHER_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
STUDENT_MODEL  = os.getenv("STUDENT_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
CLOUD_API_KEY  = os.getenv("CLOUD_API_KEY",  "change-me-in-production")
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 512
ONNX_OUT_DIR   = "/tmp/onnx_packet"

log.info(f"Device: {DEVICE}")

# Singleton teacher pipeline
_teacher_pipe = None


def _load_teacher():
    global _teacher_pipe
    if _teacher_pipe is not None:
        return
    log.info(f"Loading {TEACHER_MODEL} on {DEVICE} ...")
    bnb = BitsAndBytesConfig(load_in_4bit=True) if DEVICE == "cuda" else None
    tok = AutoTokenizer.from_pretrained(TEACHER_MODEL, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        quantization_config=bnb,
        device_map="auto" if DEVICE == "cuda" else None,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    _teacher_pipe = pipeline("text-generation", model=mdl, tokenizer=tok,
                             device_map="auto" if DEVICE == "cuda" else None)
    log.info("Teacher model ready on AMD MI300X")


app = FastAPI(
    title="Evolution Edge Cloud Agent",
    description="AMD Instinct MI300X knowledge distillation endpoint",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query:           str
    current_version: str  = "1.0.0"
    return_onnx:     bool = False


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": TEACHER_MODEL,
            "timestamp": datetime.now().isoformat()}


@app.post("/distill")
def distill(req: QueryRequest, x_api_key: str = Header(None)):
    if x_api_key != CLOUD_API_KEY:
        raise HTTPException(401, "Invalid API key")
    if not req.query.strip():
        raise HTTPException(400, "Empty query")

    log.info(f"Distilling: '{req.query[:80]}'")
    try:
        _load_teacher()
        t0 = time.time()

        # 1) High-quality answer from teacher
        answer = _generate_answer(req.query)

        # 2) Synthetic teaching examples
        examples = _generate_teaching_examples(req.query, answer)

        # 3) Optional ONNX packet
        onnx_bytes = _distill_to_onnx(examples) if req.return_onnx else None

        packet_ver = _bump_version(req.current_version)
        topics = list({req.query.split()[0].lower()} |
                      {ex["q"].split()[0].lower() for ex in examples[:3]})

        packet = {
            "answer":    answer,
            "examples":  examples,
            "topics":    topics,
            "onnx_bytes": list(onnx_bytes) if onnx_bytes else None,
            "metadata": {
                "packet_version": packet_ver,
                "base_version":   req.current_version,
                "created":        datetime.now().isoformat(),
                "size_mb":        round(len(onnx_bytes) / 1024 / 1024 if onnx_bytes else 0.5, 2),
                "num_examples":   len(examples),
                "topics":         topics,
                "provider":       f"AMD Instinct MI300X ROCm6 | {TEACHER_MODEL}",
                "distillation":   "few-shot + LoRA" if req.return_onnx else "few-shot injection",
                "elapsed_sec":    round(time.time() - t0, 2),
            },
        }
        log.info(f"Packet ready v{packet_ver} in {time.time()-t0:.1f}s")
        return JSONResponse(content=packet)

    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(500, str(e))


def _generate_answer(query: str) -> str:
    system = "You are a precise expert AI assistant. Answer in 3-5 sentences with key details."
    prompt = f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{query} [/INST]"
    out = _teacher_pipe(prompt, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True, temperature=0.7, top_p=0.9)
    text = out[0]["generated_text"]
    # Strip prompt
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1]
    return text.strip()


def _generate_teaching_examples(query: str, answer: str, n: int = 8) -> list:
    """Generate n synthetic Q/A pairs from the teacher model."""
    prompt = (
        f"Given this question: '{query}'\n"
        f"And this answer: '{answer[:300]}'\n\n"
        f"Generate {n} related Q&A pairs in JSON array format:\n"
        f'[{{"q": "...", "a": "..."}}, ...]'
    )
    out = _teacher_pipe(prompt, max_new_tokens=600, do_sample=True,
                        temperature=0.8, top_p=0.9)
    text = out[0]["generated_text"].split(prompt)[-1].strip()

    # Try to parse JSON; fallback to manual construction
    try:
        start = text.find("[")
        end   = text.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass

    # Fallback: return a minimal example
    return [{"q": query, "a": answer}]


def _distill_to_onnx(examples: list) -> bytes:
    """
    LoRA fine-tune TinyLlama student on teaching examples, then export to ONNX.
    This is the real knowledge distillation step running on MI300X.
    """
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import TrainingArguments, Trainer
    from datasets import Dataset
    import optimum.exporters.onnx as onnx_exp

    log.info(f"Fine-tuning student model with {len(examples)} examples...")

    tok = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # LoRA config
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    mdl = get_peft_model(mdl, lora_cfg)

    # Dataset
    texts = [f"Q: {ex['q']}\nA: {ex['a']}" for ex in examples]
    ds = Dataset.from_dict({"text": texts})
    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=256, padding="max_length")
    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("input_ids", "labels")

    # Quick 1-epoch fine-tune
    args = TrainingArguments(
        output_dir=ONNX_OUT_DIR, num_train_epochs=1,
        per_device_train_batch_size=2, fp16=True,
        logging_steps=5, save_steps=999, report_to="none",
    )
    trainer = Trainer(model=mdl, args=args, train_dataset=ds)
    trainer.train()

    # Merge LoRA and export to ONNX
    merged = mdl.merge_and_unload()
    os.makedirs(ONNX_OUT_DIR, exist_ok=True)
    from optimum.exporters.onnx import main_export
    main_export(merged, output=ONNX_OUT_DIR, task="causal-lm", opset=14)

    onnx_path = os.path.join(ONNX_OUT_DIR, "model.onnx")
    with open(onnx_path, "rb") as f:
        return f.read()


def _bump_version(version: str) -> str:
    parts = version.split(".")
    parts[2] = str(int(parts[2]) + 1)
    return ".".join(parts)


if __name__ == "__main__":
    log.info("Starting Evolution Edge Cloud Agent on AMD MI300X...")
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)
