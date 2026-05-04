# Evolution Edge – lablab.ai Submission

## Project Abstract (copy to lablab.ai submission form)

**Project Name:** Evolution Edge – Self-Evolving Neural Bridge

**One-Line Pitch:**
Evolution Edge: A tiny AI on your laptop (CPU/NPU) learns from AMD Instinct cloud and grows smarter over time — privacy first, cloud power when needed.

**Full Description:**

Instead of choosing between the power of the cloud and the privacy of the edge, we built a Self-Evolving Neural Bridge powered by AMD's complete AI stack.

A tiny ONNX model (distilgpt2 / TinyLlama quantised) lives on the device, running on CPU with full ONNX Runtime integration (CPUExecutionProvider; DirectML for Ryzen AI NPU when available). It handles routine queries locally with maximum privacy and sub-50ms latency.

A Symbolic Confidence Router — the neuro-symbolic bridge — monitors model confidence after every inference. When confidence drops below 0.65 OR escalation keywords are detected, only a minimal anonymised query is escalated to the cloud.

On AMD Instinct MI300X (ROCm 6.x), a full Llama-3-8B-Instruct teacher model generates a high-quality answer plus 8 synthetic teaching examples via knowledge distillation. These are packaged into a compact Knowledge Packet and streamed back to the device.

The local model integrates this packet (few-shot knowledge injection + optional LoRA weight update exported to ONNX) and permanently improves. The next time the same topic is asked, it answers locally with high confidence — no cloud call needed.

The result: **a model that evolves with you**. Privacy by default. Cloud power when needed. True lifelong learning on AMD silicon.

---

## Team Information

- **Team Name:** Evolution Edge
- **Member:** Venkata Durga Sai Sarat Chandra
- **Location:** Srikakulam, Andhra Pradesh, India
- **Hackathon:** AMD Developer Hackathon 2026 (lablab.ai, May 4–10, 2026)

---

## AMD Technologies Used

- **AMD Instinct MI300X** — cloud teacher model inference + LoRA distillation
- **ROCm 6.x** — PyTorch GPU acceleration on MI300X
- **AMD Ryzen AI NPU** — on-device NPU acceleration via ONNX Runtime DirectML EP
- **ONNX Runtime** — cross-platform local inference (CPU / NPU)
- **HuggingFace Optimum** — ONNX export and ORT model wrapper
- **PEFT / LoRA** — lightweight fine-tuning for student model on MI300X

---

## Key Innovation

Most edge AI systems are static — they never improve after deployment.
Evolution Edge introduces **persistent on-device learning** without compromising privacy:

1. **Privacy-first**: only the question text is ever sent to the cloud (no user data, no context)
2. **Self-evolving**: each cloud call permanently improves the local model
3. **Progressive autonomy**: over time, the local model handles more queries without cloud
4. **AMD-native**: designed around the AMD AI stack (MI300X + Ryzen AI + ROCm + ONNX)

---

## Demo Video Script (60 seconds)

See `demo_script.md` for the full script and recording instructions.

---

## Repository Structure

```
AMD hackthon/           ← root
├── app.py              ← run: python app.py
├── setup_local.py      ← run once: python setup_local.py
├── cloud_agent.py      ← deploy on AMD MI300X
├── cloud_agent_mock.py ← CPU simulation (default)
├── config.py           ← all settings
├── local_model/        ← ONNX inference + router + manager
├── knowledge_packet/   ← packet builder + receiver
├── requirements_local.txt
├── requirements_cloud.txt
└── README.md
```

---

## How to Run

```bash
pip install -r requirements_local.txt
python setup_local.py   # one-time
python app.py           # → http://localhost:7860
```
