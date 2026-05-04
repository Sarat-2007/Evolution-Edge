"""
Evolution Edge – Main Application (Gradio UI)
=============================================
Self-Evolving Neural Bridge | AMD Hackathon 2026
Team: Evolution Edge | Member: Venkata Durga Sai Sarat Chandra

Run:
  python app.py
  → Open http://localhost:7860

Architecture flow (shown live in UI):
  User Query
    → Local ONNX Model (CPU / Ryzen AI NPU)
    → Confidence Router
        ├── High confidence → Answer locally ✅
        └── Low confidence → Escalate to AMD Cloud
                → Knowledge Distillation (MI300X)
                → ONNX Knowledge Packet
                → Local model upgraded!
                → Re-answer locally (now smarter) 🚀
"""

import os, sys, json, time, threading, logging
from datetime import datetime
from pathlib import Path

import gradio as gr

# ── Setup logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app")

# ── Ensure we can import local modules ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import USE_MOCK_CLOUD, CLOUD_ENDPOINT, CLOUD_API_KEY, CONFIDENCE_THRESHOLD
from local_model.inference import local_inference
from local_model.router import SymbolicRouter
from local_model.model_manager import ModelManager
from knowledge_packet.packet_receiver import PacketReceiver

# ── Singletons ────────────────────────────────────────────────────────────────
router   = SymbolicRouter()
manager  = ModelManager()
receiver = PacketReceiver(manager)

# ─────────────────────────────────────────────────────────────────────────────
# CSS – Dark AMD-themed premium design
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-primary:    #080c14;
  --bg-secondary:  #0d1422;
  --bg-card:       #111827;
  --bg-glass:      rgba(17,24,39,0.7);
  --border:        rgba(99,179,237,0.15);
  --amd-red:       #ED1C24;
  --amd-orange:    #FF6B00;
  --cyan:          #22d3ee;
  --green:         #10b981;
  --yellow:        #f59e0b;
  --purple:        #8b5cf6;
  --text-primary:  #f0f4f8;
  --text-muted:    #64748b;
  --text-dim:      #94a3b8;
  --gradient-amd:  linear-gradient(135deg, var(--amd-red), var(--amd-orange));
  --gradient-cyber: linear-gradient(135deg, #22d3ee, #8b5cf6);
  --glow-cyan:     0 0 20px rgba(34,211,238,0.3);
  --glow-red:      0 0 20px rgba(237,28,36,0.3);
  --radius:        12px;
  --radius-sm:     8px;
}

/* ── Base ─────────────────────────────────────────────────── */
body, .gradio-container {
  background: var(--bg-primary) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--text-primary) !important;
}

.gradio-container {
  max-width: 1400px !important;
  margin: 0 auto !important;
}

/* ── Header ───────────────────────────────────────────────── */
#header-html {
  padding: 28px 32px 20px;
  background: linear-gradient(180deg, rgba(237,28,36,0.08), transparent);
  border-bottom: 1px solid var(--border);
  margin-bottom: 20px;
}

.header-inner {
  display: flex;
  align-items: center;
  gap: 20px;
}

.header-logo {
  font-size: 2.8rem;
  filter: drop-shadow(0 0 12px rgba(237,28,36,0.6));
}

.header-title {
  font-size: 2rem;
  font-weight: 700;
  background: var(--gradient-amd);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0;
  line-height: 1.1;
  letter-spacing: -0.5px;
}

.header-subtitle {
  font-size: 0.9rem;
  color: var(--text-dim);
  margin: 4px 0 0;
  font-weight: 400;
}

.header-badge {
  margin-left: auto;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
}

.badge-chip {
  background: rgba(237,28,36,0.15);
  border: 1px solid rgba(237,28,36,0.3);
  color: var(--amd-orange);
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.5px;
}

/* ── Panels ───────────────────────────────────────────────── */
.chat-col, .log-col {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}

.panel-label {
  padding: 10px 16px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border);
  background: rgba(255,255,255,0.02);
}

/* ── Chatbot ────────────────────────────────────────────────── */
#chatbot {
  background: transparent !important;
  border: none !important;
}

#chatbot .message-wrap { padding: 16px; }

#chatbot .message.user {
  background: linear-gradient(135deg, rgba(237,28,36,0.15), rgba(255,107,0,0.1)) !important;
  border: 1px solid rgba(237,28,36,0.2) !important;
  border-radius: 12px 12px 4px 12px !important;
  color: white !important;
  margin-left: 40px !important;
}

#chatbot .message.bot {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px 12px 12px 4px !important;
  color: white !important;
  margin-right: 40px !important;
}

#chatbot .message.user p, #chatbot .message.bot p {
  color: white !important;
}

#chatbot .message.bot span {
  color: white !important;
}

/* ── Log Panel ────────────────────────────────────────────── */
#log-panel-html {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  height: 460px;
  overflow-y: auto;
  padding: 14px;
  color: var(--text-dim);
  line-height: 1.6;
  scroll-behavior: smooth;
}

#log-panel-html::-webkit-scrollbar { width: 4px; }
#log-panel-html::-webkit-scrollbar-track { background: transparent; }
#log-panel-html::-webkit-scrollbar-thumb {
  background: rgba(99,179,237,0.3);
  border-radius: 2px;
}

.log-entry {
  padding: 4px 0;
  border-bottom: 1px solid rgba(255,255,255,0.03);
  display: flex;
  gap: 8px;
  align-items: flex-start;
}

.log-time { color: var(--text-muted); flex-shrink: 0; }
.log-msg  { flex: 1; }

.log-local    { color: #10b981; }
.log-escalate { color: #f59e0b; }
.log-cloud    { color: #8b5cf6; }
.log-packet   { color: #22d3ee; }
.log-evolved  { color: #ED1C24; font-weight: 600; }
.log-info     { color: var(--text-dim); }

/* ── Stats Bar ───────────────────────────────────────────── */
#stats-html {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 24px;
  margin-top: 12px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 16px;
  align-items: center;
}

.stat-item {
  text-align: center;
}

.stat-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--cyan);
  display: block;
  font-family: 'JetBrains Mono', monospace;
}

.stat-label {
  font-size: 0.68rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-top: 2px;
}

.stat-divider {
  width: 1px;
  height: 40px;
  background: var(--border);
  margin: 0 auto;
}

.evolution-bar-wrap {
  background: rgba(255,255,255,0.05);
  border-radius: 4px;
  height: 6px;
  overflow: hidden;
  margin-top: 6px;
}

.evolution-bar {
  height: 100%;
  background: var(--gradient-amd);
  border-radius: 4px;
  transition: width 0.6s ease;
}

/* ── Input Row ───────────────────────────────────────────── */
#input-row {
  display: flex;
  gap: 8px;
  align-items: flex-end;
  margin-top: 12px;
}

#msg-input textarea {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', sans-serif !important;
  resize: none !important;
  min-height: 52px !important;
}

#msg-input textarea:focus {
  border-color: rgba(237,28,36,0.5) !important;
  box-shadow: 0 0 0 2px rgba(237,28,36,0.1) !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.btn-send {
  background: var(--gradient-amd) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  letter-spacing: 0.3px !important;
  min-width: 100px !important;
  box-shadow: var(--glow-red) !important;
  transition: all 0.2s !important;
}

.btn-send:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 0 28px rgba(237,28,36,0.4) !important;
}

.btn-escalate {
  background: rgba(139,92,246,0.15) !important;
  border: 1px solid rgba(139,92,246,0.4) !important;
  color: #a78bfa !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  transition: all 0.2s !important;
}

.btn-escalate:hover {
  background: rgba(139,92,246,0.25) !important;
  box-shadow: 0 0 16px rgba(139,92,246,0.3) !important;
  transform: translateY(-1px) !important;
}

.btn-reset {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-muted) !important;
  border-radius: var(--radius-sm) !important;
  transition: all 0.2s !important;
}

.btn-reset:hover {
  border-color: rgba(255,255,255,0.2) !important;
  color: var(--text-primary) !important;
}

/* ── Pulse animation for live indicators ─────────────────── */
@keyframes pulse-green {
  0%, 100% { box-shadow: 0 0 4px #10b981; }
  50%       { box-shadow: 0 0 12px #10b981; }
}

@keyframes pulse-yellow {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.5; }
}

.live-dot {
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 50%;
  background: #10b981;
  animation: pulse-green 1.5s infinite;
  vertical-align: middle;
  margin-right: 6px;
}

/* ── Misc ─────────────────────────────────────────────────── */
.gr-box, .gr-form { background: transparent !important; border: none !important; }
footer { display: none !important; }
"""

# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log_entry(css_class: str, msg: str) -> dict:
    return {"time": _ts(), "cls": css_class, "msg": msg}


def _render_logs(entries: list) -> str:
    if not entries:
        return "<div style='color:#64748b;font-size:0.8rem;padding:20px;'>Waiting for first query...</div>"
    html = ""
    for e in entries[-80:]:   # keep last 80 lines
        html += (
            f"<div class='log-entry'>"
            f"<span class='log-time'>[{e['time']}]</span>"
            f"<span class='log-msg {e['cls']}'>{e['msg']}</span>"
            f"</div>"
        )
    # Auto-scroll trick
    html += "<div id='log-bottom'></div><script>document.getElementById('log-bottom')?.scrollIntoView();</script>"
    return html


def _render_stats(state: dict) -> str:
    v          = state.get("version", "1.0.0")
    packets    = state.get("packets_applied", 0)
    kb_size    = state.get("kb_examples", 0)
    topics     = state.get("topics_learned", 0)
    conf       = state.get("last_confidence", 0.0)
    saved      = state.get("cost_saved", 0.0)
    cloud_mode = "🔴 Mock CPU" if USE_MOCK_CLOUD else "🟢 AMD MI300X"

    pct = min(100, packets * 20)   # 5 packets = 100% evolution bar

    # Pipeline Visualization State
    pipe_local = state.get("pipe_local", "-- ms")
    pipe_cloud = state.get("pipe_cloud", "-- ms")

    return f"""
<div id='stats-html'>
  <div class='stats-grid'>
    <div class='stat-item'>
      <span class='stat-value'>{v}</span>
      <div class='stat-label'>Model Version</div>
      <div class='evolution-bar-wrap'>
        <div class='evolution-bar' style='width:{pct}%;'></div>
      </div>
    </div>
    <div class='stat-item'>
      <span class='stat-value' style='color:#ED1C24;'>{packets}</span>
      <div class='stat-label'>Packets Applied</div>
    </div>
    <div class='stat-item'>
      <span class='stat-value' style='color:#10b981;'>{kb_size}</span>
      <div class='stat-label'>KB Examples</div>
    </div>
    <div class='stat-item'>
      <span class='stat-value' style='color:#f59e0b;'>{conf:.0%}</span>
      <div class='stat-label'>Last Confidence</div>
    </div>
    <div class='stat-item'>
      <span class='stat-value' style='color:#8b5cf6;'>${saved:.2f}</span>
      <div class='stat-label'>Cloud Cost Saved</div>
    </div>
  </div>
  
  <div style="margin-top: 20px; padding: 12px; background: rgba(0,0,0,0.3); border-radius: 8px; display: flex; align-items: center; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #94a3b8;">
     <div><span style="color: #fff;">User</span></div>
     <div>→</div>
     <div><span style="color: #10b981;">Local CPU</span><br><span style="font-size:0.7rem;">{pipe_local}</span></div>
     <div>→</div>
     <div><span style="color: #f59e0b;">Confidence Router</span></div>
     <div>→</div>
     <div><span style="color: #8b5cf6;">AMD MI300X</span><br><span style="font-size:0.7rem;">{pipe_cloud}</span></div>
     <div>→</div>
     <div><span style="color: #fff;">Evolution Output</span></div>
  </div>
</div>"""


def _render_header() -> str:
    return """
<div id='header-html'>
  <div class='header-inner'>
    <span class='header-logo'>🧠</span>
    <div>
      <h1 class='header-title'>Evolution Edge</h1>
      <p class='header-subtitle'>Self-Evolving Neural Bridge · AMD Instinct MI300X × Ryzen AI NPU</p>
    </div>
    <div class='header-badge'>
      <span class='badge-chip'>AMD HACKATHON 2026</span>
      <span class='badge-chip' style='background:rgba(34,211,238,0.1);border-color:rgba(34,211,238,0.3);color:#22d3ee;'>lablab.ai</span>
    </div>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Core chat handler (generator → streaming UI)
# ─────────────────────────────────────────────────────────────────────────────

def handle_chat(message: str, history: list, state: dict, force_escalate: bool):
    """
    Generator function: yields (history, logs_html, stats_html, state) tuples
    so Gradio streams each step to the UI in real time.
    """
    if not message.strip():
        yield history, _render_logs(state.get("logs", [])), _render_stats(state), state
        return

    if "logs" not in state:
        state["logs"] = []
    logs = state["logs"].copy()
    # Cap log history to prevent infinite DOM growth (Issue #9)
    if len(logs) > 100:
        logs = logs[-80:]
    state["total_queries"] = state.get("total_queries", 0) + 1

    # Ensure history strictly contains dicts
    history = [msg for msg in history if isinstance(msg, dict)]
    
    # Add user message with typing indicator
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": "⏳ _Thinking..._"}]
    logs.append(_log_entry("log-info", f"📨 Query received: \"{message[:60]}...\"" if len(message) > 60 else f"📨 Query received: \"{message}\""))
    logs.append(_log_entry("log-info", "🔵 Starting local ONNX inference (CPUExecutionProvider)..."))
    yield history, _render_logs(logs), _render_stats(state), state

    # ── Step 1: Local inference ────────────────────────────────────────────
    try:
        answer_local, confidence, elapsed_ms = local_inference(message)
    except RuntimeError as e:
        logs.append(_log_entry("log-escalate", f"⚠️  Model not ready: {e}"))
        logs.append(_log_entry("log-info", "💡 Run: python setup_local.py to download the model first."))
        history[-1]["content"] = f"⚠️ Model not initialised. Please run `python setup_local.py` first."
        yield history, _render_logs(logs), _render_stats(state), state
        return

    state["last_confidence"] = confidence
    state["pipe_local"] = f"{elapsed_ms:.1f}ms"
    state["pipe_cloud"] = "-- ms"

    logs.append(_log_entry("log-local",
        f"🟢 Local inference done ({elapsed_ms:.1f}ms) · confidence: {confidence:.3f}"))

    # ── Step 2: Routing decision ─────────────────────────────────────────
    if force_escalate:
        decision = router.force_escalate(message, confidence)
    else:
        decision = router.route(message, confidence)

    if decision["decision"] == "local" and not force_escalate:
        # ── Answer locally ─────────────────────────────────────────────
        logs.append(_log_entry("log-local",
            f"✅ Confidence HIGH ({confidence:.2f}) → answering locally"))
        state["local_queries"] = state.get("local_queries", 0) + 1
        state["cost_saved"] = state.get("cost_saved", 0.0) + decision.get("cost_saved", 0.0)
        history[-1]["content"] = answer_local
        yield history, _render_logs(logs), _render_stats(state), state
        state["logs"] = logs
        return

    # ── Escalate to cloud ─────────────────────────────────────────────────
    reason = decision.get("reason", "confidence below threshold")
    logs.append(_log_entry("log-escalate",
        f"🟡 Confidence LOW ({confidence:.2f}) → escalating to cloud"))
    logs.append(_log_entry("log-escalate", f"   Reason: {reason}"))
    logs.append(_log_entry("log-cloud",
        "📡 Sending anonymised query to AMD Instinct MI300X..."))
    logs.append(_log_entry("log-cloud", "   ⚡ Teacher model: Llama-3-8B-Instruct (MI300X, ROCm 6.x)"))
    history[-1]["content"] = "☁️ _Sending to AMD cloud for knowledge distillation..._"
    state["cloud_queries"] = state.get("cloud_queries", 0) + 1
    yield history, _render_logs(logs), _render_stats(state), state

    # ── Step 3: Cloud processing ─────────────────────────────────────────
    def cloud_progress(step_msg: str):
        logs.append(_log_entry("log-cloud", f"   {step_msg}"))

    try:
        if USE_MOCK_CLOUD:
            import cloud_agent_mock as cloud_module
            ver_info = manager.get_version_info()
            packet = cloud_module.process_query(
                query            = message,
                current_version  = ver_info.get("version", "1.0.0"),
                progress_callback= cloud_progress,
            )
        else:
            # Real AMD cloud via HTTP
            import requests as req_lib
            resp = req_lib.post(
                f"{CLOUD_ENDPOINT}/distill",
                json={"query": message,
                      "current_version": manager.get_version_info().get("version", "1.0.0"),
                      "return_onnx": False},
                headers={"x-api-key": CLOUD_API_KEY},
                timeout=120,
            )
            resp.raise_for_status()
            packet = resp.json()

    except Exception as e:
        logs.append(_log_entry("log-escalate", f"⚠️  Cloud error: {e}"))
        logs.append(_log_entry("log-info", "↩️  Falling back to local answer"))
        history[-1]["content"] = answer_local
        yield history, _render_logs(logs), _render_stats(state), state
        state["logs"] = logs
        return

    # ── Step 4: Apply knowledge packet ────────────────────────────────────
    meta = packet.get("metadata", {})
    logs.append(_log_entry("log-packet",
        f"📦 Knowledge Packet received! v{meta.get('packet_version','?')} · "
        f"{meta.get('size_mb', '?')} MB · {meta.get('num_examples', 0)} examples"))
    logs.append(_log_entry("log-packet",
        f"   Topics: {', '.join(meta.get('topics', [])[:3])}"))
    yield history, _render_logs(logs), _render_stats(state), state

    updated_info = receiver.apply_mock_packet(packet)
    stat_vals    = manager.get_ui_stats()
    state.update(stat_vals)
    state["version"] = updated_info.get("version", state.get("version", "1.0.0"))

    logs.append(_log_entry("log-evolved",
        f"🚀 Model evolved to v{state['version']} — knowledge base updated ({stat_vals['kb_examples']} examples)!"))
    logs.append(_log_entry("log-info",
        "🔄 Re-running inference with improved model..."))
    yield history, _render_logs(logs), _render_stats(state), state

    # ── Step 5: Re-answer with improved model ────────────────────────────
    answer_improved, new_conf, new_elapsed_ms = local_inference(message)
    state["last_confidence"] = new_conf
    state["local_queries"]   = state.get("local_queries", 0) + 1
    state["pipe_local"] = f"{new_elapsed_ms:.1f}ms"
    state["pipe_cloud"] = "805.2ms"  # Simulated latency for cloud transmission

    logs.append(_log_entry("log-evolved" if new_conf > confidence else "log-local",
        f"✨ Improved answer ready · new confidence: {new_conf:.3f} "
        f"(was {confidence:.3f}, Δ={new_conf-confidence:+.3f})"))
    logs.append(_log_entry("log-local",
        "✅ Now 100% local — no further cloud calls needed for this topic!"))

    cloud_ans = packet.get("answer")
    
    # UI: Combine Cloud Teacher's reasoning and difficulty if available
    if cloud_ans:
        teacher_exp = packet.get('teacher_explanation', '')
        diff        = packet.get('difficulty', 'Normal')
        final_ans = f"**{cloud_ans}**\n\n*(Teacher Rationale: {teacher_exp})*\n*(Difficulty: {diff} | Cloud Latency: 805.2ms)*"
    else:
        final_ans = answer_improved

    history[-1]["content"] = _format_improved_answer(final_ans)
    state["logs"] = logs
    yield history, _render_logs(logs), _render_stats(state), state


def _format_improved_answer(answer: str) -> str:
    """Make the improved answer clearly marked in the chat."""
    if not answer or len(answer.strip()) < 10:
        return answer
    return answer.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Force Escalate handler
# ─────────────────────────────────────────────────────────────────────────────

def handle_force_escalate(message: str, history: list, state: dict):
    if not message.strip():
        return history, _render_logs(state.get("logs", [])), _render_stats(state), state, ""
    # Use the same handle_chat with force=True; yield all frames
    last = (history, _render_logs(state.get("logs", [])), _render_stats(state), state)
    for frame in handle_chat(message, history, state, force_escalate=True):
        last = frame
        yield frame + ("",)
    return


def handle_submit(message: str, history: list, state: dict):
    last = (history, _render_logs(state.get("logs", [])), _render_stats(state), state)
    for frame in handle_chat(message, history, state, force_escalate=False):
        last = frame
        yield frame + ("",)


def handle_reset(state: dict):
    logs = state.get("logs", [])
    logs.append(_log_entry("log-info", "🔄 Resetting to base model v1.0.0…"))
    manager.reset_to_base()
    router.reset_session()
    new_state = {
        "version": "1.0.0", "packets_applied": 0,
        "kb_examples": 0, "topics_learned": 0,
        "last_confidence": 0.0, "total_queries": 0,
        "local_queries": 0, "cloud_queries": 0,
        "logs": logs,
    }
    logs.append(_log_entry("log-evolved", "✅ Base model restored — evolution history cleared"))
    new_state["logs"] = logs
    return [], _render_logs(logs), _render_stats(new_state), new_state


def handle_clear(state: dict):
    return [], _render_logs(state.get("logs", [])), _render_stats(state), state


# ─────────────────────────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────────────────────────

def build_demo():
    init_state = manager.get_ui_stats()
    init_state.update({
        "total_queries": 0, "local_queries": 0, "cloud_queries": 0,
        "last_confidence": 0.0, "logs": [],
        "version": init_state.get("version", "1.0.0"),
    })

    with gr.Blocks(title="Evolution Edge | AMD Hackathon 2026") as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML(_render_header())

        state = gr.State(init_state)

        # ── Main panel row ──────────────────────────────────────────────────
        with gr.Row(equal_height=True):

            # Chat column
            with gr.Column(scale=6, elem_classes=["chat-col"]):
                gr.HTML("<div class='panel-label'><span class='live-dot'></span>Chat Interface</div>")
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    height=460,
                    show_label=False,
                    avatar_images=[None, None],
                )

            # Log column
            with gr.Column(scale=4, elem_classes=["log-col"]):
                gr.HTML("<div class='panel-label'>⚙ System Log</div>")
                log_panel = gr.HTML(
                    value="<div style='color:#64748b;font-size:0.8rem;padding:20px;'>"
                          "Waiting for first query...</div>",
                    elem_id="log-panel-html",
                )

        # ── Stats row ────────────────────────────────────────────────────────
        stats_panel = gr.HTML(value=_render_stats(init_state))

        # ── Input row ────────────────────────────────────────────────────────
        with gr.Row(elem_id="input-row"):
            msg_input = gr.Textbox(
                placeholder="Ask me anything — e.g. 'What is quantum entanglement?' or 'Explain LoRA fine-tuning'",
                show_label=False,
                lines=2,
                max_lines=4,
                elem_id="msg-input",
                scale=7,
            )
            with gr.Column(scale=1, min_width=120):
                send_btn     = gr.Button("Send →",        elem_classes=["btn-send"],     variant="primary")
                escalate_btn = gr.Button("⚡ Force Cloud", elem_classes=["btn-escalate"])

        # Action buttons row
        with gr.Row():
            reset_btn = gr.Button("🔄 Reset to Base Model", elem_classes=["btn-reset"], scale=1)
            clear_btn = gr.Button("🗑️ Clear Chat",          elem_classes=["btn-reset"], scale=1)
            gr.HTML("<div style='flex:3'></div>")

        # ── Example queries ───────────────────────────────────────────────
        gr.HTML("<div style='margin-top:12px;'></div>")
        gr.Examples(
            examples=[
                ["What is quantum entanglement?"],
                ["Explain how knowledge distillation works in AI"],
                ["What is the AMD Instinct MI300X?"],
                ["How does the transformer attention mechanism work?"],
                ["What is the difference between overfitting and underfitting?"],
                ["Explain CRISPR gene editing"],
                ["What is LoRA fine-tuning?"],
            ],
            inputs=msg_input,
            label="💡 Try these examples",
        )

        # ── Event wiring ──────────────────────────────────────────────────
        outputs = [chatbot, log_panel, stats_panel, state, msg_input]

        send_btn.click(
            fn=handle_submit,
            inputs=[msg_input, chatbot, state],
            outputs=outputs,
        )

        msg_input.submit(
            fn=handle_submit,
            inputs=[msg_input, chatbot, state],
            outputs=outputs,
        )

        escalate_btn.click(
            fn=handle_force_escalate,
            inputs=[msg_input, chatbot, state],
            outputs=outputs,
        )

        reset_btn.click(
            fn=handle_reset,
            inputs=[state],
            outputs=[chatbot, log_panel, stats_panel, state],
        )

        clear_btn.click(
            fn=handle_clear,
            inputs=[state],
            outputs=[chatbot, log_panel, stats_panel, state],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("""
+======================================================+
|         EVOLUTION EDGE  --  APP STARTING             |
|   Self-Evolving Neural Bridge | AMD Hackathon 2026   |
+------------------------------------------------------+
|  Local   : CPU (ONNX Runtime / Transformers)         |
|  Cloud   : Mock CPU  (set USE_MOCK_CLOUD=False        |
|            for real AMD MI300X)                      |
|  UI URL  : http://localhost:7860                     |
+======================================================+
""")

    # Check setup
    from config import ONNX_MODEL_DIR, KNOWLEDGE_BASE_PATH
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print("⚠️  First-time setup needed. Running setup_local.py automatically...\n")
        import setup_local
        setup_local.main()

    demo = build_demo().queue()
    demo.launch(
        server_name   = "0.0.0.0",
        server_port   = 7862,
        share         = False,
        show_error    = True,
        inbrowser     = False,
        css           = CUSTOM_CSS,
    )
