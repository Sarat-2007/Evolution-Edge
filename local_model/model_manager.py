"""
Evolution Edge – Model Manager
================================
Handles ONNX model lifecycle:
  - Loading / hot-reloading
  - Applying knowledge packets (updating knowledge base)
  - Bumping the model version tracker
  - Querying evolution stats

This is the core of the 'lifelong learning' loop on the local device.
"""

import os, json, shutil, logging
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class ModelManager:
    """Singleton-style manager for local ONNX model state."""

    def __init__(self):
        from config import (
            ONNX_MODEL_DIR, TOKENIZER_DIR, KNOWLEDGE_BASE_PATH,
            MODEL_VERSION_FILE, KNOWLEDGE_PACKETS_DIR,
        )
        self.onnx_dir          = ONNX_MODEL_DIR
        self.tokenizer_dir     = TOKENIZER_DIR
        self.kb_path           = KNOWLEDGE_BASE_PATH
        self.version_path      = MODEL_VERSION_FILE
        self.packets_dir       = KNOWLEDGE_PACKETS_DIR

    # ── Version & Stats ───────────────────────────────────────────────────────

    def get_version_info(self) -> dict:
        if not os.path.exists(self.version_path):
            return {"version": "1.0.0", "packets_applied": 0, "improvements": []}
        with open(self.version_path) as f:
            return json.load(f)

    def _save_version_info(self, info: dict):
        info["last_updated"] = datetime.now().isoformat()
        with open(self.version_path, "w") as f:
            json.dump(info, f, indent=2)

    def _bump_version(self, info: dict) -> str:
        """Increment patch version: 1.0.0 → 1.0.1 → …"""
        parts = info.get("version", "1.0.0").split(".")
        parts[2] = str(int(parts[2]) + 1)
        return ".".join(parts)

    # ── Knowledge Base ────────────────────────────────────────────────────────

    def get_knowledge_base(self) -> dict:
        if not os.path.exists(self.kb_path):
            return {"examples": [], "topics": [], "stats": {}}
        with open(self.kb_path) as f:
            return json.load(f)

    def _save_knowledge_base(self, kb: dict):
        import tempfile
        # Atomic write: write to temp file, then replace original
        dir_name = os.path.dirname(self.kb_path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(kb, f, indent=2)
            os.replace(tmp_path, self.kb_path)  # Atomic on all OSes
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def get_relevant_examples(self, query: str, top_k: int = 2) -> list:
        """
        Pseudo-learning effect: Match the most relevant past examples 
        from the knowledge base using Jaccard similarity.
        """
        kb = self.get_knowledge_base()
        examples = kb.get("examples", [])
        if not examples:
            return []
            
        def jaccard(a: str, b: str) -> float:
            # Simple bag-of-words similarity
            a_set = set(a.lower().split())
            b_set = set(b.lower().split())
            if not a_set or not b_set: return 0.0
            return len(a_set.intersection(b_set)) / len(a_set.union(b_set))
            
        scored = []
        for ex in examples:
            score = jaccard(query, ex["q"])
            scored.append((score, ex))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        # Return top_k matches, even if similarity is low, it anchors context better than pure random
        return [ex for score, ex in scored[:top_k]]

    def knowledge_base_size(self) -> int:
        return len(self.get_knowledge_base().get("examples", []))

    # ── Packet Application ────────────────────────────────────────────────────

    def apply_packet(self, packet: dict) -> dict:
        """
        Apply a knowledge packet received from the cloud.

        The packet dict contains:
          {
            "answer":        str    – the improved answer from cloud
            "examples":      list   – new Q/A teaching pairs
            "topics":        list   – topics covered
            "onnx_bytes":    bytes  – updated ONNX model bytes (optional)
            "metadata":      dict   – version info, size, timestamp
          }

        Returns updated version info dict.
        """
        meta = packet.get("metadata", {})
        log.info(f"Applying knowledge packet v{meta.get('packet_version', '?')} …")

        # ── 1. Merge new examples into knowledge base ──────────────────────
        kb       = self.get_knowledge_base()
        new_exs  = packet.get("examples", [])
        existing = {ex["q"] for ex in kb.get("examples", [])}

        added = 0
        for ex in new_exs:
            if ex["q"] not in existing:
                kb["examples"].append(ex)
                existing.add(ex["q"])
                added += 1

        # Update topics
        for topic in packet.get("topics", []):
            if topic not in kb.get("topics", []):
                kb.setdefault("topics", []).append(topic)

        # Update stats
        kb.setdefault("stats", {})
        kb["stats"]["packets_applied"] = kb["stats"].get("packets_applied", 0) + 1
        kb["stats"]["total_examples"]  = len(kb["examples"])
        kb["stats"]["last_updated"]    = datetime.now().isoformat()
        self._save_knowledge_base(kb)
        log.info(f"Knowledge base: +{added} new examples (total: {len(kb['examples'])})")

        # ── 2. Optionally replace ONNX model file ────────────────────────────
        onnx_bytes = packet.get("onnx_bytes")
        if onnx_bytes:
            onnx_path = os.path.join(self.onnx_dir, "model.onnx")
            backup    = onnx_path + ".bak"
            if os.path.exists(onnx_path):
                shutil.copy2(onnx_path, backup)
            with open(onnx_path, "wb") as f:
                f.write(onnx_bytes)
            log.info(f"ONNX model updated ({len(onnx_bytes)/1024/1024:.1f} MB)")
            # Hot-reload
            from local_model.inference import reload_model
            reload_model()

        # ── 3. Save packet to history dir ────────────────────────────────────
        self._archive_packet(packet, meta)

        # ── 4. Bump version ───────────────────────────────────────────────────
        info                 = self.get_version_info()
        info["version"]      = self._bump_version(info)
        info["packets_applied"] = info.get("packets_applied", 0) + 1
        info.setdefault("improvements", []).append({
            "packet_version": meta.get("packet_version", "?"),
            "topics":         packet.get("topics", []),
            "examples_added": added,
            "applied_at":     datetime.now().isoformat(),
        })
        self._save_version_info(info)

        log.info(f"✅ Model evolved to v{info['version']}")
        return info

    def _archive_packet(self, packet: dict, meta: dict):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"packet_{meta.get('packet_version','?')}_{ts}.json"
        path = os.path.join(self.packets_dir, name)
        safe = {k: v for k, v in packet.items() if k != "onnx_bytes"}
        with open(path, "w") as f:
            json.dump(safe, f, indent=2)
        log.debug(f"Packet archived → {path}")

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset_to_base(self):
        """
        Clear knowledge base and reset version tracker.
        Used for the demo 'Reset to Base Model' button.
        """
        kb = {"version": "1.0.0", "created": datetime.now().isoformat(),
              "examples": [], "topics": [], "stats": {}}
        self._save_knowledge_base(kb)

        info = self.get_version_info()
        info["version"]         = "1.0.0"
        info["packets_applied"] = 0
        info["improvements"]    = []
        self._save_version_info(info)

        # Reload model (clears any cached inference state)
        from local_model.inference import reload_model
        reload_model()

        log.info("🔄 Model reset to base version 1.0.0")

    # ── Summary for UI ────────────────────────────────────────────────────────

    def get_ui_stats(self) -> dict:
        info = self.get_version_info()
        kb   = self.get_knowledge_base()
        return {
            "version":           info.get("version", "1.0.0"),
            "packets_applied":   info.get("packets_applied", 0),
            "kb_examples":       len(kb.get("examples", [])),
            "topics_learned":    len(kb.get("topics", [])),
            "last_updated":      info.get("last_updated", "never"),
        }
