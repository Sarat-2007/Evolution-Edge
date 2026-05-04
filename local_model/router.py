"""
Evolution Edge – Symbolic Confidence Router
============================================
Decides whether a query should be:
  • Answered locally (high confidence, familiar topic)
  • Escalated to AMD cloud (low confidence OR new/complex topic)

This is the "Symbolic" half of the neuro-symbolic architecture:
  - Neuro  → the ONNX model generates an answer + raw confidence
  - Symbol → this router applies rule-based logic to make the final
             routing decision (keyword triggers, topic novelty, etc.)
"""

import re
import logging
from config import CONFIDENCE_THRESHOLD, ESCALATION_KEYWORDS

log = logging.getLogger(__name__)


class SymbolicRouter:
    """Neuro-symbolic router: combines model confidence with rule-based logic."""

    def __init__(self, base_threshold: float = CONFIDENCE_THRESHOLD):
        self.base_threshold     = base_threshold
        self.escalation_keywords = [kw.lower() for kw in ESCALATION_KEYWORDS]
        self._session_topics: set[str] = set()   # topics seen this session
        self._manager = None  # Cached ModelManager instance
        
    def _get_dynamic_threshold(self) -> float:
        """Adaptive threshold based on history size (simulated dynamic logic)"""
        if self._manager is None:
            from local_model.model_manager import ModelManager
            self._manager = ModelManager()
        size = self._manager.knowledge_base_size()
        # Reduce threshold by 0.02 for each known example, max reduction 0.15
        reduction = min(0.15, size * 0.02)
        return self.base_threshold - reduction

    # ── Public API ────────────────────────────────────────────────────────────

    def route(self, query: str, confidence: float) -> dict:
        """
        Decide routing for a query given the model's confidence score.

        Returns a dict:
          {
            "decision":  "local" | "escalate",
            "reason":    str,
            "confidence": float,
            "keyword_triggered": bool,
            "new_topic": bool,
          }
        """
        query_lower = query.lower().strip()

        # Calculate adaptive threshold
        current_threshold = self._get_dynamic_threshold()
        
        # Check if the local model has already learned this exact concept (Semantic Context)
        learned_context = self._manager.get_relevant_examples(query_lower, top_k=1)
        is_concept_learned = len(learned_context) > 0

        # Rule 1: model confidence below threshold (but bypass if learned)
        if confidence < current_threshold and not is_concept_learned:
            return self._decide("escalate", confidence,
                                f"Confidence {confidence:.2f} < adaptive threshold {current_threshold:.2f}",
                                False, False)

        # Rule 2: explicit complexity / explanation keywords
        keyword_hit = self._check_keywords(query_lower)
        # Bypassed ONLY if the model has genuinely mastered the concept (confidence > 0.95)
        # OR if the concept is safely cached in local Jaccard Knowledge Base!
        if keyword_hit and confidence < 0.95 and not is_concept_learned:
            return self._decide("escalate", confidence,
                                f"Escalation keyword detected: '{keyword_hit}'",
                                True, False)

        # Rule 3: new topic (not seen in this session)
        topic = self._extract_topic(query_lower)
        if topic and topic not in self._session_topics:
            # Only escalate on new topics if confidence is not very high
            if confidence < 0.80:
                self._session_topics.add(topic)
                return self._decide("escalate", confidence,
                                    f"New topic detected: '{topic}'",
                                    False, True)
            self._session_topics.add(topic)

        # Default: answer locally
        return self._decide("local", confidence,
                            f"Confidence {confidence:.2f} ≥ adaptive threshold {current_threshold:.2f}",
                            False, False)

    def force_escalate(self, query: str, confidence: float) -> dict:
        """Force escalation (demo 'Force Escalate' button)."""
        return self._decide("escalate", confidence,
                            "Manually forced escalation (demo mode)", False, False)

    def register_topic(self, topic: str):
        """Mark a topic as known (called after knowledge packet applied)."""
        self._session_topics.add(topic.lower())

    def reset_session(self):
        """Clear session topic memory (reset to base model state)."""
        self._session_topics.clear()
        log.info("Router: session topic memory cleared")

    # ── Internal Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _decide(decision, confidence, reason, keyword_triggered, new_topic) -> dict:
        # Simulate Cost-Awareness (Local compute = $0.00, Cloud API ≈ $0.04)
        cost_savings = 0.04 if decision == "local" else 0.00
        
        log.info(f"Router → {decision.upper()} | {reason} | Cost saved: ${cost_savings:.2f}")
        return {
            "decision":          decision,
            "reason":            reason,
            "confidence":        confidence,
            "keyword_triggered": keyword_triggered,
            "new_topic":         new_topic,
            "cost_saved":        cost_savings
        }

    def _check_keywords(self, query: str) -> str | None:
        for kw in self.escalation_keywords:
            if kw in query:
                return kw
        return None

    def _extract_topic(self, query: str) -> str | None:
        """
        Rough topic extraction: grab the main noun phrase after
        known question stems. Good enough for routing logic.
        """
        stems = [
            r"what is (?:a |an |the )?(.+?)(?:\?|$)",
            r"explain (.+?)(?:\?|$)",
            r"how does (.+?) work",
            r"tell me about (.+?)(?:\?|$)",
            r"describe (.+?)(?:\?|$)",
        ]
        for pattern in stems:
            m = re.search(pattern, query)
            if m:
                return m.group(1).strip()[:40]
        # Fallback: first 3 meaningful words
        words = [w for w in query.split() if len(w) > 3]
        return " ".join(words[:3]) if words else None
