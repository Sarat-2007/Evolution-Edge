"""
Evolution Edge – Knowledge Packet Receiver
==========================================
Runs on the LOCAL side.

Receives a knowledge packet from the cloud (via HTTP or mock),
validates it, and hands it to the ModelManager for integration.

Supports two transport modes:
  1. Mock (local): packet dict passed in-process
  2. HTTP (real AMD cloud): download JSON from CLOUD_ENDPOINT
"""

import os, json, time, logging
from typing import Optional

log = logging.getLogger(__name__)


class PacketReceiver:
    """Receives and validates knowledge packets from cloud or mock."""

    def __init__(self, model_manager):
        self.manager = model_manager

    # ── HTTP transport (real AMD cloud) ───────────────────────────────────────

    def download_and_apply(self, packet_url: str, api_key: str) -> dict:
        """
        Download a knowledge packet from AMD cloud endpoint and apply it.
        Called when USE_MOCK_CLOUD = False.
        """
        from config import CLOUD_TIMEOUT
        import requests

        log.info(f"📡 Downloading knowledge packet from {packet_url}")
        try:
            resp = requests.get(
                packet_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=CLOUD_TIMEOUT,
            )
            resp.raise_for_status()
            packet = resp.json()
        except requests.RequestException as e:
            log.error(f"Failed to download packet: {e}")
            raise ConnectionError(f"Cloud packet download failed: {e}") from e

        return self._apply_validated(packet)

    # ── In-process transport (mock cloud) ─────────────────────────────────────

    def apply_mock_packet(self, packet: dict) -> dict:
        """Apply a packet received in-process from cloud_agent_mock.py."""
        return self._apply_validated(packet)

    # ── Core apply logic ──────────────────────────────────────────────────────

    def _apply_validated(self, packet: dict) -> dict:
        from knowledge_packet.packet_builder import PacketBuilder
        if not PacketBuilder.validate(packet):
            raise ValueError("Invalid knowledge packet — missing required fields")

        meta = packet.get("metadata", {})
        log.info(
            f"📦 Received packet v{meta.get('packet_version','?')} | "
            f"{meta.get('num_examples', 0)} examples | "
            f"{meta.get('size_mb', 0):.1f} MB"
        )
        if "embedding" in packet:
            log.info("💎 Packet quantified & compressed for AMD NPU Execution.")

        result = self.manager.apply_packet(packet)
        log.info(f"✅ Packet applied — model evolved to v{result['version']}")
        return result

    # ── Fallback packet (graceful degradation) ────────────────────────────────

    @staticmethod
    def make_fallback_packet(query: str) -> dict:
        """
        If cloud is unreachable, return a minimal packet with a generic
        apology + pointer to try again.
        """
        return {
            "answer":   "I encountered an issue reaching the cloud. "
                        "Please try again in a moment.",
            "examples": [],
            "topics":   [],
            "metadata": {
                "packet_version": "fallback",
                "size_mb": 0.0,
                "num_examples": 0,
                "provider": "fallback",
                "created": "",
            },
        }
