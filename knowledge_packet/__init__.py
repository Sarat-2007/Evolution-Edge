"""
Evolution Edge – Knowledge Packet Package
Handles building, transmitting, and applying knowledge packets
(ONNX model updates + curated Q&A pairs) from cloud to local device.
"""
from .packet_builder import PacketBuilder
from .packet_receiver import PacketReceiver

__all__ = ["PacketBuilder", "PacketReceiver"]
