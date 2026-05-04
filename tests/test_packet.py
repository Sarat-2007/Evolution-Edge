import unittest
from knowledge_packet.packet_builder import PacketBuilder
from knowledge_packet.packet_receiver import PacketReceiver

class TestPacketLogic(unittest.TestCase):
    def test_packet_validation_missing_fields(self):
        invalid_packet = {"examples": []}
        self.assertFalse(PacketBuilder.validate(invalid_packet))

    def test_packet_validation_valid(self):
        valid_packet = {
            "answer": "Test Answer",
            "examples": [{"q": "test", "a": "test"}],
            "topics": ["test"],
            "metadata": {}
        }
        self.assertTrue(PacketBuilder.validate(valid_packet))

    def test_packet_fallback(self):
        packet = PacketReceiver.make_fallback_packet("test query")
        self.assertEqual(packet["metadata"]["provider"], "fallback")
        self.assertIn("answer", packet)

if __name__ == '__main__':
    unittest.main()
