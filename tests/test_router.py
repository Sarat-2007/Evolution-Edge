import unittest
from local_model.router import SymbolicRouter

class TestSymbolicRouter(unittest.TestCase):
    def setUp(self):
        self.router = SymbolicRouter(base_threshold=0.65)
        # Mock the ModelManager dependency for dynamic threshold
        from unittest.mock import patch
        self.mock_patch = patch('local_model.model_manager.ModelManager')
        self.mock_manager = self.mock_patch.start()
        self.mock_manager.return_value.knowledge_base_size.return_value = 0

    def tearDown(self):
        self.mock_patch.stop()

    def test_dynamic_threshold_initial(self):
        self.assertEqual(self.router._get_dynamic_threshold(), 0.65)

    def test_dynamic_threshold_experienced(self):
        self.mock_manager.return_value.knowledge_base_size.return_value = 10
        # Reduced by 10 * 0.02 = 0.20, max capped at 0.15 → 0.50
        self.assertAlmostEqual(self.router._get_dynamic_threshold(), 0.50)

    def test_high_confidence_local(self):
        decision = self.router.route("hello there", 0.90)
        self.assertEqual(decision["decision"], "local")
        self.assertEqual(decision["cost_saved"], 0.04)

    def test_low_confidence_escalation(self):
        decision = self.router.route("what is quantum computing?", 0.40)
        self.assertEqual(decision["decision"], "escalate")
        self.assertEqual(decision["cost_saved"], 0.0)

    def test_keyword_escalation_low_confidence(self):
        """Keywords should escalate when confidence is below 0.95"""
        decision = self.router.route("please explain string theory", 0.90)
        self.assertEqual(decision["decision"], "escalate")
        self.assertTrue(decision["keyword_triggered"])

    def test_keyword_bypass_high_confidence(self):
        """Keywords should NOT escalate when confidence is above 0.95 (truly mastered topic)"""
        decision = self.router.route("please explain string theory", 0.99)
        self.assertEqual(decision["decision"], "local")
        self.assertEqual(decision["cost_saved"], 0.04)

if __name__ == '__main__':
    unittest.main()
