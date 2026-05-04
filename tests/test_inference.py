import unittest
from local_model.model_manager import ModelManager

class TestInferenceMemory(unittest.TestCase):
    def setUp(self):
        self.manager = ModelManager()

    def test_jaccard_retrieval_empty(self):
        # Mock KB size to empty
        from unittest.mock import patch
        with patch.object(self.manager, 'get_knowledge_base', return_value={"examples": []}):
            matches = self.manager.get_relevant_examples("what is AI?")
            self.assertEqual(len(matches), 0)

    def test_jaccard_retrieval_matches(self):
        # Mock KB with dummy examples
        dummy_kb = {
            "examples": [
                {"q": "what is AI?", "a": "Artificial intelligence"},
                {"q": "where is paris?", "a": "France"}
            ]
        }
        from unittest.mock import patch
        with patch.object(self.manager, 'get_knowledge_base', return_value=dummy_kb):
            matches = self.manager.get_relevant_examples("what is AI programming?", top_k=1)
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["q"], "what is AI?")

if __name__ == '__main__':
    unittest.main()
