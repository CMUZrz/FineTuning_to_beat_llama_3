import unittest
from unittest.mock import patch, MagicMock
import os

# Importing  main function from dataset_tokenizer.py
from dataset_tokenizer import main_function

class TestEndToEndIntegration(unittest.TestCase):

    @patch('dataset_tokenizer.SentenceTransformer')
    @patch('dataset_tokenizer.AutoTokenizer')
    @patch('dataset_tokenizer.load_dataset')
    @patch.dict(os.environ, {"HUGGINGFACE_HUB_TOKEN": "dummy_token"})
    def test_end_to_end_standard_pipeline(self, mock_load_dataset, mock_tokenizer_class, mock_sentence_transformer_class):
        # Mock a dataset with >4000 rows
        dummy_data = [{"instruction": f"Do X{i}", "input": f"Q{i}", "output": f"A{i}"} for i in range(4000)]

        mock_dataset = MagicMock()
        mock_dataset.shuffle.return_value = mock_dataset
        mock_dataset.select.return_value = dummy_data[:4000]
        mock_dataset.train_test_split.side_effect = lambda test_size, seed: {
            'train': dummy_data[:3200],
            'test': dummy_data[3200:4000]
        }

        mock_load_dataset.return_value = mock_dataset

        # Mock tokenizer/model
        mock_tokenizer = MagicMock()
        mock_tokenizer.__call__.return_value = {
            "input_ids": [[1, 2, 3]],
            "labels": [[-100, 2, 3]],
            "offset_mapping": [[(0,0), (1,1), (2,2)]]
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        mock_sentence_transformer_class.return_value = mock_model

        # Patch push_to_hub to avoid network calls
        with patch.object(mock_dataset, 'push_to_hub', return_value=True):
            result = main_function(cli_args=[])
            self.assertIn("tokenized_train_dataset", result)
            self.assertIn("tokenized_val_dataset", result)
            self.assertIn("tokenized_test_dataset", result)
            self.assertIn("train_dataset", result)
            self.assertIn("val_dataset", result)
            self.assertIn("test_dataset", result)

    @patch('dataset_tokenizer.SentenceTransformer')
    @patch('dataset_tokenizer.AutoTokenizer')
    @patch('dataset_tokenizer.load_dataset')
    @patch.dict(os.environ, {"HUGGINGFACE_HUB_TOKEN": "dummy_token"})
    def test_end_to_end_icl_pipeline(self, mock_load_dataset, mock_tokenizer_class, mock_sentence_transformer_class):
        dummy_data = [{"instruction": f"Do X{i}", "input": f"Q{i}", "output": f"A{i}"} for i in range(4000)]

        mock_dataset = MagicMock()
        mock_dataset.shuffle.return_value = mock_dataset
        mock_dataset.select.return_value = dummy_data[:4000]
        mock_dataset.train_test_split.side_effect = lambda test_size, seed: {
            'train': dummy_data[:3200],
            'test': dummy_data[3200:4000]
        }

        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_tokenizer.__call__.return_value = {
            "input_ids": [[1, 2, 3]],
            "labels": [[-100, 2, 3]],
            "offset_mapping": [[(0,0), (1,1), (2,2)]]
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        mock_sentence_transformer_class.return_value = mock_model

        with patch('dataset_tokenizer.torch.save'), \
             patch('dataset_tokenizer.torch.load', return_value={
                 'embeddings': [0.1, 0.2], 'patient_inputs': ['Q1', 'Q2'], 'doctor_outputs': ['A1', 'A2']
             }):
            result = main_function(cli_args=['--icl'])
            self.assertIn("tokenized_train_ICL", result)
            self.assertIn("train_dataset", result)
            self.assertIn("val_dataset", result)
            self.assertIn("test_dataset", result)

if __name__ == "__main__":
    unittest.main()
