import unittest
from unittest.mock import patch, MagicMock
import logging
import numpy as np

# Import your main function from evaluate.py
from evaluate import main_function

logging.basicConfig(level=logging.DEBUG)

class TestEvaluationIntegration(unittest.TestCase):

    @patch('evaluate.load_dataset')
    @patch('evaluate.SentenceTransformer')
    @patch('evaluate.util')
    @patch('evaluate.AutoTokenizer.from_pretrained')
    @patch('evaluate.AutoModelForCausalLM.from_pretrained')
    @patch('evaluate.pipeline')
    @patch('evaluate.load_metric')
    @patch('evaluate.wandb')
    @patch('evaluate.torch.cuda.is_available')
    def test_end_to_end_with_icl_flag(
            self,
            mock_cuda,
            mock_wandb,
            mock_load_metric,
            mock_pipeline,
            mock_model_class,
            mock_tokenizer_class,
            mock_util,
            mock_sbert_class,
            mock_load_dataset
        ):
        logging.info("Starting end-to-end evaluation integration test")

        # Setup dummy dataset
        dummy_test_data = [
            {"instruction": "Instr1", "input": "Input1", "output": "Output1"},
            {"instruction": "Instr2", "input": "Input2", "output": "Output2"},
            {"instruction": "Instr3", "input": "Input3", "output": "Output3"},
            {"instruction": "Instr4", "input": "Input4", "output": "Output4"},
        ]
        mock_load_dataset.return_value = dummy_test_data
        logging.debug("Mocked load_dataset to return dummy test data")

        # Mock SentenceTransformer.encode returns tensor-like with shape for sim
        mock_sbert_instance = MagicMock()
        mock_sbert_instance.encode.side_effect = lambda x, **kwargs: MagicMock(shape=(len(x), 768))
        mock_sbert_class.return_value = mock_sbert_instance

        # Mock cosine similarity returns a tensor with shape [batch, dataset]
        class DummyTensor:
            def __getitem__(self, idx):
                return [0.9, 0.8, 0.7, 0.6]
            def shape(self):
                return (4,)
        mock_util.cos_sim.return_value = [DummyTensor()]

        # Mock tokenizer.from_pretrained returns a tokenizer mock with eos_token_id property
        mock_tokenizer = MagicMock()
        type(mock_tokenizer).eos_token_id = property(lambda self: 50256)
        mock_tokenizer_class.return_value = mock_tokenizer

        # Mock model loading
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        # Mock pipeline returns list of dicts as generation output
        mock_generator_instance = MagicMock()
        mock_generator_instance.return_value = [
            {"generated_text": "Generated output 1"},
            {"generated_text": "Generated output 2"},
            {"generated_text": "Generated output 3"},
            {"generated_text": "Generated output 4"},
        ]
        mock_pipeline.return_value = mock_generator_instance

        # Mock bertscore.compute returns dummy metrics
        mock_bertscore_compute = MagicMock()
        mock_bertscore_compute.return_value = {
            "precision": [0.9, 0.85, 0.87, 0.88],
            "recall": [0.8, 0.82, 0.83, 0.81],
            "f1": [0.85, 0.83, 0.85, 0.84],
        }
        mock_load_metric.return_value.compute = mock_bertscore_compute

        # Mock wandb.init and wandb.log
        mock_wandb.init.return_value = None
        mock_wandb.log.return_value = None
        mock_wandb.finish.return_value = None

        # Mock torch.cuda.is_available to True
        mock_cuda.return_value = True

        # CLI arguments with ICL enabled
        cli_args = [
            '--test_dataset', 'fake/dataset',
            '--model_id', 'fake/model',
            '--project', 'testproject',
            '--user_id', 'testuser',
            '--use_icl'
        ]

        # Run the main evaluation function
        ret_val = main_function(cli_args)
        logging.info(f"main_function returned: {ret_val}")

        # Assertions on function outputs and calls
        self.assertEqual(ret_val, 0, "main_function should return 0 on success")
        mock_load_dataset.assert_called_once_with('fake/dataset', split='test', token=None)
        mock_tokenizer_class.assert_called_once_with('fake/model', trust_remote_code=True)
        mock_model_class.assert_called_once_with('fake/model', trust_remote_code=True)
        mock_pipeline.assert_called_once()
        mock_load_metric.assert_called_once_with('bertscore')
        mock_wandb.init.assert_called_once()
        mock_wandb.log.assert_called()  # verify logging calls
        mock_wandb.finish.assert_called_once()

        # Log some of the wandb calls for inspection
        logs = [call[0][0] for call in mock_wandb.log.call_args_list]
        logging.debug(f"wandb logged {len(logs)} metric entries: {logs}")

if __name__ == "__main__":
    unittest.main()
