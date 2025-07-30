# Test 

import unittest
from unittest.mock import patch, MagicMock
import os
import logging

# Import your main_function from Fine_Tune.py
from Fine_Tune import main_function

logging.basicConfig(level=logging.DEBUG)

class TestQLoRAIntegration(unittest.TestCase):
    
    @patch('Fine_Tune.HfApi.create_repo')
    @patch('Fine_Tune.Trainer.push_to_hub')
    @patch('Fine_Tune.Trainer.save_model')
    @patch('Fine_Tune.AutoTokenizer.from_pretrained')
    @patch('Fine_Tune.AutoModelForCausalLM.from_pretrained')
    @patch('Fine_Tune.load_dataset')
    @patch('Fine_Tune.get_peft_model')
    @patch('Fine_Tune.prepare_model_for_kbit_training')
    @patch('Fine_Tune.LoraConfig')
    @patch('Fine_Tune.Trainer')
    @patch('Fine_Tune.set_verbosity_error')
    @patch('Fine_Tune.trainer_utils.get_last_checkpoint')
    def test_end_to_end_training_pipeline(
        self, mock_get_ckpt, mock_set_verbosity, mock_trainer_class,
        mock_lora_config, mock_prepare_kbit, mock_get_peft, mock_load_ds,
        mock_model_pretrained, mock_tokenizer_pretrained, mock_save_model,
        mock_push_to_hub, mock_create_repo
    ):
        logging.info("[TEST] Starting end-to-end QLoRA integration test")

        # Setup mocks and their return values

        # Mock dataset returns
        dummy_train = [ {'input_ids': [1,2,3], 'labels': [1,2,3]} ] * 10
        dummy_val = [ {'input_ids': [4,5,6], 'labels': [4,5,6]} ] * 3
        dummy_test = [ {'input_ids': [7,8,9], 'labels': [7,8,9]} ] * 2
        mock_load_ds.side_effect = [dummy_train, dummy_val, dummy_test]

        logging.debug("[MOCK] load_dataset patched to return dummy datasets")

        # Mock tokenizer to an object with pad_token property
        mock_tokenizer = MagicMock()
        type(mock_tokenizer).pad_token = property(lambda self: '<pad>')
        mock_tokenizer_pretrained.return_value = mock_tokenizer
        logging.debug("[MOCK] AutoTokenizer.from_pretrained patched")

        # Mock model returned from from_pretrained
        mock_model = MagicMock()
        mock_model.gradient_checkpointing_enable.return_value = None
        mock_model.config = MagicMock()
        mock_model.config.use_cache = True
        mock_model_pretrained.return_value = mock_model
        logging.debug("[MOCK] AutoModelForCausalLM.from_pretrained patched")

        # LoRA config mock returns itself
        mock_lora_config.return_value = MagicMock()
        # prepare model returns the same model
        mock_prepare_kbit.return_value = mock_model
        # get_peft_model returns the same model augmented
        mock_get_peft.return_value = mock_model
        logging.debug("[MOCK] PEFT related functions patched")

        # Trainer instance and class mocks
        mock_trainer_instance = MagicMock()
        mock_trainer_class.return_value = mock_trainer_instance
        # Setup get_last_checkpoint to simulate no checkpoint found
        mock_get_ckpt.return_value = None

        # Trainer methods push_to_hub, save_model patched
        mock_trainer_instance.push_to_hub.return_value = None
        mock_trainer_instance.save_model.return_value = None
        mock_create_repo.return_value = None
        logging.debug("[MOCK] Trainer and HfApi mock methods patched")

        # Prepare CLI arguments for the main_function
        cli_args = [
            '--torch_dataset_url', 'fake/dataset',
            '--model_id', 'fake/model',
            '--project', 'testproject',
            '--user_id', 'testuser',
            '--hf_token', 'dummy_token'
        ]

        # Execute main_function
        ret_code = main_function(cli_args)

        # Assertions to verify flow correctness
        self.assertEqual(ret_code, 0)
        mock_load_ds.assert_any_call('fake/dataset', token='dummy_token', split='train')
        mock_load_ds.assert_any_call('fake/dataset', token='dummy_token', split='validation')
        mock_load_ds.assert_any_call('fake/dataset', token='dummy_token', split='test')
        mock_model_pretrained.assert_called_once_with(
            'fake/model',
            quantization_config=unittest.mock.ANY,
            device_map='auto',
            trust_remote_code=True
        )
        mock_tokenizer_pretrained.assert_called_once_with('fake/model', trust_remote_code=True)
        mock_lora_config.assert_called_once()
        mock_prepare_kbit.assert_called_once_with(mock_model)
        mock_get_peft.assert_called_once_with(mock_model, unittest.mock.ANY)
        mock_trainer_class.assert_called_once()
        mock_save_model.assert_called_once()
        mock_push_to_hub.assert_called_once()
        mock_create_repo.assert_called_once_with(repo_id='testuser/fake-model-testproject', exist_ok=True)

        logging.info("[TEST] End-to-end integration test passed")

if __name__ == "__main__":
    unittest.main()



