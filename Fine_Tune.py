import argparse
import gc
from datetime import datetime
import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    set_verbosity_error,
    trainer_utils
)
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

from huggingface_hub import HfApi

def main_function(cli_args):
    # ------------------------------
    # Argument Parsing and Setup
    # ------------------------------
    parser = argparse.ArgumentParser(description='Train a model via QLoRA with specified options.')

    parser.add_argument(
        '--torch_dataset_url',
        type=str,
        default="lavita/ChatDoctor-HealthCareMagic-100k",
        help="HuggingFace Hub or local directory identifier for the dataset to load."
    )
    parser.add_argument(
        '--model_id',
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
        help="Model name or path (e.g., mistralai/Mistral-7B-Instruct-v0.1)"
    )
    parser.add_argument(
        '--project',
        type=str,
        default="qlora-run",
        help="Project name suffix for experiment tracking and output folders."
    )
    parser.add_argument(
        '--user_id',
        type=str,
        default="YOUR_HF_USER",  # <-- CHANGE THIS TO YOUR HF USERNAME
        help="Your Hugging Face username for uploading/checkpointing."
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help="(Optional) HF Hub token. If not set, reads from HUGGINGFACE_HUB_TOKEN env variable."
    )

    args = parser.parse_args(cli_args)

    HF_TOKEN = args.hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if HF_TOKEN is None:
        raise ValueError("No Hugging Face token provided for dataset or pushing to hub.")

    dataset_path = args.torch_dataset_url
    model_id = args.model_id
    project = args.project
    user_id = args.user_id

    # ------------------------------
    # Load and Prepare Datasets
    # ------------------------------

    # These can be customized if you want to use different splits/datasets
    print("[INFO] Downloading & loading datasets...")
    masked_train = load_dataset(dataset_path, token=HF_TOKEN, split="train")
    masked_val = load_dataset(dataset_path, token=HF_TOKEN, split="validation")
    masked_test = load_dataset(dataset_path, token=HF_TOKEN, split="test")

    # (If you want different splits or full datasets separate from tokenized/processed, load them here.)

    # ------------------------------
    # Model Setup and Quantization
    # ------------------------------
    set_verbosity_error()
    gc.collect()
    device = 0 if torch.cuda.is_available() else -1
    print(f"[INFO] Using device: {'cuda' if device == 0 else 'cpu'}")
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[INFO] Loading tokenizer for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading quantized model '{model_id}'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # ------------------------------
    # LoRA Configuration and Application
    # ------------------------------
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print("[INFO] Wrapping model with LoRA adapters...")
    model = get_peft_model(model, lora_config)

    # Assert trainable params > 0
    model.print_trainable_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ASSERT] Trainable parameters: {trainable_params}")
    assert trainable_params > 0, "No trainable parameters found with LoRA applied!"

    # ------------------------------
    # Experiment Output & Tracking Setup
    # ------------------------------
    base_model_name = model_id.split("/")[-1]
    run_name = f"{base_model_name}-{project}"
    output_dir = os.path.join(".", run_name)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------
    # Training Arguments and Trainer
    # ------------------------------
    print("[INFO] Setting up Trainer and training arguments...")
    trainer = Trainer(
        model=model,
        train_dataset=masked_train,
        eval_dataset=masked_val,
        args=TrainingArguments(
            output_dir=output_dir,
            warmup_steps=5,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            max_steps=1000,
            learning_rate=2.5e-5, # 10x smaller than the mistral learning rate change for the model 
            logging_steps=50,
            bf16=False,
            optim="paged_adamw_8bit",
            logging_dir="./logs",
            save_strategy="steps",
            save_steps=50,
            evaluation_strategy="steps",
            eval_steps=50,
            do_eval=True,
            report_to="wandb",
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # silence warnings; enable for inference

    # ------------------------------
    # Training and Checkpointing
    # ------------------------------
    last_ckpt = trainer_utils.get_last_checkpoint(output_dir)
    if last_ckpt is not None:
        print(f"[INFO] Resuming training from checkpoint in {output_dir}")
        trainer.train(resume_from_checkpoint=output_dir)
    else:
        print(f"[INFO] Starting fresh training, checkpoints will be saved to {output_dir}")
        trainer.train()

    # ------------------------------
    # Saving Model and Tokenizer
    # ------------------------------
    print("[INFO] Saving trained model and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ------------------------------
    # Uploading to Hugging Face Hub
    # ------------------------------
    dataset_name = os.path.basename(dataset_path.rstrip('/'))
    api = HfApi()
    print(f"[INFO] Creating repo USER_ID/{run_name} on Hugging Face Hub (if not already exists)...")
    api.create_repo(repo_id=f"{user_id}/{run_name}", exist_ok=True)

    print(f"[INFO] Pushing model and config to the hub in repo {user_id}/{run_name}...")
    trainer.push_to_hub(
        repo_id=f"{user_id}/{run_name}",
        commit_message=f"QLoRA fine-tuned on {dataset_name}"
    )

    print("[SUCCESS] Training complete and model pushed to hub.")
    return 0

if __name__ == "__main__":
    import sys
    main_function(sys.argv[1:])
