import argparse
from sentence_transformers import SentenceTransformer, util
import torch
from datasets import load_dataset
from transformers import AutoTokenizer # Assuming a tokenizer is needed
import os
# --- Argument Parsing ---
# Instantiate the parser

def main_function(cli_args):

    parser = argparse.ArgumentParser(description='Run pre-training with an optional ICL flag.')

    parser.add_argument('--icl', 
                        action='store_true', 
                        help='Enable In-Context Learning (ICL) pre-training (default: disabled).')

    # This argument is correctly configured.
    parser.add_argument('--torch_dataset_url', 
                        type=str, 
                        default="lavita/ChatDoctor-HealthCareMagic-100k",
                        help='Identifier for the dataset to load from Hugging Face Hub.')

    # CORRECTED: Use action='store_true'.
    # The default is False. Providing the --upload flag sets it to True.
    parser.add_argument('--upload', 
                        action='store_true', 
                        help='Enable optional uploads to the HF hub (default: disabled).')

    # CORRECTED: The help message has been fixed to be descriptive.
    parser.add_argument('--destination_torch_folder', 
                        type=str, 
                        default="hf_user_id/name",
                        help='Specify the Hugging Face Hub repository for uploads (e.g., "username/repo-name").')

    # Parse the arguments from the command line
    args = parser.parse_args(cli_args)

    # You can now use the arguments in your script
    enable_icl = args.icl
    enable_upload = args.upload
    dataset_url = args.torch_dataset_url
    destination_repo = args.destination_torch_folder


    # --- Dataset Loading and Preprocessing ---

    # Load original datasets
    # NOTE: You  need to be logged into Hugging Face Hub for this
    #OS.get

    # Optionally ensure the token is read from environment
    HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

    full_dataset = load_dataset(dataset_url, split="train")

    # Load and downsample to 4,000 total samples
    downsampled_dataset = full_dataset.shuffle(seed=42).select(range(4000))

    # Split into train (80%), temp (20%)
    train_temp_split = downsampled_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_temp_split['train']  # 3,200 samples
    temp_dataset = train_temp_split['test']    # 800 samples

    # Split temp into validation/test (50/50)
    val_test_split = temp_dataset.train_test_split(test_size=0.5, seed=42)
    val_dataset = val_test_split['train']      # 400 samples
    test_dataset = val_test_split['test']      # 400 samples


    # --- Tokenizer and Model Loading ---

    # Assuming a tokenizer is needed, as it was used in the original functions
    # Replace "bert-base-uncased" with your model of choice
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load GTR model for retrieval
    model_GTR = SentenceTransformer('sentence-transformers/gtr-t5-base')


    # --- Tokenization Functions ---

    # Function for standard masked language model training
    def tokenize_medqa_masked(examples):
        prompts = [
            f"instruction: {instr}, input: {input} please give a medical response answer: {answer}"
            for instr, input, answer in zip(examples["instruction"], examples["input"], examples["output"])
        ]
        tokenized = tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        labels = []
        for i, offsets in enumerate(tokenized["offset_mapping"]):
            answer_text = str(examples["output"][i])
            prompt_i = prompts[i]
            start_char = prompt_i.find(answer_text)
            end_char = start_char + len(answer_text)
            label_ids = [-100] * len(offsets)
            for j, (off_start, off_end) in enumerate(offsets):
                if off_start >= start_char and off_end <= end_char:
                    label_ids[j] = tokenized["input_ids"][i][j]
            labels.append(label_ids)
        tokenized["labels"] = labels
        tokenized.pop("offset_mapping")
        return tokenized

    # --- ICL-Related Setup and Functions ---

    # Create and save embeddings for retrieval
    dr_icl_pairs = [(ex["input"], ex["output"]) for ex in train_dataset]
    patient_inputs = [pair[0] for pair in dr_icl_pairs]
    doctor_outputs = [pair[1] for pair in dr_icl_pairs]

    patient_embeddings = model_GTR.encode(
        patient_inputs,
        convert_to_tensor=True,
        show_progress_bar=True
    )

    torch.save({
        'embeddings': patient_embeddings,
        'patient_inputs': patient_inputs,
        'doctor_outputs': doctor_outputs
    }, "medqa_gtr_embeddings.pt")

    # Load retrieval corpus
    retrieval_corpus = torch.load("medqa_gtr_embeddings.pt")
    corpus_embeddings = retrieval_corpus['embeddings']
    corpus_inputs = retrieval_corpus['patient_inputs']
    corpus_outputs = retrieval_corpus['doctor_outputs']

    def retrieve_top_k(query_input, top_k=3):
        query_embedding = model_GTR.encode(query_input, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_k_indices = torch.topk(similarities, k=top_k).indices.tolist()
        return [
            {"patient_input": corpus_inputs[idx], "label": corpus_outputs[idx], "score": similarities[idx].item()}
            for idx in top_k_indices
        ]

    # Function for ICL-based masked language model training
    def tokenize_medqa_masked_icl(examples):
        icl_contexts = [
            "\n\n".join([f"Input: {r['patient_input']}\nOutput: {r['label']}" for r in retrieve_top_k(input_text, top_k=3)])
            for input_text in examples["input"]
        ]
        prompts = []
        answer_positions = []
        for icl, instr, input_txt, answer in zip(icl_contexts, examples["instruction"], examples["input"], examples["output"]):
            question_part = f"instruction: {instr}, input: {input_txt} please give a medical response answer: "
            full_prompt = f"{icl}\n\n{question_part}{answer}"
            answer_start = len(icl) + len("\n\n") + len(question_part)
            answer_end = answer_start + len(answer)
            answer_positions.append((answer_start, answer_end))
            prompts.append(full_prompt)

        tokenized = tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )

        labels = []
        for i, offsets in enumerate(tokenized["offset_mapping"]):
            start_char, end_char = answer_positions[i]
            label_row = [
                tokenized["input_ids"][i][j] if (off_start >= start_char and off_end <= end_char) else -100
                for j, (off_start, off_end) in enumerate(offsets)
            ]
            labels.append(label_row)

        tokenized["labels"] = labels
        tokenized.pop("offset_mapping")
        return tokenized


    # --- Conditional Tokenization based on Command-Line Flag ---
    print(f"Running with ICL pre-training: {enable_icl}")

    if enable_icl:
        # This block runs if --icl is provided
        tokenized_train_ICL = train_dataset.map(tokenize_medqa_masked_icl, batched=True, remove_columns=train_dataset.column_names)
        print("Completed ICL-based tokenization.")
    else:
    # If standard tokenization, upload all three tokenized splits.
        tokenized_train_dataset = train_dataset.map(tokenize_medqa_masked, batched=True, remove_columns=train_dataset.column_names)
        tokenized_val_dataset = val_dataset.map(tokenize_medqa_masked, batched=True, remove_columns=val_dataset.column_names)
        tokenized_test_dataset = test_dataset.map(tokenize_medqa_masked, batched=True, remove_columns=test_dataset.column_names)
        tokenized_train_dataset.push_to_hub(destination_repo, config_name="train_tokenized", token=HF_TOKEN, private=True)
        tokenized_val_dataset.push_to_hub(destination_repo, config_name="validation_tokenized", token=HF_TOKEN, private=True)
        tokenized_test_dataset.push_to_hub(destination_repo, config_name="test_tokenized", token=HF_TOKEN, private=True)

    if enable_upload and HF_TOKEN:

        # Remove slashes and hyphens from repo names
        tokenized_train_dataset.push_to_hub(
            destination_repo,  # Valid format: namespace/repo_name
            token=HF_TOKEN,
            private=True,
            #repo_type="dataset"  # Explicitly specify dataset type
        )

        tokenized_val_dataset.push_to_hub(
            destination_repo,
            token=HF_TOKEN,
            private=True,
            #repo_type="dataset"
        )

        tokenized_test_dataset.push_to_hub(
            destination_repo,
            token=HF_TOKEN,
            private=True,
        )

        # Remove slashes and hyphens from repo names
        train_dataset.push_to_hub(
            destination_repo,  # Valid format: namespace/repo_name
            token=HF_TOKEN,
            private=True,
            #repo_type="dataset"  # Explicitly specify dataset type
        )

        val_dataset.push_to_hub(
            destination_repo,
            token=HF_TOKEN,
            private=True,
            #repo_type="dataset"
        )

        test_dataset.push_to_hub(
            destination_repo,
            token=HF_TOKEN,
            private=True,
        )
        
    return {
        "tokenized_train_dataset": tokenized_train_dataset,
        "tokenized_val_dataset": tokenized_val_dataset,
        "tokenized_test_dataset": tokenized_test_dataset,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }


if __name__ == "__main__":
    import sys
    main_function(sys.argv[1:])
    
    