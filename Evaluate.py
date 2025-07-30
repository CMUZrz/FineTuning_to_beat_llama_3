import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from evaluate import load as load_metric
import wandb

# --- Batched Retrieval for In-Context Learning ---
def retrieve_top_k(inputs, embeddings, outputs, model_GTR, util, query_inputs, top_k=5):
    """Retrieve top-k similar patient examples for given queries using embedding similarity."""
    # query_inputs: list of str (batch)
    query_embeddings = model_GTR.encode(query_inputs, convert_to_tensor=True)
    # Compute pairwise cosine similarity matrix [batch, dataset]
    similarities = util.cos_sim(query_embeddings, embeddings)  # shape [batch, N]
    all_retrieved = []
    for sim_vector in similarities:
        top_indices = torch.topk(sim_vector, k=top_k).indices.tolist()
        retrieved = [
            {
                "input": inputs[idx],      # generalized for key "input"
                "label": outputs[idx],
                "score": sim_vector[idx].item()
            }
            for idx in top_indices
        ]
        all_retrieved.append(retrieved)
    return all_retrieved  # List of length = len(query_inputs), each a list[dict]

def icl_prompts_batch(inputs, dataset_inputs, dataset_outputs, model_GTR, util, top_k=5):
    """Build ICL context strings for a batch of inputs by retrieving top_k for each."""
    retrieved_batches = retrieve_top_k(dataset_inputs, model_GTR.embeddings, dataset_outputs, model_GTR, util, inputs, top_k=top_k)
    prompt_contexts = []
    for input_clause, retrieved in zip(inputs, retrieved_batches):
        context = ""
        for r in retrieved:
            context += f"Input: {r['input']}\nOutput: {r['label']}\n\n"
        prompt_contexts.append(context)
    return prompt_contexts  # List of context strings per input

def main_function(cli_args):
    parser = argparse.ArgumentParser(description='Evaluate a tuned model via unbiased LLM with optional ICL retrieval.')
    parser.add_argument('--model_id', type=str, default="mistralai/Mistral-7B-Instruct-v0.1",
                        help="Model name or path for text generation (e.g., mistralai/Mistral-7B-Instruct-v0.1)")
    parser.add_argument('--test_dataset', type=str, required=True,
                        help="Path or HuggingFace repo for the test eval split (e.g., lavita/ChatDoctor-HealthCareMagic-100k)")
    parser.add_argument('--use_icl', action='store_true',
                        help="Enable in-context learning retrieval for prompt construction.")
    parser.add_argument('--icl_input_field', type=str, default="input",
                        help="Which field to use for retrieval/prompt in ICL mode (default: input)")
    parser.add_argument('--icl_label_field', type=str, default="output",
                        help="Which field to use for label/reference in ICL mode (default: output)")
    parser.add_argument('--judge_model', type=str, default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
                        help="Model name for BERTScore metric judge.")
    parser.add_argument('--project', default="medical-qa-evaluation", help="wandb project name")
    parser.add_argument('--hf_token', default=None, help="Optional: HF_TOKEN (or from env)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for text generation and evaluation.")
    parser.add_argument('--icl_top_k', type=int, default=5, help="Top-k retrieval for ICL context (if enabled)")
    args = parser.parse_args(cli_args)

    HF_TOKEN = args.hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if HF_TOKEN is None:
        raise ValueError("No Hugging Face token provided for dataset or model.")

    # --- Dataset Loading ---
    test_dataset = load_dataset(args.test_dataset, split="test", token=HF_TOKEN)
    # For ICL: You may want retrieval on train or whole dataset (here, we use test_dataset by default)
    icl_inputs = [ex[args.icl_input_field] for ex in test_dataset]
    icl_labels = [ex[args.icl_label_field] for ex in test_dataset]

    # --- wandb Setup ---
    wandb.init(
        project=args.project,
        config={
            "model": args.model_id,
            "eval_metric": "BERTScore",
            "test_set_size": len(test_dataset),
            "use_icl": args.use_icl,
            "icl_top_k": args.icl_top_k,
            "judge_model": args.judge_model
        }
    )

    # --- Model & Tokenizer Loading ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        temperature=0.0,
        top_k=50,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id
    )

    bertscore = load_metric("bertscore")

    # --- Optional: ICL retrieval model setup (if use_icl) ---
    if args.use_icl:
        from sentence_transformers import SentenceTransformer, util
        # Load or compute retrieval embeddings
        model_GTR = SentenceTransformer('sentence-transformers/gtr-t5-base')
        dataset_embeddings = model_GTR.encode(icl_inputs, convert_to_tensor=True, show_progress_bar=True)
        # Attach embeddings to model_GTR for convenience
        model_GTR.embeddings = dataset_embeddings  # Nonstandard, but fine for this script
    else:
        model_GTR = None

    # --- Batched Prediction Generation and Evaluation ---
    batch_size = args.batch_size
    all_predictions = []
    all_references = []
    test_f1, test_recall, test_precision = [], [], []

    num_examples = len(test_dataset)
    print("[INFO] Starting batch evaluation...")
    for i in tqdm(range(0, num_examples, batch_size), desc="Batch Evaluating"):
        batch_examples = test_dataset[i: i+batch_size]
        batch_inputs = [ex[args.icl_input_field] for ex in batch_examples]
        batch_instructions = [ex.get("instruction", "") for ex in batch_examples]
        batch_outputs = [ex[args.icl_label_field] for ex in batch_examples]

        # --- ICL context generation per batch (if enabled) ---
        if args.use_icl:
            batch_icl_contexts = icl_prompts_batch(
                batch_inputs,
                icl_inputs,
                icl_labels,
                model_GTR,
                util,
                top_k=args.icl_top_k
            )
            batch_prompts = [
                icl_ctx + f"Instruction: {ins}\nInput: {inp}\nResponse:"
                for icl_ctx, ins, inp in zip(batch_icl_contexts, batch_instructions, batch_inputs)
            ]
        else:
            batch_prompts = [
                f"Instruction: {ins}\nInput: {inp}\nResponse:"
                for ins, inp in zip(batch_instructions, batch_inputs)
            ]

        # Calculate batch target lengths (maximum reference length)
        batch_target_lengths = [
            len(tokenizer(out)["input_ids"]) for out in batch_outputs
        ]
        max_new_tokens = min(max(batch_target_lengths), 256)

        batch_outputs_gen = generator(
            batch_prompts,
            max_new_tokens=max_new_tokens
        )  # Returns list[dict]

        batch_generated_texts = [out["generated_text"] for out in batch_outputs_gen]

        # Truncate predictions and references to 512 tokens/characters for scoring
        predictions_trunc = [s[:512] for s in batch_generated_texts]
        references_trunc = [ref[:512] for ref in batch_outputs]

        # Accumulate for overall metrics
        all_predictions.extend(predictions_trunc)
        all_references.extend(references_trunc)

        # --- BERTScore this batch ---
        out = bertscore.compute(
            predictions=predictions_trunc,
            references=references_trunc,
            model_type=args.judge_model,
            num_layers=12,
            batch_size=32,
            lang="en",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        test_precision.extend(out["precision"])
        test_recall.extend(out["recall"])
        test_f1.extend(out["f1"])

        # Log batch-wise metrics to wandb
        wandb.log({
            "batch/precision": float(np.mean(out["precision"])),
            "batch/recall": float(np.mean(out["recall"])),
            "batch/f1": float(np.mean(out["f1"])),
            "batch_start_idx": i
        })

    # --- Final aggregate metrics ---
    mean_precision = float(np.mean(test_precision))
    mean_recall = float(np.mean(test_recall))
    mean_f1 = float(np.mean(test_f1))

    wandb.log({
        "eval/precision": mean_precision,
        "eval/recall": mean_recall,
        "eval/f1": mean_f1
    })
    print(f'[RESULT] Model averages - f1: {mean_f1:.4f} | recall: {mean_recall:.4f} | precision: {mean_precision:.4f}')

    wandb.finish()
    return 0

if __name__ == "__main__":
    import sys
    main_function(sys.argv[1:])
