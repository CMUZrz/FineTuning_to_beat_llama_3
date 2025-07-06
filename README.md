

# Final Executive Summary: Can Smaller, Efficiently Fine-Tuned LLMs Outperform Larger Models?

**Authors:** Dan Jung, Dhruva Byrapatna, Zachary Zdobinski
*A project for the 10-423/623 Generative AI Course.*

## 🌟 Highlights \& Major Results

This project demonstrates that smaller, efficiently fine-tuned Large Language Models (LLMs) can outperform larger, more computationally expensive counterparts on specialized tasks.

* **Small Model Outperforms Large Model:** Our fine-tuned **Mistral-7B** model achieved higher accuracy and better performance metrics than the much larger **Llama 3 70B** model on two out of three domain-specific datasets (GSM8K and BeerAdvocate).
* **QLoRA is Highly Effective:** Using Quantized Low-Rank Adaptation (QLoRA) for fine-tuning consistently provided significant performance improvements over the baseline Mistral-7B model across all datasets.
* **Combined Approach for Complex Tasks:** For complex, text-heavy tasks like those in the Healthcare and Math (GSM8K) datasets, a combined approach of training with both In-Context Learning (ICL) examples and QLoRA yielded the best results.
* **Method Effectiveness is Task-Dependent:** The effectiveness of different fine-tuning methods varies by dataset. While combining QLoRA and Dr.ICL was powerful for some tasks, using QLoRA alone was the most effective strategy for the BeerAdvocate dataset, which features many numeric categories.


## 📖 Table of Contents

* [Project Overview](#-project-overview)
* [Methods](#-methods)
* [Datasets](#-datasets)
* [Experimental Results](#-experimental-results)
* [Conclusion](#-conclusion)
* [Future Work](#-future-work)
* [Code Overview](#-code-overview)
* [Key Related Work](#-key-related-work)


## Project Overview

Recent advances in generative AI have been driven by massive LLMs with billions of parameters. However, training and fine-tuning these models is prohibitively expensive for many. This project investigates whether smaller LLMs, using parameter-efficient fine-tuning (PEFT) methods, can match or exceed the performance of larger models on specialized tasks.

Our objective is to show that a smaller model, when trained on targeted datasets with efficient methods like QLoRA and Demonstration-Retrieved In-Context Learning (Dr.ICL), can achieve higher accuracy, providing a viable alternative for specialized AI applications.

## 🛠️ Methods

We evaluated the performance of a small model (Mistral-7B-Instruct) against a large one (Llama 3 70B) using several enhancement techniques.

### Baseline

The baseline performance was established using the untuned Mistral-7B and Llama 3 70B models on each dataset in a zero-shot setting.

### Main Methods

1. **Parameter-Efficient Fine-Tuning (QLoRA):** This was our core technique for adapting the smaller model. QLoRA significantly reduces memory requirements and computational cost by combining:
    * **4-bit NormalFloat Quantization:** Compresses model weights to a 4-bit format.
    * **Double Quantization:** A second quantization step for even greater memory savings.
    * **Paged Optimizers:** Prevents out-of-memory errors by managing memory efficiently between the CPU and GPU.
    * **Low-Rank Adaptation (LoRA):** Freezes most model parameters and only trains a small set of new, "adapter" weights.
2. **Demonstration-Retrieval for In-Context Learning (Dr.ICL):** This method improves model performance at inference time without changing model weights. For a given question, it retrieves semantically similar examples from a dataset and adds them to the prompt, giving the model a relevant demonstration of the task.

### Integrated Pipeline

We followed a systematic process:

1. **Establish Baselines:** Evaluate untuned models.
2. **Fine-Tuning:** Apply QLoRA to the Mistral-7B model for each dataset.
3. **In-Context Learning:** Apply Dr.ICL to the untuned model.
4. **Combined Methods:** Integrate QLoRA and Dr.ICL to assess their joint effect.
5. **Comparative Analysis:** Compare all results to identify the most effective strategies.

## 📊 Datasets

We used three distinct datasets to test performance on specialized tasks:

* **Healthcare Magic Dataset:** A high-quality dataset of over 100,000 anonymized patient-doctor conversations. We used a subset for fine-tuning and evaluation, measuring performance with BERTScore against ground-truth answers.
* **BeerAdvocate Dataset:** A large collection of beer reviews. The task was to classify beers into official Beer Judge Certification Program (BJCP) styles based on user descriptions of aroma, flavor, and appearance.
* **GSM8K Dataset:** A collection of 8,500 grade-school math word problems. The task was to reason through the problem and provide a final numerical answer. Performance was measured by the accuracy of the final answer.


## 📈 Experimental Results

Our experiments show that a fine-tuned small model can consistently outperform a much larger model. The combination of methods that worked best depended on the specific task.

The full results are summarized in the table below:


| Model | Dataset | Accuracy | Precision | Recall | F1 Score |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **Mistral Base** | Healthcare | - | .87 | .79 | .83 |
|  | Beer | - | 0.71 | 0.72 | 0.72 |
|  | Math | 0.11 | - | - | - |
| **Base+ICL** | Healthcare | - | .85 | .86 | .86 |
|  | Beer | - | 0.53 | 0.67 | 0.59 |
|  | Math | 0.37 | - | - | - |
| **Base+QLoRA** | Healthcare | - | .86 | .87 | .87 |
|  | Beer | - | **0.74** | **0.83** | **0.79** |
|  | Math | 0.35 | - | - | - |
| **Base+ICL+QLoRA** | Healthcare | - | .85 | .86 | .86 |
|  | Beer | - | 0.61 | 0.78 | 0.69 |
|  | Math | 0.31 | - | - | - |
| **Base ICL train+QLoRA** | Healthcare | - | **.89** | **.90** | **.89** |
|  | Beer | - | 0.61 | 0.78 | 0.69 |
|  | Math | **0.42** | - | - | - |
| **Llama 3.3 70B** | Healthcare | - | .87 | .88 | .87 |
|  | Beer | - | 0.60 | 0.66 | 0.63 |
|  | Math | 0.33 | - | - | - |

## ✅ Conclusion

Our research confirms that **parameter-efficient fine-tuning (QLoRA) enables smaller LLMs to achieve or surpass the performance of much larger models** on specialized, domain-specific tasks. The fine-tuned Mistral-7B model outperformed the Llama 3 70B model on the GSM8K and BeerAdvocate datasets.

We also found that the effectiveness of demonstration-retrieval in-context learning (Dr.ICL) was highly dependent on the dataset. While it provided a boost for some tasks, particularly when integrated into the training process, it was not universally beneficial. This highlights the importance of selecting the right fine-tuning strategy for a given application.

## 🚀 Future Work

* **Refine Data Quality and Prompt Engineering:** Improving training data and prompt design could further enhance model performance, especially when combining QLoRA and Dr.ICL.
* **Evaluate on New Domains:** Our approach could be tested on other technical datasets, such as those from the nuclear industry or other scientific fields, to generalize our findings.


## 💻 Code Overview

The implementation for this project was handled in several parts:

* **Mistral-7B Evaluations:** Used HuggingFace libraries to run the base Mistral-7B model and evaluate its outputs. A key challenge was creating automated evaluation scripts for each dataset's unique response format.
* **Llama 3 Evaluations:** As the Llama 3 70B model was too large to run locally, we used API services to get performance benchmarks.
* **Dr.ICL Implementation:** We built a pipeline to retrieve the top-k most similar examples from a dataset based on cosine similarity of embeddings. These examples were then prepended to the prompt at inference time.
* **QLoRA Implementation:** We used HuggingFace's `peft` and `transformers` libraries to implement QLoRA. This involved tokenizing the data, applying a mask to ensure the model only trained on the "answer" part of the examples, and configuring the model for 4-bit training.


## 📚 Key Related Work

Our work is built upon the insights from several foundational papers in the field of efficient model training and in-context learning:

* **AnyTaskTune: Advanced Domain-Specific Solutions through Task-Fine-Tuning** (Cui et al., 2024): Introduced a methodology for optimizing LLMs for domain-specific tasks by creating precise, tailored datasets.
* **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023): The core technical paper for the QLoRA method, which formed the backbone of our fine-tuning approach.
* **Dr.ICL: Demonstration-Retrieved In-Context Learning** (Luo et al., 2023): Proposed the method of retrieving demonstrations to boost in-context learning, which we implemented and tested.
* **Textbooks Are All You Need** (Gunasekar et al., 2023): Showcased the power of training on high-quality, "textbook" style data to achieve strong performance with smaller models.
* **Large Dual Encoders Are Generalizable Retrievers** (Ni et al., 2021): This work supported our choice of the GTR-T5 model as a dense retriever for our Dr.ICL implementation, showing its effectiveness in zero-shot retrieval.

