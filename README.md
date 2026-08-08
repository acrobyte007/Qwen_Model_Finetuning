This repository contains fine‑tuned versions of **Qwen 2.5** (1.5B and 3B) on the **MedQuad** medical question‑answer dataset. Both models are adapted using **LoRA** (rank 16, all linear layers) and evaluated against their base counterparts on a held‑out test set.

## Dataset: MedQuad

MedQuad is a collection of medical question‑answer pairs derived from authoritative sources (e.g., NIH, medical textbooks). It covers a wide range of diseases, treatments, and clinical concepts. The dataset is formatted as:

- **Question**: A medical question (e.g., *What are the symptoms of glaucoma?*)
- **Answer**: A detailed, evidence‑based response.

We used the version available at:  
`/kaggle/input/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research/medquad.csv`

### Train‑Test Split

- **Total samples**: ~16,407
- **Training set**: 90% (~14,766 samples)
- **Test set**: 10% (~1,641 samples)

The split was performed **randomly** with a fixed seed for reproducibility.

## Models

| Model | Base | Fine‑tuned |
| :--- | :--- | :--- |
| Qwen 2.5 1.5B | `Qwen/Qwen2.5-1.5B-Instruct` | LoRA adapter (rank 16, all linear layers) |
| Qwen 2.5 3B   | `Qwen/Qwen2.5-3B-Instruct`   | LoRA adapter (rank 16, all linear layers) |

## Evaluation Metrics

We report **ROUGE‑1, ROUGE‑2, and ROUGE‑L** (F1 scores) on the test split. ROUGE measures n‑gram overlap between generated answers and ground‑truth references.

### Results: Qwen 2.5 1.5B

| Metric | Base Model | Fine‑tuned | Improvement |
| :--- | :--- | :--- | :--- |
| ROUGE‑1 | 0.2790 | 0.4245 | **+52.2%** |
| ROUGE‑2 | 0.0609 | 0.2350 | **+285.9%** |
| ROUGE‑L | 0.1360 | 0.3001 | **+120.7%** |

### Qwen2.5‑3B

| Metric  | Base Model | Fine-Tuned | Improvement |
| ------- | ---------: | ---------: | ----------: |
| ROUGE-1 |     0.1904 | **0.3504** |  **+84.1%** |
| ROUGE-2 |     0.0455 | **0.1982** | **+335.5%** |
| ROUGE-L |     0.1160 | **0.2543** | **+119.3%** |

> **Note:** The 3B base model already exhibits strong medical knowledge, hence the improvements are more modest yet consistent across all metrics. In contrast, the 1.5B model shows larger relative gains because its base performance is lower, demonstrating that fine‑tuning is particularly beneficial for smaller models.
# Qualitative Examples (from 1.5B model)

### Example 1

**Question**

> What are the treatments for Pulmonary alveolar proteinosis acquired?

**Observation**

The fine-tuned model generated a clinically relevant treatment-oriented response that captured the main therapeutic concepts, although it differed from the reference wording.

**ROUGE-1:** 0.2520

---

### Example 2

**Question**

> What is primary ciliary dyskinesia?

**Observation**

The model correctly described the disease as a respiratory disorder and identified key symptoms such as recurrent respiratory infections and breathing difficulties. Although some secondary details were omitted, the generated response remained medically relevant.

**ROUGE-1:** 0.3968

---

### Example 3

**Question**

> What genetic changes are related to biotin-thiamine-responsive basal ganglia disease?

**Observation**

The model incorrectly generated a different gene name instead of **SLC19A3**, demonstrating a factual hallucination despite achieving a relatively high ROUGE score.

**ROUGE-1:** 0.4353

This example highlights that lexical overlap metrics alone cannot fully measure factual correctness in medical question answering.

---

# Error Analysis

Although LoRA significantly improved ROUGE scores, qualitative analysis revealed several limitations:

* Some responses paraphrased the reference answer, resulting in lower ROUGE despite being semantically correct.
* Certain answers omitted secondary clinical details.
* A small number of predictions hallucinated biomedical entities, such as incorrect gene names.

These observations indicate that ROUGE should be complemented with semantic and factuality-based evaluation metrics for medical applications.

## How to Reproduce

1. Clone this repository.
2. Load the base model and fine‑tuned adapter (provided in the release assets).
3. Run evaluation on the MedQuad test set using the provided evaluation script.
4. Downloade the LoRA adapter folder in zip file from kaggle and unzip it in local
5. Download the base Qwen-2.5 1.5B model from huggingface and unzip it in local
6. Merge the LoRA adapter with Qwen-2.5 1.5B model by running the merge_model.py script in terminal
7. Teste local reliability / latency by running the benchmark.py script in terminal
8. Create FastAPI inference

## Conclusion

Fine‑tuning Qwen 2.5 with LoRA consistently improves ROUGE scores over the base models. The 1.5B variant shows the largest relative gains, while the 3B model benefits from a stronger starting point but still yields measurable improvements.
