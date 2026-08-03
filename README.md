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

### Results: Qwen 2.5 3B

| Metric | Base Model | Fine‑tuned | Improvement |
| :--- | :--- | :--- | :--- |
| ROUGE‑1 | 0.3472 | 0.3591 | **+3.4%** |
| ROUGE‑2 | 0.1929 | 0.2025 | **+5.0%** |
| ROUGE‑L | 0.2515 | 0.2602 | **+3.5%** |

> **Note:** The 3B base model already exhibits strong medical knowledge, hence the improvements are more modest yet consistent across all metrics. In contrast, the 1.5B model shows larger relative gains because its base performance is lower, demonstrating that fine‑tuning is particularly beneficial for smaller models.

## How to Reproduce

1. Clone this repository.
2. Load the base model and fine‑tuned adapter (provided in the release assets).
3. Run evaluation on the MedQuad test set using the provided evaluation script.

## Conclusion

Fine‑tuning Qwen 2.5 with LoRA consistently improves ROUGE scores over the base models. The 1.5B variant shows the largest relative gains, while the 3B model benefits from a stronger starting point but still yields measurable improvements.
