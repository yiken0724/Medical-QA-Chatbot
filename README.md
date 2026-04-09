# Fine-Tuned Flan-T5 with Retrieval-Augmented Generation for Medical Q&A

A medical question-answering chatbot built by fine-tuning Google's **Flan-T5-Base** on a large doctor-patient dialogue corpus, augmented with **Retrieval-Augmented Generation (RAG)** grounded in authoritative medical sources. The project investigates how LoRA fine-tuning and external retrieval each contribute to improving the factual accuracy and quality of AI-generated medical responses.

---

## Motivation

With 29% of people now using LLM chatbots to seek medical information (Yun & Bickmore, 2025), the risk of hallucinated or misleading health advice is a real safety concern. This project explores a practical mitigation approach: domain-specific fine-tuning combined with retrieval grounding from trusted medical websites to reduce hallucinations while maintaining a helpful, conversational tone.

---

## Repository Structure

```
.
├── preprocessing.ipynb                          # Data cleaning, filtering, and formatting pipeline
├── train.py                                     # LoRA fine-tuning script (runs on GPU cluster)
├── lora-tune-v9.sh                              # SLURM job submission script for SMU SCIS GPU cluster
├── evaluation.ipynb                             # Quantitative (ROUGE-L, BERTScore) and qualitative evaluation
├── inference-final.ipynb                        # Inference and qualitative evaluation notebook
├── medical_qa_t5base_sample_evaluated.csv       # Baseline Flan-T5 sample outputs
├── medical_qa_t5base+lora_sample_evaluated.csv  # LoRA fine-tuned model sample outputs
├── medical_qa_t5base+rag_sample_evaluated.csv   # Base model + RAG sample outputs
├── medical_qa_t5base+lora+rag_sample_evaluated.csv  # LoRA + RAG sample outputs
├── requirements.txt
├── pyproject.toml
└── cred.env                                     # API credentials (not committed — see Setup)
```

---

## Dataset

**Source:** [`ruslanmv/ai-medical-chatbot`](https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot) on HuggingFace

A corpus of ~257,000 real doctor-patient exchanges with three fields:

- `Description` — the patient's question
- `Patient` — additional background and context from the patient
- `Doctor` — the doctor's response (used as the training target)

### Preprocessing

Run `preprocessing.ipynb` to clean and prepare the dataset. The pipeline covers:

1. **Data Cleaning** — removes ~10,300 duplicate entries, strips non-ASCII characters, normalises to lowercase
2. **Length Filtering** — removes entries with questions under 10 characters, answers under 20 characters, or any field over 1,000 characters (~24,000 rows removed, leaving ~222,000)
3. **Conversational Text Processing** — strips generic openers ("Hi Doctor,", "Many thanks") and sign-offs ("Regards, Dr. John") so the model focuses on substantive medical content
4. **Formatting & Splitting** — formats into `Question: ... Answer:` instruction prompts and splits 80/10/10 into train, validation, and test sets

---

## Model Architecture

### Base Model: Flan-T5-Base (248M parameters)

Google's Flan-T5 was chosen over base T5 because it has been instruction-tuned on a diverse set of NLP tasks, producing more coherent outputs even before fine-tuning. The base variant was selected to fit within the compute constraints of the SMU SCIS GPU cluster.

### Fine-Tuning: LoRA (Low-Rank Adaptation)

LoRA inserts small trainable adapter layers while keeping the original model weights frozen, reducing the number of trainable parameters to ~9 million (under 4% of the full model). Training took 2.5–4 hours per run on RTX 3090 GPUs.

The final configuration (v9) targeted both **attention components** (encoder and decoder self-attention and cross-attention) and **feed-forward network weights**, with the **Adafactor** optimiser and its adaptive learning rate enabled.

Key hyperparameters:

```
r=16, lora_alpha=32, lora_dropout=0.1
target_modules: encoder/decoder attention + feed-forward network weights
optim="adafactor" with relative_step=True, scale_parameter=True, warmup_init=True
num_train_epochs=2
per_device_train_batch_size=8, gradient_accumulation_steps=4
generation_max_length=256, generation_num_beams=4
```

### Retrieval-Augmented Generation (RAG)

At inference time, the user's query is issued to the **Google Custom Search API**, constrained to the following trusted medical sources:

- [www.cda.gov.sg](https://www.cda.gov.sg) — Communicable Diseases Agency of Singapore
- [www.moh.gov.sg](https://www.moh.gov.sg) — Ministry of Health Singapore
- [medlineplus.gov](https://medlineplus.gov) — U.S. National Library of Medicine
- [www.who.int](https://www.who.int) — World Health Organisation

The top retrieved text snippet is cleaned and appended to the input query before being passed to the model, grounding the response in verified medical information and providing a source citation in the output.

---

## Results

### Quantitative Evaluation (ROUGE-L & BERTScore)

| Configuration              | ROUGE-L | BERTScore |
|----------------------------|---------|-----------|
| Flan-T5-Base (baseline)    | 0.0773  | 0.7974    |
| + LoRA v5                  | 0.0812  | 0.7992    |
| + LoRA v8 (Adafactor ALR)  | 0.1165  | 0.8172    |
| + LoRA v9 (+ FFN tuning)   | 0.1186  | 0.8175    |
| + RAG only                 | 0.0964  | 0.8127    |
| **+ LoRA v9 + RAG**        | **0.1704** | **0.8495** |

The best performance was achieved by combining LoRA fine-tuning with RAG, validating that domain adaptation and retrieval grounding are complementary.

### Qualitative Summary

- **Baseline Flan-T5**: Paraphrased patient input rather than answering; collapsed into hallucinations (e.g. outputting sequences of Roman numerals) on unfamiliar queries
- **LoRA only**: Adopted an empathetic "doctor" persona with fluent, confident responses — but prone to hallucinations with no citations to verify claims
- **RAG only**: Factually precise with source citations, but dry and clinical in tone; lacked the empathetic framing expected in a patient-facing tool
- **LoRA + RAG (best)**: Combined the empathetic tone of LoRA with the factual grounding of RAG, producing the highest-scoring responses both quantitatively and qualitatively

---

## Setup

### 1. Create a virtual environment and install dependencies

This project uses `uv` by default.

- With `uv`:
```bash
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv sync
uv lock
```

- With standard `venv` and pip:
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Configure API credentials

Create a `cred.env` file in the project root with your Google Custom Search API credentials:

```
GOOGLE_API_KEY=your_api_key_here
GOOGLE_CSE_ID=your_custom_search_engine_id_here
```

> **Note:** `cred.env` is not committed to the repository. Keep your credentials private.

---

## Running the Project

### Preprocessing
Run `preprocessing.ipynb` to download the dataset and produce the formatted train/validation/test splits.

### Training
Submit the training job on the SCIS GPU cluster:
```bash
sbatch lora-tune-v9.sh
```
Or run directly:
```bash
python train.py
```
Ensure the `medical_qa` processed dataset is available in the project root and that GPU access is configured before training.

### Evaluation
Run `evaluation.ipynb` to compute ROUGE-L and BERTScore metrics across all model configurations. Run `inference-final.ipynb` for qualitative sample generation and comparison.

---

## References

- Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Longpre et al. (2023). [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/abs/2301.13688)
- Shazeer & Stern (2018). [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)
- Yun & Bickmore (2024). [Online Health Information Seeking in the Era of LLMs](https://doi.org/10.2196/68560)
- Clark & Bailey (2024). [Chatbots in Health Care](https://www.ncbi.nlm.nih.gov/books/NBK602381/)
