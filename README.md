---

# 🇺🇦 ZNO Solver

This project solves Ukrainian ZNO (university entrance exam) questions using a fine-tuned LLM that runs **completely offline**.

It uses a "Teacher-Student" approach: we used a big model (Gemini 2.0) to generate reasoning data, and then taught a smaller model (Qwen 2.5 7B) to think like the teacher.

### Results

* **Base Model:** Qwen 2.5-7B-Instruct
* **Method:** LoRA Fine-Tuning + Logit Scoring
* **Target Score:** ~0.50+ accuracy (Beating the text-only baseline of 0.34)

---

## Pipeline

### Phase 1: The "Teacher" (Data Generation)

We took the raw ZNO training questions and asked **Gemini 2.0 Flash** to explain the answers.

* **Input:** Question + Options
* **Output:** Reasoning ("Chain of Thought") + Correct Letter
* **Result:** A dataset of ~20,000 examples where the AI explains *why* an answer is right.

### Phase 2: The "Student" (Fine-Tuning)

We trained a lightweight adapter (LoRA) for **Qwen 2.5 7B**.

* **Goal:** Teach Qwen to mimic Gemini's reasoning style.
* **Hardware:** T4 x2 or A100 GPU.
* **Outcome:** A small `adapter_model` file (~200MB) that makes Qwen smarter at ZNO.

### Phase 3: The "Exam" (Offline Inference)

The competition forbids internet access.

* **Challenge:** Installing libraries (`bitsandbytes`, `peft`) without `pip install`.
* **Solution:** We downloaded all Python wheels into a zip file (`offline_libs.zip`) and installed them locally.
* **Strategy:** We use **Logit Scoring**. Instead of asking the model to write text (which can be messy), we mathematically calculate the probability of letters **А, Б, В, Г, Д**. The letter with the highest % wins.

---

## Reproduction

### 1. Training (Phase 1 & 2)

Run the `train.py` (or notebook) to generate the adapter.

* *Input:* `zno.train.jsonl`
* *Output:* `zno-my-adapter/` folder

### 2. Prepare Offline Libraries (Phase 3)

Run the script to download dependencies:

```bash
pip download -d offline_libs --no-deps bitsandbytes peft accelerate transformers tokenizers safetensors sentencepiece
zip -r offline_libs.zip offline_libs

```

### 3. Kaggle Submission (Phase 4)

1. **Upload Datasets to Kaggle:**
* `zno-libs-final` (Your `offline_libs.zip`)
* `zno-my-adapter` (Your trained folder)


2. **Add Base Model:** Search and add `Qwen/Qwen2.5-7B-Instruct`.
3. **Run Inference:** Use the code in `submission.ipynb`.
* **Config:** `BATCH_SIZE = 1` (For stability on P100 GPU).
* **Prompt:** Uses the specific instruction *"Дай відповідь буквою-варіантом..."* from the [UNLP 2025 Paper](https://aclanthology.org/2025.unlp-1.2.pdf).



---

## 📂 File Structure

```text
├── train_pipeline/
│   ├── 1_generate_reasoning.py   # Ask Gemini to explain answers
│   └── 2_finetune_qwen.py        # Train Qwen with LoRA
├── offline_utils/
│   └── download_wheels.sh        # Script to get libraries for offline use
├── submission/
│   └── main_inference.ipynb      # The final code for Kaggle (Logit Scoring)
└── README.md

```

## 📜 Credits

Based on the methodology from **"Benchmarking Multimodal Models for Ukrainian Language Understanding"**.

* **Lecture/Paper:** Paniv et al., UNLP 2025.
* **Key Insight:** Logit scoring on Cyrillic tokens outperforms text generation for multiple-choice QA.
