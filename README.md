

### README.md

```markdown
# 🇺🇦 ZNO Solver: Teacher-Student Reasoning Approach

This project solves Ukrainian ZNO (university entrance exam) questions using a fine-tuned LLM that runs **completely offline** within Kaggle's restricted environment.

It uses a "Teacher-Student" approach: we used a high-reasoning model (Gemini) to generate a "Chain-of-Thought" dataset, which was then used to fine-tune a smaller model (Qwen 2.5 7B) to achieve high performance with limited resources.

### Results
* **Base Model:** Qwen 2.5-7B-Instruct
* **Method:** LoRA Fine-Tuning + Logit Scoring
* **Performance:** Optimized for accuracy using probability distribution over Cyrillic tokens (А, Б, В, Г, Д).

---

##  The Pipeline

### Phase 1: The "Professor" (Data Generation)
 took the raw ZNO training questions and utilized **Gemini 1.5/2.0** to generate explanations.
* **Script:** `generate_gemini.py`
* **Output:** `zno_gemini_reasoning.jsonl` (Explanations + Correct Answers)

### Phase 2: The "Student" (Fine-Tuning)
 trained a LoRA adapter to teach Qwen to mimic the "Chain-of-Thought" reasoning style.
* **Notebook:** `full_code_hw3.ipynb`
* **Adapter:** `zno-my-adapter/` (Weights stored via Git LFS)

### Phase 3: The "Exam" (Offline Inference)
To comply with competition rules (no internet), we use a local installation strategy.
* **Library Prep:** Dependencies are bundled in `offline_libs.zip`.
* **Strategy:** **Logit Scoring**. Instead of text generation, we calculate the mathematical probability of the letters **А, Б, В, Г, Д** at the final token position.

---

## Project Structure

```text
.
├── zno-my-adapter/          # Fine-tuned LoRA weights (Git LFS)
├── offline_libs.zip         # Pre-downloaded Python wheels for Kaggle
├── full_code_hw3.ipynb      # Main training & experimentation notebook
├── report.ipynb             # Final analysis and metrics
├── generate_gemini.py       # Dataset generation script (Teacher phase)
├── zno_gemini_reasoning.py  # Utility for processing teacher data
├── zno_gemini_reasoning.jsonl # The generated "Chain-of-Thought" dataset
├── EXPERIMENT_REPORT.md     # Detailed breakdown of training runs
└── README.md

```

---

##  Reproduction & Kaggle Deployment

### 1. Git LFS Setup

The model weights and library zip exceed GitHub's standard size limit.

```bash
git lfs install
git lfs track "*.zip"
git lfs track "*.safetensors"

```

### 2. Kaggle Setup (Phase 4)

1. **Upload Datasets:** * Create a dataset `zno-libs-final` and upload `offline_libs.zip`.
* Create a dataset `zno-my-adapter` and upload the contents of your adapter folder.


2. **Add Base Model:** Search Kaggle Models for `Qwen2.5-7B-Instruct`.
3. **Run Solver:** Use `zno-solver-offline (1).ipynb`.
* **Inference Strategy:** Uses **Logit Scoring** on Cyrillic markers.
* **Prompting:** Implements the *"Дай відповідь буквою-варіантом..."* instruction optimized for Ukrainian multiple-choice tasks.



---

## 📜 Credits & References

Based on the methodology from **"Benchmarking Multimodal Models for Ukrainian Language Understanding"**.

* **Source:** Paniv et al., UNLP 2025.
* **Key Insight:** Logit scoring on Cyrillic tokens outperforms text generation for multiple-choice QA in low-resource settings.

```


```



**Would you like me to help you draft the `EXPERIMENT_REPORT.md` content based on your training results?**
