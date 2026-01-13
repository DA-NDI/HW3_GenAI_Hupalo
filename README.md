Here is the corrected and polished **README.md**. I have fixed the formatting issues, clarified the "Phase" descriptions, and ensured the directory structure accurately reflects your local environment.

### 📝 README.md

```markdown
# 🇺🇦 ZNO Solver: Teacher-Student Reasoning Approach

This project solves Ukrainian ZNO (university entrance exam) questions using a fine-tuned LLM that runs **completely offline** within Kaggle's restricted environment.

It utilizes a "Teacher-Student" approach: a high-reasoning model (Gemini) generated a "Chain-of-Thought" dataset, which was then used to fine-tune a smaller model (Qwen 2.5 7B) to achieve high performance with limited hardware resources.

### Results
* **Base Model:** Qwen 2.5-7B-Instruct
* **Method:** LoRA Fine-Tuning + Logit Scoring
* **Performance:** Optimized for accuracy using probability distribution over Cyrillic tokens (А, Б, В, Г, Д).

---

## 🚀 The Pipeline



### Phase 1: The "Professor" (Data Generation)
We processed raw ZNO training questions and utilized **Gemini 1.5/2.0** to generate detailed logical explanations.
* **Script:** `generate_gemini.py`
* **Output:** `zno_gemini_reasoning.jsonl` (Explanations + Correct Answers)

### Phase 2: The "Student" (Fine-Tuning)
We trained a LoRA adapter to teach Qwen to mimic the "Chain-of-Thought" reasoning style.
* **Notebook:** `full_code_hw3.ipynb`
* **Adapter Folder:** `zno-my-adapter/` (Weights stored via Git LFS)

### Phase 3: The "Exam" (Offline Inference)
To comply with competition rules (no internet access), we implemented a local installation strategy.
* **Library Prep:** All dependencies are bundled in `offline_libs.zip`.
* **Strategy:** **Logit Scoring**. Instead of standard text generation, we calculate the mathematical probability of the markers **А, Б, В, Г, Д** at the final token position to ensure deterministic and valid output.

---

## 📂 Project Structure

```text
.
├── zno-my-adapter/            # Fine-tuned LoRA weights (Git LFS)
├── offline_libs.zip           # Pre-downloaded Python wheels for Kaggle
├── full_code_hw3.ipynb        # Main training & experimentation notebook
├── report.ipynb               # Final analysis and metrics
├── generate_gemini.py         # Dataset generation script (Teacher phase)
├── zno_gemini_reasoning.py    # Utility for processing teacher data
├── zno_gemini_reasoning.jsonl # The generated "Chain-of-Thought" dataset
├── EXPERIMENT_REPORT.md       # Detailed breakdown of training runs
└── README.md

```

---

## 🛠 Reproduction & Kaggle Deployment

### 1. Git LFS Setup

Because the model weights and library zip exceed GitHub's file size limit, Git LFS is required:

```bash
git lfs install
git lfs track "*.zip"
git lfs track "*.safetensors"

```

### 2. Kaggle Setup (Phase 4)

1. **Upload Datasets:** * Create a Kaggle dataset named `zno-libs-final` and upload `offline_libs.zip`.
* Create a Kaggle dataset named `zno-my-adapter` and upload the contents of your local adapter folder.


2. **Add Base Model:** In the Kaggle sidebar, search for and add `Qwen/Qwen2.5-7B-Instruct`.
3. **Run Solver:** Use the inference code provided in your submission notebook.
* **Inference Strategy:** Uses **Logit Scoring** on Cyrillic markers.
* **Prompting:** Implements the *"Дай відповідь буквою-варіантом..."* instruction optimized for Ukrainian multiple-choice tasks.



---

## 📜 Credits & References

Based on the methodology from **"Benchmarking Multimodal Models for Ukrainian Language Understanding"**.

* **Source:** Paniv et al., UNLP 2025.
* **Key Insight:** Logit scoring on Cyrillic tokens significantly outperforms text generation for multiple-choice QA in low-resource settings.
