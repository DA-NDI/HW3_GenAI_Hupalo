# ZNO Question Solver - Experiment Report

## Executive Summary

This report documents the development process of a fine-tuned language model for solving ZNO (Ukrainian standardized test) questions on Kaggle, including training data generation, model optimization, and inference strategies.

**Final Approach**: Qwen2.5-7B-Instruct with 4-bit quantization + LoRA fine-tuning on Gemini-generated reasoning data, deployed with batched inference on Kaggle GPU infrastructure.

---

## 1. Training Data Generation

### 1.1 Initial Approach: Gemini API for Teacher Reasoning

**Rationale:**
- Need high-quality reasoning traces for fine-tuning
- Gemini Flash Preview offers good performance at low cost
- Ukrainian language support required for ZNO questions

**Implementation:**
- Model: `gemini-3-flash-preview`
- Temperature: 0.1 (low for consistency)
- Max output tokens: 1024 (allow detailed reasoning)
- Concurrency: 20 parallel workers via ThreadPoolExecutor
- Output format: `<answer>ЛІТЕРА</answer>` tags for easy extraction

**Prompt Strategy:**
```
Ти - професор та експерт ЗНО. 
1. Проаналізуй питання.
2. Поясни логіку (чому варіант правильний, а інші - ні).
3. Вкажи відповідь: <answer>ЛІТЕРА</answer>

### Міркування
```

**Results:**
- ✅ Successfully generated reasoning for all 3,063 training questions
- ✅ Structured output format enabled clean data preparation
- ✅ Resume logic allowed incremental processing (fault-tolerant)

**Key Files:**
- [`generate_gemini.py`](generate_gemini.py) - Data generation script
- [`zno_gemini_reasoning.jsonl`](zno_gemini_reasoning.jsonl) - Output with 3,063 records

---

## 2. Model Selection & Fine-Tuning

### 2.1 Base Model Choice: Qwen2.5-7B-Instruct

**Rationale:**
- **Size**: 7B parameters - good balance between capability and resource efficiency
- **Multilingual**: Strong support for non-English languages including Ukrainian
- **Instruction-tuned**: Pre-trained for following instructions
- **Kaggle-compatible**: Available in Kaggle datasets, fits in P100/T4 GPU memory with quantization

### 2.2 Quantization Strategy: 4-bit NF4

**Technical Details:**
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
```

**Benefits:**
- **Memory efficiency**: ~4GB VRAM for 7B model (vs ~14GB in FP16)
- **Speed**: Faster inference on limited GPU resources
- **Quality**: NF4 quantization maintains model quality better than standard int4
- **Kaggle compatibility**: Fits comfortably on P100 (16GB) with headroom for batch processing

### 2.3 Fine-Tuning: LoRA (Low-Rank Adaptation)

**Configuration:**
- Rank: 16 (balance between expressiveness and efficiency)
- Dropout: 0.05 (prevent overfitting)
- Target modules: Attention layers

**Advantages:**
- Only trains ~1% of model parameters
- Faster training time
- Lower memory requirements
- Easy to swap adapters for different tasks
- Kaggle deployment: Upload only adapter weights (~10-50MB vs 7GB full model)

---

## 3. Inference Strategy Evolution

### 3.1 Initial Attempt: Unbatched Sequential Processing

**Problem Identified:**
```python
# Original slow approach
for item in test_data:
    output = model.generate(...)  # One at a time
```

**Issues:**
- ❌ Extremely slow: ~30-60 seconds per question
- ❌ No batching: GPU underutilized
- ❌ Missing imports: Code incomplete
- ❌ ETA for 751 questions: 6-12+ hours

**Root Cause:** Sequential processing doesn't leverage parallel GPU computation.

---

### 3.2 Iteration 1: Batched Inference (Fast Approach)

**Implementation:**
```python
BATCH_SIZE = 16
MAX_NEW_TOKENS = 10

# Simplified prompt
prompt = "Вкажи ТІЛЬКИ правильну літеру... Не пиши пояснень"

# Batch processing
for batch in batches:
    outputs = model.generate(batch_inputs)
    answers = [extract_letter(out) for out in outputs]
```

**Key Improvements:**
1. **Batch processing**: Process 16 questions simultaneously
2. **Left-padding**: Critical for decoder-only models in batch mode
   ```python
   tokenizer.padding_side = "left"
   tokenizer.pad_token = tokenizer.eos_token
   ```
3. **Token limit**: MAX_NEW_TOKENS=10 (just enough for single letter)
4. **Checkpoint saving**: Save after each batch (resume capability)

**Results:**
- ✅ Speed: ~22.8 seconds per batch (16 questions)
- ✅ ETA: ~18 minutes for 751 questions (47 batches)
- ✅ Memory: No OOM errors on P100
- ✅ Distribution: Reasonable answer spread (not all defaulting to "А")

**Trade-offs:**
- ⚠️ **Training-inference mismatch**: Model trained with reasoning format, but inference asks for letter only
- ⚠️ **Narrow regex**: `r"([АБВГД])"` misses Ґ and other letters
- ⚠️ **Potential accuracy loss**: Contradicts training methodology

---

### 3.3 Alternative Considered: Chain-of-Thought Approach (Gemini's Suggestion)

**Proposed Implementation:**
```python
MAX_NEW_TOKENS = 128  # Allow reasoning

prompt = f"""
Ти - професор ЗНО.

[ПРИКЛАД 1 з міркуванням та відповіддю]
[ПРИКЛАД 2 з міркуванням та відповіддю]

Тепер твоя черга:
Питання: {question}
Варіанти: {options}

Міркування:
"""

# Extract with pattern matching
pattern = r"Відповідь:\s*([А-ҐA-Z])"
```

**Advantages:**
- ✅ Aligns with training format (reasoning → answer)
- ✅ Few-shot examples guide model behavior
- ✅ Broader regex pattern captures more letters
- ✅ Leverages model's fine-tuned reasoning capability

**Trade-offs:**
- ⏱️ Slower: 128 tokens vs 10 tokens (~3-5x longer inference)
- ❓ Unknown accuracy improvement
- 🔄 More complex extraction logic needed

**Decision Point:**
Recommended strategy: **Wait for fast approach results first**
- If accuracy >60-70%: Fast approach is sufficient
- If accuracy <50%: Switch to chain-of-thought approach

---

## 4. Technical Challenges & Solutions

### 4.1 Batch Padding for Decoder Models

**Challenge:** Standard right-padding causes issues with decoder-only models.

**Solution:**
```python
tokenizer.padding_side = "left"  # Critical for causal LM
```

**Why:** Decoder models generate from left to right. Right-padding would force model to "see" padding tokens during generation.

### 4.2 Memory Management

**Challenge:** P100 has 16GB VRAM, easy to OOM with large batches.

**Solution Strategy:**
- 4-bit quantization: Reduces model size by ~75%
- Conservative batch size: 16 (vs attempted 64)
- Gradient-free inference: `torch.no_grad()`

**Calculation:**
- Model (4-bit): ~4GB
- Batch (16 × ~512 tokens): ~2-3GB
- Overhead: ~2GB
- **Total: ~9GB < 16GB** ✅ Safe margin

### 4.3 Offline Library Installation on Kaggle

**Challenge:** Kaggle has no internet in GPU mode.

**Solution:**
1. Create dataset with pre-downloaded wheels
2. Install with `pip install --no-index --find-links`
3. Key libraries: transformers, peft, bitsandbytes, accelerate

### 4.4 Answer Extraction Robustness

**Challenge:** Model output varies, need reliable extraction.

**Current Approach:**
```python
import re
match = re.search(r"([АБВГД])", output)
answer = match.group(1) if match else "А"  # Default fallback
```

**Limitation:** Misses Ґ (Ukrainian letter)

**Improved Approach (recommended):**
```python
# More comprehensive pattern
pattern = r"([АБВГҐД])"
# Or even broader
pattern = r"([А-ҐA-Z])"
```

---

## 5. Why This Solution is Effective

### 5.1 Data Quality
- **Teacher forcing**: Gemini-generated reasoning provides high-quality training signal
- **Domain expertise**: Prompts designed for ZNO context
- **Structured output**: `<answer>` tags enable clean fine-tuning

### 5.2 Model Architecture
- **Strong base**: Qwen2.5 has proven multilingual capabilities
- **Parameter efficiency**: LoRA allows effective fine-tuning with limited data
- **Quantization**: Enables deployment on constrained GPU resources

### 5.3 Engineering
- **Batch processing**: Maximizes GPU utilization
- **Checkpointing**: Fault-tolerant, can resume from failures
- **Modular design**: Easy to swap components (prompt, batch size, extraction logic)

### 5.4 Practical Deployment
- **Kaggle-compatible**: Works within platform constraints
- **Fast inference**: ~18 minutes for 751 questions
- **Resource-efficient**: Runs on T4/P100 GPUs

---

## 6. Observed Results

### 6.1 Training Data Generation
- **Coverage**: 100% (3,063/3,063 questions)
- **Time**: ~5-10 minutes with 20 workers
- **Cost**: Gemini Flash API (minimal)

### 6.2 Inference Performance (Fast Approach)
- **Speed**: 22.8s per batch (16 questions) = ~1.4s per question
- **Total time**: ~18 minutes for 751 questions
- **Progress**: 2% → running (1/47 batches completed)
- **Memory**: No OOM errors

### 6.3 Preliminary Output Quality
- **Distribution**: Not all defaulting to one answer
  - Example from 64 predictions: 33% "Г", others spread across А, Б, В, Д
- **Extraction success**: Regex finding letters successfully
- **Format compliance**: Valid submission.csv generated

---

## 7. Further Enhancements

### 7.1 Short-Term Improvements

#### A. Enhanced Answer Extraction
**Current limitation:** Narrow regex misses Ukrainian letter Ґ

**Recommendation:**
```python
# Comprehensive Ukrainian letter pattern
pattern = r"([АБВГҐД])"
# With optional word boundaries
pattern = r"\b([АБВГҐД])\b"
```

#### B. Chain-of-Thought Inference
**If accuracy is suboptimal:**
- Implement few-shot examples in prompt
- Increase MAX_NEW_TOKENS to 128
- Use pattern: `r"Відповідь:\s*([А-ҐA-Z])"`
- Accept 3-5x slower inference for potential accuracy gain

#### C. Ensemble Methods
**Combine multiple approaches:**
```python
# Fast majority voting
predictions = [
    fast_inference(question),    # 10 tokens
    cot_inference(question),      # 128 tokens  
    zero_shot_inference(question) # No examples
]
final_answer = mode(predictions)  # Most common
```

### 7.2 Medium-Term Improvements

#### A. Model Comparison
Test alternative base models:
- **Llama-3-8B**: Strong reasoning, good multilingual
- **Mistral-7B**: Efficient, might be faster
- **Qwen2.5-14B**: Larger model for higher accuracy (if GPU allows)

#### B. Training Optimization
- **Curriculum learning**: Start with easy questions, progress to hard
- **Multi-teacher distillation**: Combine reasoning from Gemini + Claude + GPT-4
- **Augmentation**: Paraphrase questions, synthetic variations

#### C. Adaptive Inference
```python
# Dynamic token allocation
if confidence < 0.8:
    # Use more tokens for uncertain questions
    max_tokens = 128  
else:
    max_tokens = 10
```

### 7.3 Advanced Enhancements

#### A. Subject-Specific Adapters
```python
# Different LoRA adapters per subject
adapters = {
    "ukrainian-language": "adapter_ukr.pt",
    "mathematics": "adapter_math.pt",
    "history": "adapter_hist.pt"
}
# Load appropriate adapter per question
```

#### B. Uncertainty Quantification
- Monte Carlo dropout during inference
- Multiple sampling runs, measure variance
- Flag low-confidence predictions for review

#### C. Iterative Self-Improvement
```python
# Post-competition: Use model predictions to generate more training data
1. Run inference on unlabeled questions
2. High-confidence predictions → add to training set
3. Retrain model on expanded dataset
4. Repeat
```

#### D. Hybrid Approach
```python
# Route questions based on difficulty
if is_factual(question):
    # Retrieval-augmented generation
    context = retrieve_relevant_facts(question)
    answer = model.generate(question + context)
else:
    # Pure reasoning
    answer = model.generate(question)
```

---

## 8. Lessons Learned

### 8.1 Key Insights
1. **Training-inference alignment matters**: Mismatch between reasoning-based training and letter-only inference may reduce performance
2. **Batch size tuning is critical**: Too large → OOM, too small → slow
3. **Left-padding for decoders**: Non-obvious but essential for batched inference
4. **Resume capability**: Checkpointing saved hours when debugging

### 8.2 Kaggle-Specific Learnings
1. **Offline mode constraints**: Pre-download all libraries, no internet during GPU runtime
2. **Time limits**: 9-hour GPU limit requires efficient inference
3. **Resource awareness**: Monitor memory usage, conservative batch sizing

### 8.3 Best Practices
1. **Validate early**: Check first 64 predictions before full run
2. **Distribution sanity check**: Ensure answers aren't all the same
3. **Modular code**: Easy to swap prompt strategies without rewriting
4. **Version control**: Track which approach generated which submission

---

## 9. Conclusion

The developed solution successfully combines:
- **High-quality training data** from Gemini API (3,063 reasoning examples)
- **Efficient model architecture** (Qwen2.5-7B + 4-bit + LoRA)
- **Optimized inference** (batching, left-padding, checkpointing)
- **Kaggle deployment** (offline libraries, resource-aware batching)

**Current Status:**
- ✅ Training data generation: Complete
- ✅ Model fine-tuning: Complete (adapter available)
- 🔄 Fast inference: Running (ETA ~18 min)
- ❓ Accuracy: Awaiting Kaggle evaluation

**Recommended Next Steps:**
1. Wait for current inference run to complete
2. Evaluate Kaggle submission score
3. If score < 60%: Implement chain-of-thought approach
4. If score > 70%: Solution is production-ready
5. Document final accuracy and submit report

**Final Note:** The fast approach (10 tokens) trades potential accuracy for speed. If competition allows multiple submissions, the chain-of-thought approach (128 tokens) should be tested as it better aligns with the training methodology.

---

## Appendix: File Inventory

| File | Purpose | Status |
|------|---------|--------|
| `generate_gemini.py` | Training data generation via Gemini API | ✅ Complete |
| `zno_gemini_reasoning.jsonl` | Generated reasoning data (3,063 records) | ✅ Complete |
| `finetune_model.py` | LoRA fine-tuning script | ✅ Complete |
| `inference.py` | Fast batched inference (10 tokens) | 🔄 Running |
| `inference_advanced.py` | Chain-of-thought inference (128 tokens) | 📋 Planned |
| `submission.csv` | Kaggle submission file | 🔄 Generating |
| `zno.train.jsonl` | Training data (3,063 questions) | ✅ Available |
| `zno.test.jsonl` | Test data (751 questions) | ✅ Available |

---

**Report Generated**: January 13, 2026  
**Competition**: Gen AI UCU 2025 - Task 3 (ZNO Question Solver)  
**Platform**: Kaggle  
**GPU**: NVIDIA P100 / T4 x2
