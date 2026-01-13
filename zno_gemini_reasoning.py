import google.generativeai as genai
import json
import os
import time
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
# 1. Get Key: https://aistudio.google.com/app/apikey
API_KEY = "AIza" # PASTE YOUR KEY HERE
MODEL_NAME = "gemini-2.0-flash"

INPUT_FILE = "zno.train.jsonl"
OUTPUT_FILE = "zno_gemini_reasoning.jsonl"

genai.configure(api_key=API_KEY)

# Safety: OFF (History/Lit questions often trigger false positives)
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

def get_reasoning(item):
    # Skip if we already have reasoning (resume logic)
    if 'teacher_reasoning' in item:
        return item

    options_text = "\n".join([f"{opt['marker']}: {opt['text']}" for opt in item['answers']])
    
    prompt = f"""
    Ти - експерт з українського ЗНО. Проаналізуй питання та дай детальне пояснення.

    Питання: {item['question']}
    Варіанти відповідей:
    {options_text}

    Завдання:
    1. Проаналізуй питання українською мовою.
    2. Поясни, чому правильна відповідь саме така.
    3. Вкажи фінальну відповідь у форматі: <answer>ЛІТЕРА</answer>

    Формат відповіді:
    ### Міркування
    [Твоє пояснення тут]

    ### Відповідь
    <answer>X</answer>
    """

    # Retry logic handles Free Tier rate limits automatically
    for attempt in range(5):
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1, max_output_tokens=1024
                )
            )
            item['teacher_reasoning'] = response.text
            return item

        except Exception as e:
            # 429 = Too Many Requests (Free Tier limit)
            if "429" in str(e) or "503" in str(e):
                time.sleep(2 + attempt * 2) # Backoff
            else:
                print(f"Error {item.get('id')}: {e}")
                return None
    return None

def main():
    # Load Data
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f]

    # Check existing progress
    processed_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                processed_count += 1
    
    print(f"Starting. Done so far: {processed_count}/{len(all_items)}")
    
    # Filter items that need processing
    # (Simplified logic: we just append new ones)
    # Ideally, load IDs of done items to skip properly. 
    # For now, let's just process everything and if file exists, you handle it.
    
    # WORKERS:
    # Use 2 for Free Tier. Use 10 for Paid Tier.
    MAX_WORKERS = 2 

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # We skip the first 'processed_count' items
            items_to_do = all_items[processed_count:]
            
            results = list(tqdm(executor.map(get_reasoning, items_to_do), total=len(items_to_do)))
            
            for res in results:
                if res:
                    f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f_out.flush()

if __name__ == "__main__":
    main()
