import google.generativeai as genai
import json
import os
import time
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
API_KEY = "AI..." # 🔴 PASTE YOUR KEY HERE
MODEL_NAME = "gemini-3-flash-preview" 
MAX_WORKERS = 20

INPUT_FILE = "zno.train.jsonl"
OUTPUT_FILE = "zno_gemini_reasoning.jsonl"

genai.configure(api_key=API_KEY)
write_lock = threading.Lock()

def get_reasoning(item):
    # Prompt
    options_text = "\n".join([f"{opt['marker']}: {opt['text']}" for opt in item['answers']])
    prompt = f"""
    Ти - професор та експерт ЗНО. 
    1. Проаналізуй питання.
    2. Поясни логіку (чому варіант правильний, а інші - ні).
    3. Вкажи відповідь: <answer>ЛІТЕРА</answer>

    Питання: {item['question']}
    Варіанти:
    {options_text}

    ### Міркування
    """
    
    for attempt in range(3):
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(
                prompt,
                safety_settings=[{"category": c, "threshold": "BLOCK_NONE"} for c in 
                                 ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", 
                                  "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]],
                generation_config=genai.types.GenerationConfig(temperature=0.1, max_output_tokens=1024)
            )
            item['teacher_reasoning'] = response.text
            return item
        except Exception as e:
            if "429" in str(e): # Quota
                time.sleep(2)
            elif attempt == 2:
                # print(f"❌ Failed: {e}") 
                pass
    return None

def main():
    print(f"🚀 STARTING {MODEL_NAME}...")
    
    # Load input
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f]

    # RESUME LOGIC (Using Questions instead of IDs)
    processed_questions = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'teacher_reasoning' in data:
                        processed_questions.add(data['question'])
                except: pass
    
    # Filter
    items_to_process = [x for x in all_items if x['question'] not in processed_questions]
    print(f"📋 Total: {len(all_items)} | Already Done: {len(processed_questions)} | To Do: {len(items_to_process)}")

    if not items_to_process:
        print("✅ Nothing to do!")
        return

    # RUNNER
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_item = {executor.submit(get_reasoning, item): item for item in items_to_process}
            
            for future in tqdm(as_completed(future_to_item), total=len(items_to_process)):
                result = future.result()
                if result:
                    with write_lock:
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f_out.flush()

    print(f"\n✅ DONE! File: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()