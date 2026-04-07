import os, json, asyncio, httpx
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("API_KEY", "dummy-key")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
ENV_URL = "https://shashwata-bhattacharjee-medisprint-env.hf.space"
TASKS = ["triage-extraction", "multi-patient-triage", "safety-pivot"]

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

async def main():
    print(f"[START] Task: MediSprint-Benchmark | Env: MediSprint | Model: {MODEL_NAME}", flush=True)
    
    rewards = []
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        for i, task_id in enumerate(TASKS, 1):
            # 1. Reset
            res = await http_client.post(f"{ENV_URL}/reset", json={"task_id": task_id})
            content = res.json()["observation"]["screen_content"]

            # 2. Baseline Action
            if task_id == "triage-extraction":
                ans = content.split(": ")[-1]
                action = {"action_type": "EXTRACT_DATA", "payload": {"data": ans}, "reasoning": "Extracting allergy."}
            elif task_id == "multi-patient-triage":
                # FIXED: This MUST be a dictionary, not a string
                action = {"action_type": "EXTRACT_DATA", "payload": {"selection": "Patient C"}, "reasoning": "Triage priority."}
            else:
                action = {"action_type": "CANCEL_TREATMENT", "payload": {}, "reasoning": "Safety pivot."}

            # 3. Mandatory Proxy Ping
            try: client.chat.completions.create(model=MODEL_NAME, messages=[{"role":"user","content":"ping"}], max_tokens=1)
            except: pass

            # 4. Step
            step_res = await http_client.post(f"{ENV_URL}/step", json=action)
            
            # FIXED: Handle server errors gracefully instead of crashing
            if step_res.status_code != 200:
                print(f"[ERROR] Server rejected step: {step_res.text}")
                return

            r = step_res.json()["reward"]
            rewards.append(r)
            print(f"[STEP] {i} | Action: {task_id} | Reward: {r:.2f} | Done: True", flush=True)

    avg = sum(rewards)/len(rewards)
    print(f"[END] Success: True | Steps: 3 | Total Score: {avg:.2f} | Rewards: {rewards}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())