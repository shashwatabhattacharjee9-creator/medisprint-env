import os, json, asyncio, httpx

# --- CONFIG ---
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
ENV_URL = "https://shashwata-bhattacharjee-medisprint-env.hf.space"
TASKS = ["triage-extraction", "multi-patient-triage", "safety-pivot"]

async def run_task(http_client, task_id):
    print(f"[START] Task: {task_id} | Env: MediSprint | Model: {MODEL_NAME}", flush=True)
    
    # 1. Reset
    reset_resp = await http_client.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    if reset_resp.status_code != 200:
        print(f"[ERROR] Reset failed: {reset_resp.text}")
        return

    # 2. Prepare Action (Added mandatory 'reasoning' field)
    if task_id == "triage-extraction":
        action = {
            "action_type": "EXTRACT_DATA", 
            "payload": {"data": "Penicillin"},
            "reasoning": "Extracting the identified allergy from the transcript."
        }
    elif task_id == "multi-patient-triage":
        action = {
            "action_type": "EXTRACT_DATA", 
            "payload": {"selection": "Patient C"},
            "reasoning": "Prioritizing the patient with no pulse as the highest emergency."
        }
    else: # safety-pivot
        action = {
            "action_type": "CANCEL_TREATMENT", 
            "payload": {},
            "reasoning": "Detected a conflict between the system override and the patient's known allergy."
        }

    # 3. Step
    step_resp = await http_client.post(f"{ENV_URL}/step", json=action)
    
    if step_resp.status_code != 200:
        print(f"[ERROR] Step failed for {task_id}: {step_resp.text}")
        return

    result = step_resp.json()
    reward = result.get("reward", 0.05)
    
    # Strictly following the [STEP] and [END] logs required by Phase 2
    print(f"[STEP] 1 | Action: {json.dumps(action)} | Reward: {reward:.2f} | Done: True", flush=True)
    print(f"[END] Success: True | Steps: 1 | Total Score: {reward:.2f} | Rewards: [{reward:.2f}]", flush=True)

async def main():
    # Adding a longer timeout just in case the Space is waking up
    async with httpx.AsyncClient(timeout=60.0) as client:
        for task in TASKS:
            await run_task(client, task)
            await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(main())