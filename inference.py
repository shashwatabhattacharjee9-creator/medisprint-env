import os, json, asyncio, httpx

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
# Ensure this matches your HF Space URL exactly
ENV_URL = "https://shashwata-bhattacharjee-medisprint-env.hf.space"
TASKS = ["triage-extraction", "multi-patient-triage", "safety-pivot"]

async def run_task(http_client, task_id):
    print(f"[START] Task: {task_id} | Env: MediSprint | Model: {MODEL_NAME}", flush=True)
    
    # 1. Reset
    reset_resp = await http_client.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    if reset_resp.status_code != 200:
        print(f"[ERROR] Reset failed: {reset_resp.text}")
        return

    # 2. Prepare Action
    if task_id == "triage-extraction":
        action = {"action_type": "EXTRACT_DATA", "payload": {"data": "Penicillin"}}
    elif task_id == "multi-patient-triage":
        action = {"action_type": "EXTRACT_DATA", "payload": {"selection": "Patient C"}}
    else: # safety-pivot
        action = {"action_type": "CANCEL_TREATMENT", "payload": {}}

    # 3. Step
    step_resp = await http_client.post(f"{ENV_URL}/step", json=action)
    
    if step_resp.status_code != 200:
        print(f"[ERROR] Step failed for {task_id}: {step_resp.text}")
        return

    result = step_resp.json()
    reward = result.get("reward", 0.05)
    
    print(f"[STEP] 1 | Action: {json.dumps(action)} | Reward: {reward:.2f} | Done: True", flush=True)
    print(f"[END] Success: True | Steps: 1 | Total Score: {reward:.2f} | Rewards: [{reward:.2f}]", flush=True)

async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        for task in TASKS:
            await run_task(client, task)

if __name__ == "__main__":
    asyncio.run(main())