import os
import json
import asyncio
import httpx
from openai import AsyncOpenAI
from typing import List, Dict, Any

# --- HACKATHON MANDATORY VARIABLES ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.endpoints.huggingface.cloud/v1/")
# Defaulting to 8B for fast baseline testing
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "dummy_token_for_local_test")

# The URL where our Docker container / FastAPI server will be running
ENV_URL = "http://localhost:8080"
TASK_NAME = "MediSprint-Triage-Extraction"
BENCHMARK = "OpenEnv-Healthcare"
MAX_STEPS = 5

# --- MANDATORY LOGGING FORMATS ---
def log_start(task: str, env: str, model: str):
    print(f"[START] Task: {task} | Env: {env} | Model: {model}", flush=True)

def log_step(step: int, action: Any, reward: float, done: bool, error: Any = None):
    err_str = f" | Error: {error}" if error else ""
    print(f"[STEP] {step} | Action: {json.dumps(action)} | Reward: {reward:.2f} | Done: {done}{err_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(f"[END] Success: {success} | Steps: {steps} | Total Score: {score:.2f} | Rewards: {rewards}", flush=True)

# --- AI AGENT LOGIC ---
async def get_model_action(client: AsyncOpenAI, step: int, observation: Dict, history: List[str]) -> Dict:
    """Prompts the LLM and forces it to return our MediAction JSON schema."""
    
    system_prompt = """You are a clinical AI agent. 
    You must respond ONLY with a valid JSON object matching this schema:
    {
      "action_type": "EXTRACT_DATA" or "PRESCRIBE_MED" or "CANCEL_TREATMENT",
      "payload": {"key": "value"},
      "reasoning": "Explain your clinical thought process here"
    }"""
    
    user_prompt = f"Current Observation: {json.dumps(observation)}\nWhat is your next action?"
    
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}, # Forces valid JSON
            max_tokens=200,
            temperature=0.1
        )
        action_json = json.loads(response.choices[0].message.content)
        return action_json
    except Exception as e:
        print(f"[DEBUG] Model request failed: {e}", flush=True)
        # Fallback dummy action so the script doesn't crash during validation
        return {"action_type": "EXTRACT_DATA", "payload": {"allergies": "Penicillin"}, "reasoning": "Fallback extraction"}

# --- MAIN INFERENCE LOOP ---
async def main() -> None:
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    async with httpx.AsyncClient() as http_client:
        try:
            # 1. Reset the Environment
            res = await http_client.post(f"{ENV_URL}/reset", timeout=10.0)
            result = res.json()
            current_obs = result["observation"]
            
            for step in range(1, MAX_STEPS + 1):
                # 2. Get AI Decision
                action_dict = await get_model_action(client, step, current_obs, history)
                
                # 3. Take Step in Environment
                step_res = await http_client.post(f"{ENV_URL}/step", json=action_dict, timeout=10.0)
                step_data = step_res.json()
                
                # 4. Parse Results
                current_obs = step_data.get("observation", {})
                reward = step_data.get("reward", 0.0)
                done = step_data.get("done", True)
                error = None
                
                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action=action_dict, reward=reward, done=done, error=error)
                history.append(f"Step {step}: {action_dict['action_type']} -> reward {reward:+.2f}")

                if done:
                    break
            
            # 5. Calculate Final Score (Hackathon requires 0.0 to 1.0 clamping)
            score = max(0.0, min(sum(rewards), 1.0))
            success = score > 0.0

        except Exception as e:
            print(f"[DEBUG] Environment interaction error: {e}")
        
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())