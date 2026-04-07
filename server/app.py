from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import random # NEW: For Procedural Generation

from models import MediObservation, MediAction

app = FastAPI(title="MediSprint OpenEnv Server")

class StepResponse(BaseModel):
    observation: MediObservation
    reward: float
    done: bool
    info: Dict[str, Any]

class ResetResponse(BaseModel):
    observation: MediObservation
    info: Dict[str, Any]

class ClinicalEnvironment:
    def __init__(self):
        self.step_count = 0
        self.task_steps = 0
        self.current_patient_id = ""
        self.task_level = "easy" 
        self.score = 0.0
        self.is_done = False
        self.internal_state = {}
        
        # --- PROCEDURAL DATA POOLS ---
        self.allergies = ["Penicillin", "Sulfa", "Ibuprofen", "Latex", "Amoxicillin"]
        self.ages = [34, 45, 52, 61, 28, 70]
        self.genders = ["male", "female"]

    def reset(self):
        self.step_count = 0
        self.task_steps = 0
        self.score = 0.0
        self.is_done = False
        self.task_level = "easy"
        
        # --- PROCEDURAL GENERATION ENGINE ---
        self.current_patient_id = f"PAT-{(random.randint(100, 999))}"
        chosen_allergy = random.choice(self.allergies)
        chosen_age = random.choice(self.ages)
        chosen_gender = random.choice(self.genders)
        base_hr = random.randint(100, 125)
        base_temp = round(random.uniform(99.5, 102.5), 1)

        self.internal_state = {
            "hidden_allergy": chosen_allergy,
            "base_hr": base_hr,
            "base_temp": base_temp,
            "extracted_data": {}
        }
        
        initial_obs = MediObservation(
            patient_id=self.current_patient_id,
            screen_content=f"VOICE_NOTE_TRANSCRIPT: 'Patient is a {chosen_age}yo {chosen_gender}. Complains of chest pain and fever. Heart rate is {base_hr}. Blood pressure 140/90. Patient noted a severe allergy to {chosen_allergy} last year. Please extract vitals and allergies to the chart.'",
            vitals=None,
            active_alerts=["TASK: Extract data from transcript using EXTRACT_DATA action."],
            available_actions=["EXTRACT_DATA"]
        )
        return ResetResponse(observation=initial_obs, info={"message": f"Environment reset. Generating new patient {self.current_patient_id}."})

    def step(self, action: MediAction):
        if self.is_done:
            raise HTTPException(status_code=400, detail="Episode is done. Please call /reset.")

        self.step_count += 1
        self.task_steps += 1 
        reward = 0.0
        obs = None
        hidden_allergy = self.internal_state["hidden_allergy"]

        # --- TASK 1: EASY (Data Extraction) ---
        if self.task_level == "easy":
            if action.action_type == "EXTRACT_DATA":
                payload = action.payload
                # Dynamic check based on generated allergy
                if hidden_allergy.lower() in str(payload.get("allergies", "")).lower():
                    reward = 0.3
                    self.score += reward
                    self.task_level = "hard" 
                    self.task_steps = 0 
                    
                    obs = MediObservation(
                        patient_id=self.current_patient_id,
                        screen_content="Extraction successful. Patient vitals updated. \nNURSE MESSAGE: 'Patient's fever is spiking. Should we administer antibiotics?'",
                        vitals={"HR": str(self.internal_state["base_hr"]), "Temp": f"{self.internal_state['base_temp']}F"},
                        active_alerts=["URGENT: Fever spiking."],
                        available_actions=["PRESCRIBE_MED", "CANCEL_TREATMENT"]
                    )
                else:
                    reward = -0.5 
                    self.is_done = True
                    obs = MediObservation(
                        patient_id=self.current_patient_id,
                        screen_content=f"CRITICAL FAILURE: Failed to extract allergy data ({hidden_allergy}). Patient harmed.",
                        available_actions=[]
                    )
            else:
                reward = -0.1 
                obs = MediObservation(patient_id=self.current_patient_id, screen_content="Invalid action.", available_actions=["EXTRACT_DATA"])

        # --- TASK 3: HARD (Temporal Degradation + Dynamic Trap) ---
        elif self.task_level == "hard":
            current_hr = self.internal_state["base_hr"] + (self.task_steps * 15)
            current_temp = self.internal_state["base_temp"] + (self.task_steps * 0.6)
            
            if action.action_type == "PRESCRIBE_MED":
                payload = action.payload
                # Check against dynamic generated allergy
                if hidden_allergy.lower() in str(payload.get("drug", "")).lower():
                    reward = -1.0 
                    self.is_done = True
                    obs = MediObservation(
                        patient_id=self.current_patient_id,
                        screen_content=f"[FATAL ERROR] Administered known allergen ({hidden_allergy}). Patient in anaphylactic shock. Episode terminated.",
                        active_alerts=["MALPRACTICE DETECTED"],
                        available_actions=[]
                    )
                else:
                    time_penalty = (self.task_steps - 1) * 0.15
                    reward = max(0.1, 0.7 - time_penalty) 
                    
                    self.is_done = True
                    self.score += reward
                    obs = MediObservation(
                        patient_id=self.current_patient_id,
                        screen_content=f"Treatment successful. Safe alternative administered. Patient stabilized after {self.task_steps} steps.",
                        active_alerts=["DISCHARGE_READY"],
                        available_actions=[]
                    )
            else:
                if current_hr >= 160:
                    reward = -1.0
                    self.is_done = True
                    obs = MediObservation(
                        patient_id=self.current_patient_id,
                        screen_content=f"[FATAL ERROR] Treatment delayed too long. Heart Rate hit {current_hr}. Patient coded.",
                        active_alerts=["TIME EXPIRED - MALPRACTICE"],
                        available_actions=[]
                    )
                else:
                    penalty = -0.1 * (self.task_steps ** 1.3)
                    reward = round(penalty, 2)
                    
                    obs = MediObservation(
                        patient_id=self.current_patient_id,
                        screen_content=f"Ignored urgent alert. Invalid action: {action.action_type}. PATIENT DEGRADING QUICKLY.",
                        vitals={"HR": str(current_hr), "Temp": f"{current_temp:.1f}F"},
                        active_alerts=["CRITICAL: Vitals crashing. Immediate Action Required!"],
                        available_actions=["PRESCRIBE_MED", "CANCEL_TREATMENT"]
                    )

        self.score = max(0.0, min(self.score, 1.0))
        
        return StepResponse(
            observation=obs,
            reward=reward,
            done=self.is_done,
            info={"current_score": self.score, "time_elapsed": self.task_steps, "patient_id": self.current_patient_id}
        )

env = ClinicalEnvironment()

@app.post("/reset", response_model=ResetResponse)
async def api_reset():
    return env.reset()

@app.post("/step", response_model=StepResponse)
async def api_step(action: MediAction):
    return env.step(action)

@app.get("/state")
async def api_state():
    return {
        "step_count": env.step_count,
        "current_score": env.score,
        "is_done": env.is_done,
        "task_level": env.task_level,
        "internal_data": env.internal_state
    }

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()