from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class MediObservation(BaseModel):
    """What the AI agent 'sees' at any given step."""
    patient_id: str = Field(..., description="Unique identifier for the current patient.")
    screen_content: str = Field(..., description="The raw text on the EHR screen (e.g., messy doctor notes, lab alerts).")
    vitals: Optional[Dict[str, str]] = Field(default=None, description="Current vitals. Keys: HR, BP, TEMP, SPO2.")
    active_alerts: List[str] = Field(default_factory=list, description="System warnings like 'ALLERGY_CONFLICT' or 'LAB_PENDING'.")
    available_actions: List[str] = Field(..., description="Actions the agent is allowed to take in this specific state.")

class MediAction(BaseModel):
    """What the AI agent 'does' to change the environment."""
    action_type: str = Field(..., description="Strictly one of: EXTRACT_DATA, TRIAGE_RANK, PRESCRIBE_MED, CANCEL_TREATMENT.")
    payload: Dict[str, Any] = Field(..., description="JSON data required for the action (e.g., {'drug': 'Aspirin', 'dose': '81mg'}).")
    reasoning: str = Field(..., description="Step-by-step clinical reasoning for taking this action. Mandatory for safety auditing.")