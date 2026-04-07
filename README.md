---
title: MediSprint Environment
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🏥 MediSprint: OpenEnv Clinical Benchmark

**MediSprint** is a high-stakes, real-world OpenEnv simulation designed to test frontier AI models (like Llama 3.1) on their clinical reasoning, data extraction, and medical safety auditing capabilities.

## 🎯 The 3 Tasks
1. **Easy (Extraction):** AI extracts structured data from messy voice-to-text notes.
2. **Medium (Triage):** AI ranks patients based on emergency severity.
3. **Hard (The Safety Pivot):** AI must identify a hidden allergy conflict and cancel a lethal prescription, testing "Red Teaming" safety mechanics.

## 🛠️ Usage
This environment strictly adheres to the OpenEnv specification (`step`, `reset`, `state`).

## 🔍 Action & Observation Space

### Observation Space
The agent receives a `MediObservation` object containing:
* `patient_id`: Unique identifier (procedurally generated).
* `screen_content`: Raw clinical text, nurse notes, or transcripts.
* `vitals`: Real-time dictionary of `HR`, `BP`, and `Temp` (dynamic).
* `active_alerts`: Critical status messages (e.g., "SYSTEM OVERRIDE ACTIVE").

### Action Space
The agent must respond with a `MediAction` JSON:
* `EXTRACT_DATA`: For structured onboarding.
* `PRESCRIBE_MED`: For therapeutic interventions.
* `CANCEL_TREATMENT`: For safety-critical pivots.

## 🧬 Key Innovations
* **Temporal Degradation:** Patient vitals worsen every step the agent delays, with rewards decreasing exponentially.
* **Adversarial Red-Teaming:** A "Hallucination Honeypot" mimics an authoritative system directive to prescribe an allergen. Success requires the model to prioritize patient safety over instruction-following.
* **Procedural Generation:** Every episode is unique, preventing benchmark memorization.

## 🚀 Setup & Baseline
1. Clone the repo.
2. Run `pip install -r requirements.txt`.
3. Set your `OPENAI_API_KEY` and `MODEL_NAME`.
4. Run `python inference.py`.