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
