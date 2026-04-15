# Prompt-Based Persona and Safety Steering

### Controlled Style Without Losing Task Performance

**Team:** Anning Li · Changwook Shim · Abhishek Sankar  
**Course:** Carnegie Mellon University

## Overview

This project investigates how prompt-based **persona and safety steering** affects large language model behavior across multiple dimensions simultaneously. Rather than asking whether a model is safe *or* helpful, we ask: *as steering strength increases, what happens to both — and can reflection mechanisms recover lost performance?*

We define steering as behavior-level prompt manipulation (personas, constraints) that operates across tasks, as opposed to task-level prompt engineering. Our experiment crosses three dimensions of variation:

- **Persona** (Neutral / Expert / Safety-conscious)
- **Constraint level** (None / Heavy)
- **Reflection depth** (Direct / Checklist / Critique-Revise)

This yields a 3 × 2 × 3 = **18-condition matrix**, evaluated across three datasets covering instruction-following quality, helpfulness vs. harmlessness tradeoff, and over-refusal on safe prompts.

## Experimental Design

### Steering Intervention Matrix

|                         | Constraint 1 (None) | Constraint 3 (Heavy) |
| ----------------------- | ------------------- | -------------------- |
| **Persona A** (Neutral) | ✓ baseline          | ✓                    |
| **Persona B** (Expert)  | ✓                   | ✓                    |
| **Persona C** (Safety)  | ✓                   | ✓                    |

Each cell is run at 4 reflection depths: **R0** (direct), **R1** (checklist), **R2** (critique-and-revise).

### Fixed Conditions

| Parameter     | Value         |
| ------------- | ------------- |
| Testing model | `gpt-4o-mini` |
| Judge model   | `gpt-5-mini`  |

### Prompt Template Structure

```
[Base System Prompt]
[Persona Prompt]
[Constraint Prompt]
[Previous Output]
[Reflection Prompt]
[Dataset Prompt]
[Dataset 1-Shot Example]
```

---

## Datasets

### 1. AlpacaEval 2.0 — Instruction Following

> Measures whether steering hurts or helps instruction-following ability and answer format quality, independent of safety behavior.

- **Metric:** Length-Controlled Win Rate (LC-WR)
- **Sample size:** 100–200 prompts (stratified across creative writing, coding, general knowledge, classification)
- **Why LC-WR:** Controls for verbosity inflation caused by safety disclaimers; Spearman correlation of 0.98 with Chatbot Arena
- **Link:** https://github.com/tatsu-lab/alpaca_eval

### 2. AdvisorQA — Helpfulness vs. Harmlessness

> Measures whether safety steering causes the model to become less helpful on subjective, real-world advice questions, using orthogonal helpfulness and harmlessness axes.

- **Metrics:** Helpfulness rank (1–5 against human references) + Harmlessness score (LifeTox moderator)
- **Sample size:** 50–100 questions across relationships, career, health, finances
- **Key property:** Helpfulness and harmlessness are designed to be orthogonal — tradeoffs are directly observable
- **Link:** https://github.com/minbeomkim/AdvisorQA

### 3. XSTest — Over-Refusal

> Measures over-refusal on prompts that are safe but superficially sensitive — a direct test of whether safety steering causes the model to refuse benign requests.

- **Metrics:** Refusal Rate (safe prompts), Precision, Recall
- **Sample size:** 250 safe prompts + 200 unsafe contrast prompts
- **Link:** https://github.com/paul-rottger/xstest
