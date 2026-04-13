# Illinois Advanced Research Center SG (IARCS) — Take Home Assignment

## How to Run

1. Install dependencies.

```powershell
pip install pandas python-dotenv pydantic openai google-genai mistralai numpy scikit-learn
```

2. Add API keys in `.env`
```env
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
MISTRAL_API_KEY=your_mistral_key
```

3. Run the base multi-model evaluation.

```powershell
python task.py
```

4. Run the RAG pipeline.

```powershell
python rag.py
```

**Author:** Vaishavi Venkatesh  
**Date:** 13/04/2026

> This project explores the use of Large Language Models (LLMs) to map CVE vulnerability descriptions to relevant MITRE ATT&CK techniques.

---

## Table of Contents

1. [Models Evaluated](#1-models-evaluated)
2. [Prompt Design Choices](#2-prompt-design-choices)
3. [Ensuring Accuracy](#3-ensuring-accuracy)
4. [Challenges and Limitations](#4-challenges-and-limitations)
5. [Analytics and Metrics](#5-analytics-and-metrics)
   - [5.1 Accuracy](#51-accuracy)
   - [5.2 Precision](#52-precision)
   - [5.3 Hallucination Rate](#53-hallucination-rate)
   - [5.4 Latency](#54-latency)
6. [Bonus: RAG Implementation](#bonus-rag-implementation)
   - [Advantages and Disadvantages](#advantages-and-disadvantages-of-using-rag-for-cve-mapping)
   - [Task 1: Confidence Scoring](#task-1-confidence-scoring-llm-output-enhancement)
   - [Task 2: Justification Field](#task-2-justification-field-explainability-layer)
   - [Task 3: RAG Integration](#task-3-rag-retrieval-augmented-generation-integration)
   - [RAG Results](#rag-results)
   - [RAG Overall Metrics](#rag-overall-metrics)
7. [Appendix](#appendix)

---

## 1. Models Evaluated

The following LLMs were tested:

| Model | Type | Why it was used |
|---|---|---|
| GPT-5.4 | Proprietary (OpenAI) | Current State-of-the-Art. Used as the flagship benchmark due to its superior agentic reasoning and the lowest recorded hallucination rate (18% lower than GPT-5.2). It excels at "Chain of Thought" classification for complex CVEs. |
| GPT-4o | Proprietary (OpenAI) | Used as the primary benchmark model due to its strong reasoning ability, high accuracy, and low hallucination rate. It provided the most consistent and reliable MITRE ATT&CK mappings. |
| GPT-4o-mini | Proprietary (OpenAI, lightweight) | Used to evaluate performance of a smaller, faster model. It offers lower latency and cost, but helped highlight trade-offs such as occasional hallucinations and reduced accuracy. |
| Mistral-medium-latest | Open-source | Included to satisfy the requirement of using an open-source model. It demonstrated strong coverage (high recall) by suggesting more techniques, but often over-generated results, requiring validation. |

---

## 2. Prompt Design Choices

The pipeline used a **two-part prompt strategy**:

- A **system prompt** established the model as a cybersecurity analyst with knowledge of MITRE ATT&CK, and constrained output to a strict JSON schema.
- A **user prompt** provided the CVE ID plus its full NVD description, then instructed the model to infer applicable technique IDs, names, and reasoning.

Several prompt engineering techniques were used to improve accuracy and consistency:

- **Few-shot prompting:** A small number of mapping examples were included to guide the model's reasoning and improve consistency, making sure the number of examples was limited to avoid overfitting.
- **Context window considerations:** Only essential information (CVE description, key attributes, and instructions) was included to maintain focus.
- **Lost-in-the-middle mitigation:** Key instructions (objective and output rules) were placed at both the beginning and end of the prompt to ensure they were consistently followed.
- **Chain-of-thought prompting:** The model was guided to reason step-by-step (vulnerability type → attack vector → impact → technique), improving overall response.

---

## 3. Ensuring Accuracy

- **MITRE ATT&CK dataset validation:** All predicted technique IDs were cross-checked against the official MITRE ATT&CK Enterprise dataset (v17.1).
- **Hallucination detection:** Hallucinated or invalid technique predictions were explicitly filtered during post-processing. The system validates predicted technique IDs against the retrieved candidate set, and any technique not present in the retrieved MITRE list is flagged as a hallucination, removed from the final structured output, and logged in `model_comparisons_log.json`.
- **Multi-model comparison:** Outputs from multiple LLMs were compared to identify consistent techniques across models. Techniques predicted by multiple models were considered more reliable.
- **Filtering and parsing:** Responses were programmatically parsed to extract only structured fields (technique ID, name), ensuring clean and valid JSON output.

---

## 4. Challenges and Limitations

- **Smaller model hallucinations:** Lightweight models (e.g., GPT-4o-mini) showed higher hallucination rates. When few-shot examples were included, they sometimes overfit to patterns and reused incorrect logic. Without examples, they were more likely to generate non-existent MITRE technique IDs.
- **Limited local model experimentation:** Running large open-source models locally was constrained by hardware limitations, which restricted experimentation with a wider range of local LLMs. As a result, most evaluations relied on API-based models.
- **Inconsistent outputs:** Even with temperature set to 0 and a fixed seed (e.g., `seed = 42`), some models still produced varying outputs across runs, indicating that true determinism is not always guaranteed in LLM inference APIs due to backend conditions.

---

## 5. Analytics and Metrics

The ground-truth ("correct") technique labels were derived through manual analysis of the CVE descriptions by interpreting the underlying attack behavior and mapping it to the most appropriate MITRE ATT&CK technique:

1. `CVE-2021-21148` → **T1203** (Exploitation for Client Execution)
2. `CVE-2020-1472` → **T1068** (Exploitation for Privilege Escalation), **T1210** (Exploitation of Remote Services)
3. `CVE-2021-21975` → **T1190** (Exploit Public-Facing Application)

### 5.1 Accuracy

> Partial credit (0.5) is given for CVEs with multiple ground truth techniques where only one is found.

| Model | CVE-21148 | CVE-1472 | CVE-21975 | Total Correct | Accuracy |
|---|---|---|---|---|---|
| gpt-5.4 | 1/1 | 2/2 | 1/1 | **4/4** | **1.0** |
| gpt-4o | 1/1 | 2/2 | 0/1 | **3/4** | **0.75** |
| gpt-4o-mini | 1/1 | 1/2 | 1/1 | **3/4** | **0.75** |
| mistral-medium | 0/1 | 1/2 | 1/1 | **2/4** | **0.5** |

### 5.2 Precision

> This metric penalises over-generation.

| Model | Total Techniques Returned | Ground Truth Hits | Precision |
|---|---|---|---|
| gpt-5.4 | 5 | 4 | **0.8** |
| gpt-4o | 4 | 3 | **0.75** |
| gpt-4o-mini | 10 | 3 | **0.3** |
| mistral-medium | 13 | 2 | **0.15** |

### 5.3 Hallucination Rate

> Proportion of predicted technique IDs that are invalid/hallucinated per CVE.

| Model | CVE-2021-21148 | CVE-2020-1472 | CVE-2021-21975 | Avg Hallucination Rate |
|---|---|---|---|---|
| GPT-4o-mini | 0.20 | 0.40 | 0.00 | **0.20** |
| GPT-4o | 0.00 | 0.00 | 0.00 | **0.00** |
| GPT-5.4 | 0.00 | 0.00 | 0.00 | **0.00** |
| Mistral-medium-latest | 0.00 | 0.00 | 0.00 | **0.00** |

### 5.4 Latency

> Raw response time in seconds per CVE.

| Model | CVE-2021-21148 | CVE-2020-1472 | CVE-2021-21975 | Avg Latency (s) |
|---|---|---|---|---|
| GPT-4o-mini | 8.33 | 4.51 | 3.64 | **5.49** |
| GPT-4o | 3.47 | 2.56 | 2.79 | **2.94** |
| GPT-5.4 | 4.27 | 3.89 | 3.49 | **3.88** |
| Mistral-medium-latest | 3.69 | 3.73 | 3.65 | **3.69** |

---

## Bonus: RAG Implementation

### Advantages and Disadvantages of Using RAG for CVE Mapping

| Pros | Cons |
|---|---|
| Grounded in external MITRE ATT&CK knowledge, improving factual consistency and reducing unsupported reasoning | High semantic similarity between techniques makes fine-grained classification difficult |
| Reduces hallucination by constraining the model to retrieved candidate techniques instead of free generation | MITRE descriptions are often short or abstract, limiting discriminative power |
| Improves interpretability since each prediction is based on a visible candidate set | Retrieval can over-prioritise keyword matching over true semantic similarity |

---

### Task 1: Confidence Scoring (LLM Output Enhancement)

To improve interpretability, each candidate MITRE ATT&CK technique was assigned a **confidence score** by the LLM. This score reflects how strongly the model believes the CVE maps to a given technique based on semantic alignment between:

- CVE description (vulnerability behavior)
- RAG-retrieved technique descriptions
- Historical attack patterns in MITRE ATT&CK

The final output includes:
- Top-k predicted techniques
- Confidence score per technique (0–1 scale)
- Natural language justification

---

### Task 2: Justification Field (Explainability Layer)

Each prediction includes a **reasoning field** generated by the LLM. This serves two purposes:

- Improves transparency of mapping decisions
- Enables manual validation of whether reasoning aligns with MITRE definitions

**Example** (from `CVE-2020-1472`):
> *"The CVE describes an elevation of privilege vulnerability, which directly aligns with T1068. The adversary exploits a software vulnerability to elevate privileges."*

---

### Task 3: RAG (Retrieval-Augmented Generation) Integration

**Pipeline:**

1. Extract CVE description
2. Retrieve top-k relevant MITRE techniques using embedding similarity
3. Provide retrieved technique descriptions as context to LLM
4. LLM selects and ranks best matching techniques

---

### RAG Results

#### CVE-2021-21148 — Chrome V8 heap buffer overflow

| Technique ID | Technique Name | Confidence | Ground Truth | Retrieval Rank |
|---|---|---|---|---|
| T1203 | Exploitation for Client Execution | 0.95 | ✓ Hit | #4 (score 0.795) |
| T1190 | Exploit Public-Facing Application | 0.85 | Extra | #2 (score 0.800) |
| T1068 | Exploitation for Privilege Escalation | 0.60 | Extra | #1 (score 0.810) |

#### CVE-2020-1472 — Zerologon (Netlogon)

| Technique ID | Technique Name | Confidence | Ground Truth | Retrieval Rank |
|---|---|---|---|---|
| T1068 | Exploitation for Privilege Escalation | 0.95 | ✓ Hit | #1 (score 0.864) |
| T1210 | Exploitation of Remote Services | 0.85 | ✓ Hit | #3 (score 0.838) |
| T1548 | Abuse Elevation Control Mechanism | 0.80 | Extra | #5 (score 0.827) |

#### CVE-2021-21975 — VMware vRealize SSRF

| Technique ID | Technique Name | Confidence | Ground Truth | Retrieval Rank |
|---|---|---|---|---|
| T1190 | Exploit Public-Facing Application | 0.90 | ✓ Hit | #1 (score 0.824) |
| T1212 | Exploitation for Credential Access | 0.85 | Extra | #7 (score 0.805) |
| T1210 | Exploitation of Remote Services | 0.70 | Extra | #2 (score 0.823) |

---

### RAG Overall Metrics

| Metric | CVE-2021-21148 | CVE-2020-1472 | CVE-2021-21975 | Overall |
|---|---|---|---|---|
| Ground truth hits | 1 / 1 | 2 / 2 | 1 / 1 | **4 / 4** |
| Techniques returned | 3 | 3 | 3 | 9 |
| Precision | 0.33 | 0.67 | 0.33 | 0.44 |
| Accuracy | 1.0 | 1.0 | 1.0 | **1.0** |
| Hallucination rate | 0.0 | 0.0 | 0.0 | **0.0** |
| Latency (s) | 3.78 | 5.82 | 4.81 | avg 4.80 |

---

## Appendix

### Experiments with Smaller Models (Ollama & Gemini 2.5 Flash)

Additional experiments were conducted using smaller open-source models via Ollama (e.g., DeepSeek-R1) and Gemini 2.5 Flash to evaluate fully local inference. However, several challenges were observed:

- **Few-shot biasing effect:** When few-shot examples were included in the prompt, smaller models overfitted and heavily relied on them, tending to copy patterns directly rather than reasoning based on the CVE description. This resulted in biased and less generalizable mappings.
- **Performance without examples:** After removing few-shot examples to reduce bias, model performance degraded significantly. The models:
  - Struggled to identify relevant MITRE ATT&CK techniques
  - Produced hallucinated technique IDs
  - Produced valid technique IDs with incorrect names
  - Lacked consistency in mapping logic
- **Gemini timeouts:** Gemini model requests continued timing out due to high model demand and token exhaustion. Gemini results are therefore not included in the report or the JSON output files.
