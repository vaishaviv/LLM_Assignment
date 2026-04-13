import json, os, time, re
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# ── Models to run ──────────────────────────────────────────────────────────────
MODELS = [
    "gemini-2.5-flash",   # Gemini  
    "gpt-4o-mini",        # OpenAI
    "gpt-4o",             # OpenAI
    "gpt-5.4",            # OpenAI
    "mistral-medium-latest" # Mistral
]

# ── Schema ──────────────────────────────────────────────────────────────
class MitreMapping(BaseModel):
    technique_id: str
    technique_name: str
    reasoning: str

class MitreResponse(BaseModel):
    mappings: list[MitreMapping]

# ── CVEs ───────────────────────────────────────────────────────────────────────
CVES = {
    "CVE-2021-21148": "Heap buffer overflow in V8 in Google Chrome prior to 88.0.4324.150 allowed a remote attacker to potentially exploit heap corruption via a crafted HTML page.",
    "CVE-2020-1472":  "An elevation of privilege vulnerability exists when an attacker establishes a vulnerable Netlogon secure channel connection to a domain controller, using the Netlogon Remote Protocol (MS-NRPC). Also known as 'Zerologon'.",
    "CVE-2021-21975": "Server Side Request Forgery in vRealize Operations Manager API prior to 8.4 may allow a malicious actor with network access to perform an SSRF attack to steal administrative credentials."
}

SYSTEM_PROMPT = """
### ROLE
You are a Senior Cyber Threat Intelligence (CTI) Analyst. Your specialty is mapping technical vulnerabilities to the MITRE ATT&CK framework with high precision.

### OBJECTIVE
Classify the provided CVE by identifying the technical root cause and mapping it to the most specific MITRE ATT&CK Technique ID.

### REQUIRED DATA INPUTS
To classify accurately, you must identify:
1. THE VULNERABILITY: (e.g., Buffer Overflow, Use-after-free, Improper Input Validation).
2. THE VECTOR: (e.g., Network, Local, Physical).
3. THE PRIVILEGE: (e.g., Unauthenticated, User-level, SYSTEM/Root).
4. THE BEHAVIOR: (Does it allow for Execution, Persistence, or Privilege Escalation?)

### CLASSIFICATION STEPS (Chain of Thought)
Follow these steps strictly:
Step 1: Extract technical keywords from the description.
Step 2: Determine the attack behavior (e.g., code execution, privilege escalation, lateral movement).
Step 3: Match the goal and vector to a MITRE ATT&CK Technique.
Step 4: Validate if a sub-technique is more appropriate for greater specificity.
Step 5: Multi-Stage Analysis. If the CVE involves multiple distinct stages of an attack, include all relevant techniques in the JSON array, limited to the 5 most relevant ones.

### OUTPUT FORMAT
Provide your internal reasoning first to ensure accuracy, then provide the final mapping in JSON format.

[Reasoning]
<Your step-by-step analysis here>

[JSON]
[
  {{"technique_id": "TXXXX", "technique_name": "Name", "reasoning": "Explanation of why this technique was chosen"}}
]

### EXAMPLES
Good Example:
CVE-2017-0144 (EternalBlue) is an SMB Remote Code Execution.
Mapping: [{{'technique_id': 'T1210', 'technique_name': 'Exploitation of Remote Services', 'reasoning': 'EternalBlue exploits a vulnerability in a REMOTE protocol.'}}]

Good Example:
CVE: CVE-2021-34527 (PrintNightmare) allows user to run code as SYSTEM.
Mapping: [{{"technique_id": "T1068", "technique_name": "Exploration for Privilege Escalation", "reasoning": "The vulnerability allows for privilege escalation."}}]

Bad Example:
CVE-2017-0144 (EternalBlue) is an SMB Remote Code Execution.
Mapping: [{{'technique_id': 'T1059', 'technique_name': 'Command and Scripting Interpreter', 'reasoning': 'The vulnerability allows for code execution.'}}]

## INPUT DATA
CVE ID: {cve_id}
CVE Description: {description}
"""

EXCEL_URL = "https://attack.mitre.org/docs/attack-excel-files/v17.1/enterprise-attack/enterprise-attack-v17.1-techniques.xlsx"

# ── client initialisation ─────────────────────
def get_provider(model_name: str) -> str:
    if model_name.startswith("gpt"):       return "openai"
    if model_name.startswith("gemini"):    return "gemini"
    if model_name.startswith("mistral"):   return "mistral"
    raise ValueError(f"Unknown model: {model_name}")

def get_client(provider: str):
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if provider == "gemini":
        from google import genai
        return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    if provider == "mistral":
        from mistralai.client import Mistral
        return Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# cache clients so we don't re-instantiate on every CVE
_client_cache = {}
def client_for(provider):
    if provider not in _client_cache:
        _client_cache[provider] = get_client(provider)
    return _client_cache[provider]

# ── Provider adapters ──────────────────────────────────────────────────────────
# Each adapter receives the model name + formatted prompt.
# Each adapter returns (raw_text: str, mappings: list[MitreMapping])

def call_openai(model_name: str, prompt: str) -> tuple[str, list[MitreMapping]]:
    client = client_for("openai")
    response = client.beta.chat.completions.parse(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format=MitreResponse,
        temperature=0
    )
    raw = response.choices[0].message.content
    parsed = response.choices[0].message.parsed
    mappings = parsed.mappings if parsed else []
    return raw, mappings

def call_gemini(model_name: str, prompt: str) -> tuple[str, list[MitreMapping]]:
    client = client_for("gemini")
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": list[MitreMapping],
            "temperature": 0.0,
            "seed": 42
        }
    )
    raw = response.text
    mappings = response.parsed or []
    return raw, mappings

def call_mistral(model_name: str, prompt: str) -> tuple[str, list[MitreMapping]]:
    client = client_for("mistral")
    response = client.chat.complete(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
        random_seed=42
    )
    raw = response.choices[0].message.content
    # Mistral returns raw JSON string — parse it manually
    mappings = _parse_mistral_response(raw)
    return raw, mappings

def _parse_mistral_response(text: str) -> list[MitreMapping]:
    """Mistral doesn't support Pydantic natively, so we parse + validate manually."""
    try:
        # Handle both {"mappings": [...]} and bare [...] responses
        data = json.loads(text)
        items = data.get("mappings", data) if isinstance(data, dict) else data
        return [MitreMapping(**item) for item in items]
    except Exception as e:
        print(f"    ! Mistral parse error: {e}")
        return []

# ── Dispatch table — the only place provider logic lives ──────────────────────
ADAPTERS = {
    "openai":  call_openai,
    "gemini":  call_gemini,
    "mistral": call_mistral,
}

def query_model(model_name: str, prompt: str) -> tuple[str, list[MitreMapping]]:
    provider = get_provider(model_name)
    return ADAPTERS[provider](model_name, prompt)

# ── MITRE data ─────────────────────────────────────────────────────────────────
print("Loading MITRE ATT&CK v17.1...")
df = pd.read_excel(EXCEL_URL)
mitre_lookup = dict(zip(df['ID'], df['name']))
valid_ids = set(df['ID'])

def generate_mitre_link(tid: str) -> str:
    return f"https://attack.mitre.org/techniques/{tid.replace('.', '/')}/"

# ── Main loop ───────────────────────────
final_results_output = []
raw_llm_responses    = []
performance_logs     = []

for cve_id, desc in CVES.items():
    print(f"\nProcessing {cve_id}...")
    
    # Store per-model results for this CVE for the comparison table
    cve_result = {"cve": cve_id, "per_model": {}}

    for model_name in MODELS:
        print(f"  > Querying {model_name}...")
        prompt = SYSTEM_PROMPT.format(cve_id=cve_id, description=desc)

        start_time = time.time()
        try:
            raw, mappings = query_model(model_name, prompt)
        except Exception as e:
            print(f"    ! Error: {e}")
            continue
        duration = time.time() - start_time

        print(f"    Raw: {raw[:120]}...")
        print(f"    Parsed: {mappings}")

        # Validate against MITRE
        valid, hallucinations = [], []
        for m in mappings:
            tid = m.technique_id.upper().strip()
            if tid in valid_ids:
                valid.append({
                    "technique_id":   tid,
                    "technique_name": mitre_lookup[tid],
                    "reasoning":      m.reasoning,
                    "mitre_link":     generate_mitre_link(tid)
                })
            else:
                hallucinations.append(tid)

        # Raw log
        raw_llm_responses.append({
            "cve":         cve_id,
            "model":       model_name,
            "raw_content": raw
        })

        # Performance log
        performance_logs.append({
            "model":             model_name,
            "cve":               cve_id,
            "latency_seconds":   round(duration, 2),
            "valid_count":       len(valid),
            "hallucinated_ids":  hallucinations,
            "hallucination_rate": round(len(hallucinations) / len(mappings), 2) if mappings else 0
        })

        cve_result["per_model"][model_name] = valid

    final_results_output.append(cve_result)

# ── Export ─────────────────────────────────────────────────────────────────────
with open("final_results.json", "w") as f:
    json.dump(final_results_output, f, indent=2)
with open("raw_llm_output.json", "w") as f:
    json.dump(raw_llm_responses, f, indent=2)
with open("model_performance_logs.json", "w") as f:
    json.dump(performance_logs, f, indent=2)

print("\nDone. Output files written.")
