import json
import os
import time
import numpy as np
import pandas as pd
from mistralai.client import Mistral
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# ==============================
# CONFIG
# ==============================
ATTACK_FILE   = "enterprise-attack-v17.1-techniques.xlsx"
EMBED_MODEL   = "mistral-embed" 
MISTRAL_MODEL = "mistral-medium-latest"
TOP_K         = 8

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)

CVES = {
    "CVE-2021-21148": "Heap buffer overflow in V8 in Google Chrome prior to 88.0.4324.150 allowed a remote attacker to potentially exploit heap corruption via a crafted HTML page.",
    "CVE-2020-1472":  "An elevation of privilege vulnerability exists when an attacker establishes a vulnerable Netlogon secure channel connection to a domain controller, using the Netlogon Remote Protocol (MS-NRPC). Also known as 'Zerologon'.",
    "CVE-2021-21975": "Server Side Request Forgery in vRealize Operations Manager API prior to 8.4 may allow a malicious actor with network access to perform an SSRF attack to steal administrative credentials."
}

# ==============================
# DATA LOADING
# ==============================
def load_attack_data(path: str) -> list[dict]:
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    return [
        {
            "technique_id": row["ID"],
            "name":         row["name"],
            "description":  str(row["description"])
        }
        for _, row in df.iterrows()
    ]

# ==============================
# EMBEDDING
# ==============================
def get_embedding(text: str) -> np.ndarray:
    """Embed a single string using Mistral API."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        inputs=[text] # Mistral expects a list
    )
    return np.array(response.data[0].embedding, dtype="float32")

def build_embeddings(techniques: list[dict], batch_size: int = 50) -> np.ndarray:
    """Embed all techniques by chunking into batches to avoid Mistral API token limits."""
    texts = [f"{t['name']}. {t['description']}" for t in techniques]
    all_embeddings = []

    print(f"  Embedding {len(texts)} techniques in batches of {batch_size}...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = client.embeddings.create(
                model=EMBED_MODEL,
                inputs=batch
            )
            # Extract embeddings from this batch
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # Optional: Add a tiny sleep if you hit rate limits (429 errors)
            # time.sleep(0.1) 
            
            print(f"    Processed {i + len(batch)}/{len(texts)}...")
        except Exception as e:
            print(f"    ! Error processing batch starting at index {i}: {e}")
            raise e

    return np.array(all_embeddings, dtype="float32")

# ==============================
# RETRIEVAL
# ==============================
def retrieve_top_k(query: str,
                   techniques: list[dict],
                   embeddings: np.ndarray,
                   k: int = TOP_K) -> list[dict]:
    query_vec = get_embedding(query).reshape(1, -1)
    scores    = cosine_similarity(query_vec, embeddings)[0]
    top_idx   = np.argsort(scores)[-k:][::-1]
    return [
        {
            "technique_id": techniques[i]["technique_id"],
            "name":         techniques[i]["name"],
            "description":  techniques[i]["description"],
            "score":        float(scores[i])
        }
        for i in top_idx
    ]

# ==============================
# LLM CALL
# ==============================
def ask_mistral(cve_desc: str, candidates: list[dict]) -> str | None:
    candidate_text = "\n".join([
        f"{i+1}. {c['technique_id']} - {c['name']}: {c['description'][:200]}"
        for i, c in enumerate(candidates)
    ])

    prompt = f"""
You are a Cybersecurity Analyst and MITRE ATT&CK Mapping Expert.

CVE Description: {cve_desc}

Based on the CVE above, evaluate the candidate techniques below and select
ALL that are relevant. For each match, explain WHY it fits.

Candidates:
{candidate_text}

Return a JSON object in this exact format:
{{
  "candidates": [
    {{
      "technique_id":   "TXXXX",
      "technique_name": "Name",
      "confidence":     0.0,
      "reasoning":      "Why this technique matches the CVE mechanism."
    }}
  ]
}}

Rules:
- confidence must be a float between 0.0 and 1.0 (e.g. 0.85).
- Do NOT assume or infer information from other sources, ONLY use information from the retrieved candidates.
- Rank by confidence descending.
- Omit techniques that do not clearly match.
"""

    try:
        response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            random_seed=42
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"    ! Mistral API error: {e}")
        return None

# ==============================
# VALIDATION
# ==============================
def validate_output(llm_output: str, valid_ids: set) -> tuple[list[dict], list[str]]:
    """
    Parse the LLM JSON and check each technique_id against valid MITRE IDs.
    Returns (validated_candidates, hallucinated_ids).
    """
    if not llm_output:
        return [], []
    try:
        parsed = json.loads(llm_output)

        # handle both {"candidates": [...]} and a bare list
        items = parsed.get("candidates", parsed) if isinstance(parsed, dict) else parsed
        if isinstance(items, dict):
            items = [items]  # single object fallback

        valid, hallucinations = [], []
        for c in items:
            tid = str(c.get("technique_id", "")).upper().strip()
            if tid in valid_ids:
                c["technique_id"] = tid   # normalise casing
                c["is_valid"]     = True
                valid.append(c)
            else:
                c["is_valid"] = False
                hallucinations.append(tid)

        return valid, hallucinations

    except Exception as e:
        print(f"    ! Parse error: {e}")
        return [], []

# ==============================
# MAIN PIPELINE
# ==============================
def run_rag():
    # ── Load MITRE data ───────────────────────────────────────────────
    print("Loading MITRE ATT&CK data...")
    techniques = load_attack_data(ATTACK_FILE)
    valid_ids  = set(t["technique_id"] for t in techniques)
    print(f"  Loaded {len(techniques)} techniques.")

    # ── Build / load embedding cache ──────────────────────────────────
    # Cache filename includes model name so switching models
    # never silently reuses stale embeddings
    safe_model_name = EMBED_MODEL.replace(":", "_").replace("/", "_")
    cache_file      = f"embeddings_{safe_model_name}.npy"

    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from '{cache_file}'...")
        tech_embeddings = np.load(cache_file)
    else:
        print("Building embeddings (one-time, batched)...")
        tech_embeddings = build_embeddings(techniques)
        np.save(cache_file, tech_embeddings)
        print(f"  Saved to '{cache_file}'.")

    print(f"  Embedding matrix: {tech_embeddings.shape}")

    # ── Per-CVE pipeline ──────────────────────────────────────────────
    results = []

    for cve_id, desc in CVES.items():
        print(f"\n{'='*60}")
        print(f"Processing {cve_id}")

        # Enrich the query to improve retrieval relevance
        query = (
            f"Vulnerability description: {desc}\n"
            f"Focus on: attack vector, exploitation method, and impact."
        )

        # 1. Retrieve top-k candidates
        retrieved = retrieve_top_k(query, techniques, tech_embeddings, TOP_K)
        print("  Retrieved candidates:")
        for r in retrieved:
            print(f"    {r['technique_id']:<12} {r['name']:<45} score: {r['score']:.3f}")

        # 2. LLM decision over retrieved candidates
        print("  Querying Mistral...")
        t0         = time.time()
        llm_output = ask_mistral(desc, retrieved)
        latency    = round(time.time() - t0, 2)
        print(f"  Raw output: {llm_output}")
        print(f"  Latency: {latency}s")

        # 3. Validate output against MITRE
        valid_techniques, hallucinations = validate_output(llm_output, valid_ids)

        print(f"  Valid techniques : {[t['technique_id'] for t in valid_techniques]}")
        print(f"  Hallucinations   : {hallucinations}")

        results.append({
            "cve":                 cve_id,
            "retrieved_candidates": retrieved,
            "llm_output":          llm_output,
            "valid_techniques":    valid_techniques,
            "hallucinations":      hallucinations,
            "hallucination_rate":  round(
                len(hallucinations) / (len(valid_techniques) + len(hallucinations)), 2
            ) if (valid_techniques or hallucinations) else 0,
            "latency_seconds":     latency
        })

    # ── Export ────────────────────────────────────────────────────────
    with open("rag_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Done. Results saved to rag_results.json")

if __name__ == "__main__":
    run_rag()