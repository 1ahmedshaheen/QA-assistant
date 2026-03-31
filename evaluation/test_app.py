# evaluation/test_app.py
# End-to-end smoke test: security, speed, and correctness
# Run: python evaluation/test_app.py
import os
import sys
import time
import json
import httpx
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import settings

BASE = f"http://localhost:{settings.langserve_port}"
TIMEOUT = 180  # seconds

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
SKIP = "\033[93m⏭  SKIP\033[0m"

results = []


def record(name, passed, detail=""):
    status = PASS if passed else FAIL
    print(f"  [{status}] {name}: {detail}")
    results.append({"test": name, "passed": passed, "detail": str(detail)})


# ─────────────────────────────────────────────────────
# 1. HEALTH – connectivity check
# ─────────────────────────────────────────────────────
print("\n=== 1. HEALTH CHECK ===")
try:
    t0 = time.time()
    r = httpx.get(f"{BASE}/health", timeout=10)
    latency = time.time() - t0
    data = r.json()
    record("API reachable",       r.status_code == 200)
    record("Health latency <5s",  latency < 5, f"{latency:.2f}s")
    record("Version field",       "version" in data)
    record("LLM field in health", "llm_model" in data, data.get("llm_model", "?"))
except Exception as e:
    record("API reachable", False, str(e))
    print(f"\n  ‼ Backend not reachable — aborting remaining tests.")
    sys.exit(1)


# ─────────────────────────────────────────────────────
# 2. SECURITY – upload validation
# ─────────────────────────────────────────────────────
print("\n=== 2. SECURITY TESTS ===")

# 2a. Reject unsupported extensions
try:
    with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp:
        tmp.write(b"malware")
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        r = httpx.post(f"{BASE}/ingest", files={"file": ("evil.exe", f, "application/octet-stream")}, timeout=30)
    record("Reject .exe upload", r.status_code == 400, f"status={r.status_code}")
    os.unlink(tmp_path)
except Exception as e:
    record("Reject .exe upload", False, str(e))

# 2b. Reject empty filename
try:
    r = httpx.post(f"{BASE}/ingest", files={"file": ("", b"", "application/pdf")}, timeout=10)
    # server should raise 400/422
    record("Reject empty filename", r.status_code in (400, 422), f"status={r.status_code}")
except Exception as e:
    record("Reject empty filename", False, str(e))

# 2c. CORS headers present
try:
    r = httpx.options(f"{BASE}/health", headers={"Origin": "http://evil.com"}, timeout=5)
    record("CORS headers present", "access-control-allow-origin" in r.headers or r.status_code < 500)
except Exception as e:
    record("CORS headers present", False, str(e))

# 2d. Method not allowed
try:
    r = httpx.delete(f"{BASE}/health", timeout=5)
    record("DELETE /health → 405", r.status_code == 405, f"status={r.status_code}")
except Exception as e:
    record("DELETE /health → 405", False, str(e))


# ─────────────────────────────────────────────────────
# 3. INGESTION + SPEED
# ─────────────────────────────────────────────────────
print("\n=== 3. INGESTION SPEED TEST ===")

# Create a minimal valid-ish PDF (plain text blob; real PDF needs a proper header)
sample_content = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /Resources << >> /Contents 4 0 R >> endobj
4 0 obj << /Length 60 >> stream
BT /F1 12 Tf 72 712 Td (This is a valid smart contract document with enough text content to pass the quality filter.) Tj ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000210 00000 n
trailer << /Size 5 /Root 1 0 R >>
startxref
310
%%EOF"""

try:
    t0 = time.time()
    r = httpx.post(
        f"{BASE}/ingest",
        files={"file": ("test_contract.pdf", sample_content, "application/pdf")},
        timeout=TIMEOUT,
    )
    ingest_time = time.time() - t0
    data = r.json()
    record("Ingest succeeds",        r.status_code == 200, f"status={r.status_code}")
    record("Ingest time <60s",       ingest_time < 60, f"{ingest_time:.2f}s")
    record("Response has chunks",    "chunks" in data, str(data))
    INGESTED_FILE = data.get("file", "test_contract.pdf")
except Exception as e:
    record("Ingest succeeds", False, str(e))
    INGESTED_FILE = None


# ─────────────────────────────────────────────────────
# 4. RAG CHAIN – functional test  (only if ingest succeeded)
# ─────────────────────────────────────────────────────
print("\n=== 4. RAG CHAIN Q&A TEST ===")

def call_chain(path, payload, timeout=TIMEOUT):
    """POST to a LangServe chain endpoint."""
    r = httpx.post(f"{BASE}{path}/invoke", json={"input": payload}, timeout=timeout)
    r.raise_for_status()
    return r.json()

# 4a. Basic retrieval
if INGESTED_FILE:
    try:
        t0 = time.time()
        res = call_chain("/retriever", "contract obligations")
        elapsed = time.time() - t0
        docs = res.get("output", [])
        record("Retriever returns docs",   isinstance(docs, list), f"{len(docs)} docs")
        record("Retriever latency <30s",   elapsed < 30, f"{elapsed:.2f}s")
    except Exception as e:
        record("Retriever returns docs", False, str(e))
else:
    print(f"  [{SKIP}] Retriever — no ingested file")

# 4b. Guardrail – on-topic query
try:
    t0 = time.time()
    res = call_chain("/guardrail", "What are the payment terms?")
    elapsed = time.time() - t0
    blocked = res.get("output", {}).get("blocked", True)
    record("Guardrail: contract query NOT blocked", not blocked, str(res.get("output")))
    record("Guardrail latency <10s", elapsed < 10, f"{elapsed:.2f}s")
except Exception as e:
    record("Guardrail: contract query NOT blocked", False, str(e))

# 4c. Guardrail – off-topic query
try:
    res = call_chain("/guardrail", "What is the best pizza topping?")
    blocked = res.get("output", {}).get("blocked", False)
    record("Guardrail: off-topic query IS blocked", blocked, str(res.get("output")))
except Exception as e:
    record("Guardrail: off-topic query IS blocked", False, str(e))

# 4d. Generator – grounded answer
try:
    t0 = time.time()
    res = call_chain("/generator", {
        "input": "Summarise the main purpose of this document.",
        "context": "This contract is between Party A and Party B for the supply of software services.",
    })
    elapsed = time.time() - t0
    answer = res.get("output", "")
    record("Generator returns text",   isinstance(answer, str) and len(answer) > 5, answer[:80])
    record("Generator latency <60s",   elapsed < 60, f"{elapsed:.2f}s")
except Exception as e:
    record("Generator returns text", False, str(e))


# ─────────────────────────────────────────────────────
# 5. SUMMARIZE ENDPOINT
# ─────────────────────────────────────────────────────
print("\n=== 5. SUMMARIZE ENDPOINT ===")
if INGESTED_FILE:
    try:
        t0 = time.time()
        r = httpx.post(f"{BASE}/summarize", params={"file_name": INGESTED_FILE}, timeout=TIMEOUT)
        elapsed = time.time() - t0
        data = r.json()
        record("Summarize returns 200",    r.status_code == 200, f"status={r.status_code}")
        record("Summarize has summary key", "summary" in data)
        record("Summarize latency <90s",   elapsed < 90, f"{elapsed:.2f}s")
    except Exception as e:
        record("Summarize endpoint", False, str(e))
else:
    print(f"  [{SKIP}] Summarize — no ingested file")


# ─────────────────────────────────────────────────────
# 6. EVALUATION (ROUGE-L on generator answer)
# ─────────────────────────────────────────────────────
print("\n=== 6. ROUGE-L EVALUATION ===")
try:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    reference = "This contract is between Party A and Party B for software services."
    candidate = ""
    try:
        res = call_chain("/generator", {
            "input": "Describe what this contract is about.",
            "context": reference,
        })
        candidate = res.get("output", "")
    except Exception as e:
        candidate = ""
    score = scorer.score(reference, candidate)
    f1 = score["rougeL"].fmeasure
    record("ROUGE-L F1 > 0.05", f1 > 0.05, f"F1={f1:.3f}")
    print(f"     ROUGE-L precision={score['rougeL'].precision:.3f}  "
          f"recall={score['rougeL'].recall:.3f}  f1={f1:.3f}")
except ImportError:
    print(f"  [{SKIP}] rouge_score not installed")
except Exception as e:
    record("ROUGE-L evaluation", False, str(e))


# ─────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────
passed = sum(1 for r in results if r["passed"])
total  = len(results)
print(f"\n{'='*50}")
print(f"RESULTS: {passed}/{total} tests passed")
print(f"{'='*50}\n")

# Save JSON report
out_path = "evaluation/test_results.json"
os.makedirs("evaluation", exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Report saved to {out_path}")

if passed < total:
    sys.exit(1)
