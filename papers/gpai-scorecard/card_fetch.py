"""G5 card fetcher -- freeze the population: fetch each signatory's flagship doc, pin it, convert to text.

PREREG: papers/gpai-scorecard/PREREG_gpai_scorecard_2026_07_14.md (frozen BEFORE this ran on any doc).
Population authority: the Commission signatory list (page-update 2026-04-23), 24 signatories.
Doc rule (frozen): provider-published model/system card (or technical report where that is the
provider's primary eval document) for its most capable GPAI model as of the freeze date; format
preference HF raw README.md > provider PDF > provider HTML.

For each provider this script tries its candidate URLs in the frozen preference order, records
EVERY attempt (status, bytes), pins the winning fetch (sha256 of raw bytes + of extracted text),
converts to text (md passthrough / pypdf / html2text -- converter versions recorded), and writes:
  - raw bytes + extracted .txt under _cards/   (gitignored -- license/copyright, never committed)
  - CARD_MANIFEST.json                          (committed -- the frozen population)

Resolver fallback: if every candidate for an HF-hosted provider fails, query the HF API for the
author's most-downloaded text-generation model and fetch its README (choice + rule recorded).
Providers with no locatable doc land in stratum NO-PUBLIC-FLAGSHIP-DOC, reported never blamed.

Usage: python papers/gpai-scorecard/card_fetch.py
"""
from __future__ import annotations
import hashlib, importlib.metadata, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
CARDS = HERE / "_cards"
MANIFEST = HERE / "CARD_MANIFEST.json"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) fathom-lab-styxx-g5-scorecard/1.0 (receipt-binding research; contact via github.com/fathom-lab/styxx)"}
TIMEOUT = 60

# ---- the frozen population: 24 Commission-listed signatories (page-update 2026-04-23) ----
# candidates in the prereg's frozen preference order: HF raw > provider PDF > provider HTML
POPULATION = [
    {"provider": "OpenAI", "signed": "full", "candidates": [
        ("pdf", "https://deploymentsafety.openai.com/gpt-5-5/gpt-5-5.pdf"),
        ("html", "https://openai.com/index/gpt-5-5-system-card/")]},
    {"provider": "Anthropic", "signed": "full", "candidates": [
        ("pdf", "https://www-cdn.anthropic.com/14e4fb01875d2a69f646fa5e574dea2b1c0ff7b5.pdf"),
        ("html", "https://www.anthropic.com/claude-opus-4-5-system-card")]},
    {"provider": "Google", "signed": "full", "candidates": [
        ("pdf", "https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-1-Pro-Model-Card.pdf"),
        ("pdf", "https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf"),
        ("html", "https://deepmind.google/models/model-cards/gemini-3-pro/")]},
    {"provider": "Microsoft", "signed": "full", "hf_author": "microsoft", "candidates": [
        ("md", "https://huggingface.co/microsoft/phi-4/raw/main/README.md"),
        ("html", "https://huggingface.co/microsoft/phi-4")]},
    {"provider": "Amazon", "signed": "full", "candidates": [
        ("pdf", "https://assets.amazon.science/96/7d/0d3e59514abf8fdcfafcdc574300/nova-tech-report-20250317-0810.pdf"),
        ("html", "https://www.amazon.science/publications/the-amazon-nova-family-of-models-technical-report-and-model-card")]},
    {"provider": "IBM", "signed": "full", "hf_author": "ibm-granite", "candidates": [
        ("md", "https://huggingface.co/ibm-granite/granite-4.1-8b/raw/main/README.md"),
        ("md", "https://huggingface.co/ibm-granite/granite-4.0-h-small/raw/main/README.md")]},
    {"provider": "Mistral AI", "signed": "full", "hf_author": "mistralai", "candidates": [
        ("md", "https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512/raw/main/README.md"),
        ("html", "https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512")]},
    {"provider": "Aleph Alpha", "signed": "full", "hf_author": "Aleph-Alpha", "candidates": [
        ("md", "https://huggingface.co/Aleph-Alpha/Pharia-1-LLM-7B-control/raw/main/README.md")]},
    {"provider": "Cohere", "signed": "full", "hf_author": "CohereLabs", "candidates": [
        ("md", "https://huggingface.co/CohereLabs/c4ai-command-a-03-2025/raw/main/README.md"),
        ("html", "https://huggingface.co/CohereLabs/c4ai-command-a-03-2025")]},
    {"provider": "xAI", "signed": "partial-safety-only", "candidates": [
        ("pdf", "https://data.x.ai/2025-11-17-grok-4-1-model-card.pdf"),
        ("pdf", "https://data.x.ai/2025-08-20-grok-4-model-card.pdf")]},
    {"provider": "ServiceNow", "signed": "full", "hf_author": "ServiceNow-AI", "candidates": [
        ("md", "https://huggingface.co/ServiceNow-AI/Apriel-1.5-15b-Thinker/raw/main/README.md")]},
    {"provider": "WRITER", "signed": "full", "candidates": [
        ("html", "https://dev.writer.com/home/models")]},
    {"provider": "Black Forest Labs", "signed": "full", "hf_author": "black-forest-labs", "candidates": [
        ("md", "https://huggingface.co/black-forest-labs/FLUX.1-dev/raw/main/README.md")]},
    {"provider": "Almawave", "signed": "full", "hf_author": "Almawave", "candidates": [
        ("md", "https://huggingface.co/Almawave/Velvet-14B/raw/main/README.md")]},
    {"provider": "Fastweb", "signed": "full", "hf_author": "Fastweb", "candidates": [
        ("md", "https://huggingface.co/Fastweb/FastwebMIIA-7B/raw/main/README.md")]},
    {"provider": "LINAGORA", "signed": "full", "hf_author": "OpenLLM-France", "candidates": [
        ("md", "https://huggingface.co/OpenLLM-France/Lucie-7B-Instruct/raw/main/README.md")]},
    {"provider": "Domyn", "signed": "full", "hf_author": "iGeniusAI", "candidates": [
        ("html", "https://build.nvidia.com/igenius/colosseum_355b_instruct_16k/modelcard"),
        ("md", "https://huggingface.co/iGeniusAI/Italia-9B-Instruct-v0.1/raw/main/README.md"),
        ("html", "https://huggingface.co/iGeniusAI/Italia-9B-Instruct-v0.1")]},
    {"provider": "Pleias", "signed": "full", "hf_author": "PleIAs", "candidates": []},   # resolver
    {"provider": "Bria AI", "signed": "full", "hf_author": "briaai", "candidates": [
        ("md", "https://huggingface.co/briaai/BRIA-3.2/raw/main/README.md")]},
    {"provider": "Accexible", "signed": "full", "candidates": []},
    {"provider": "AI Studio Delta", "signed": "full", "candidates": []},
    {"provider": "Dweve", "signed": "full", "candidates": []},
    {"provider": "Lawise", "signed": "full", "candidates": []},
    {"provider": "Open Hippo", "signed": "full", "candidates": []},
]


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(url: str):
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT, allow_redirects=True)
        return r.status_code, (r.content if r.status_code == 200 else b""), None
    except Exception as e:
        return -1, b"", str(e)[:200]


def pdf_to_text(raw: bytes) -> str:
    import io
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(raw))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def html_to_text(raw: bytes) -> str:
    import html2text
    h = html2text.HTML2Text()
    h.ignore_links = False        # links are the CONSTRUCT (binding = linkage); never drop them
    h.ignore_images = True
    h.body_width = 0
    return h.handle(raw.decode("utf-8", errors="replace"))


def hf_resolve(author: str):
    """Fallback: the author's most-downloaded model on HF (choice recorded, rule disclosed)."""
    code, raw, err = fetch(f"https://huggingface.co/api/models?author={author}&sort=downloads&direction=-1&limit=10")
    if code != 200:
        return None
    try:
        models = json.loads(raw)
    except Exception:
        return None
    if not models:
        return None
    return models[0].get("modelId") or models[0].get("id")


def convert(fmt: str, raw: bytes, versions: dict):
    if fmt == "md":
        return raw.decode("utf-8", errors="replace"), "passthrough"
    if fmt == "pdf":
        return pdf_to_text(raw), f"pypdf {versions['pypdf']}"
    return html_to_text(raw), f"html2text {versions['html2text']}"


def try_candidate(fmt, url, versions, attempts):
    """Fetch + convert; a candidate WINS only if the EXTRACTED TEXT is substantive
    (>2000 chars) -- JS-shell HTML pages (big raw, empty text) fall through."""
    code, raw, err = fetch(url)
    rec = {"url": url, "format": fmt, "status": code, "bytes": len(raw), "error": err}
    if code == 200 and len(raw) > 2000:
        try:
            text, converter = convert(fmt, raw, versions)
        except Exception as e:
            rec["error"] = f"convert-failed: {str(e)[:150]}"
            attempts.append(rec); return None
        rec["text_chars"] = len(text)
        if len(text) > 2000:
            attempts.append(rec)
            return (fmt, url, raw, text, converter)
        rec["error"] = "extracted text too small (JS shell or empty doc)"
    attempts.append(rec)
    time.sleep(1)
    return None


def main() -> int:
    CARDS.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    versions = {p: importlib.metadata.version(p) for p in ("requests", "pypdf", "html2text")}
    entries = {}
    for spec in POPULATION:
        prov = spec["provider"]
        attempts = []
        won = None
        candidates = list(spec["candidates"])
        resolved_note = None
        if not candidates and spec.get("hf_author"):
            mid = hf_resolve(spec["hf_author"])
            if mid:
                resolved_note = f"resolved via HF API most-downloaded for author {spec['hf_author']}: {mid}"
                candidates = [("md", f"https://huggingface.co/{mid}/raw/main/README.md"),
                              ("html", f"https://huggingface.co/{mid}")]
        for fmt, url in candidates:
            won = try_candidate(fmt, url, versions, attempts)
            if won:
                break
        if won is None and spec.get("hf_author") and spec["candidates"]:
            mid = hf_resolve(spec["hf_author"])
            if mid:
                resolved_note = f"candidates failed; resolved via HF API most-downloaded for author {spec['hf_author']}: {mid}"
                for fmt, url in [("md", f"https://huggingface.co/{mid}/raw/main/README.md"),
                                 ("html", f"https://huggingface.co/{mid}")]:
                    won = try_candidate(fmt, url, versions, attempts)
                    if won:
                        break
        entry = {"signed": spec["signed"], "attempts": attempts, "resolved": resolved_note,
                 "fetched_utc": now}
        if won is None:
            entry["stratum_candidate"] = "NO-PUBLIC-FLAGSHIP-DOC"
            entry["doc"] = None
            print(f"[{prov}] NO DOC ({len(attempts)} attempts)", flush=True)
        else:
            fmt, url, raw, text, converter = won
            slug = prov.lower().replace(" ", "_")
            ext = {"md": "md", "pdf": "pdf", "html": "html"}[fmt]
            raw_path = CARDS / f"{slug}.{ext}"
            raw_path.write_bytes(raw)
            txt_path = CARDS / f"{slug}.txt"
            txt_path.write_text(text, encoding="utf-8", newline="\n")
            entry["stratum_candidate"] = "PARTIAL-SIGNATORY" if spec["signed"].startswith("partial") else "FETCHED"
            entry["doc"] = {"url": url, "format": fmt, "raw_sha256": sha256(raw), "raw_bytes": len(raw),
                            "converter": converter, "text_sha256": sha256(text.encode("utf-8")),
                            "text_chars": len(text), "raw_file": raw_path.name, "text_file": txt_path.name}
            print(f"[{prov}] {fmt} {len(raw)}b -> {len(text)}c  {url}", flush=True)
    # Meta contrast arm: rung-1 receipts, never re-fetched here (frozen at ae51f0f / oath-economy)
        entries[prov] = entry

    manifest = {
        "_what": "G5 frozen population manifest -- EU GPAI Code-of-Practice signatories' flagship docs",
        "_prereg": "papers/gpai-scorecard/PREREG_gpai_scorecard_2026_07_14.md",
        "_authority": "https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai (page-update 2026-04-23; 24 signatories)",
        "_doc_rule": "flagship model/system card or primary eval tech report; HF raw > PDF > HTML; no swaps after freeze",
        "_texts_not_committed": "raw+text live in _cards/ (gitignored); sha256 pins re-verification",
        "_meta_contrast_arm": "papers/oath-economy/ rung-1 receipts (201 claims, 0 BOUND), reused frozen -- never headline",
        "frozen_utc": now, "tool_versions": versions,
        "providers": entries}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    n_doc = sum(1 for e in entries.values() if e["doc"])
    print(f"\nMANIFEST FROZEN: {n_doc} docs fetched of {len(entries)} signatories -> {MANIFEST.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
