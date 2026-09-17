"""Fetch the seventeen pass-3 sources named in PROTOCOL_sand_prior_art_pass3_2026_09_14.md, save the
bytes, hash them, extract full text (PyMuPDF for PDFs, a tag-stripper for HTML), and write the fetch
record. Nothing here reads or scores; the readers do that from the extracted text."""
import datetime as dt
import hashlib
import html
import json
import os
import re
import subprocess
import sys

OUT = os.path.dirname(os.path.abspath(__file__))
SOURCES = {
    "L01": ["https://arxiv.org/pdf/2107.14203"],
    "L02": ["https://arxiv.org/pdf/2209.08443"],
    "L03": ["https://arxiv.org/pdf/2304.14106"],
    "L04": ["https://arxiv.org/pdf/2407.01235"],
    "L05": ["https://arxiv.org/pdf/2109.03228"],
    "L06": ["https://arxiv.org/pdf/1911.05248"],
    "L07": ["https://arxiv.org/pdf/2408.04667"],
    "L08": ["https://arxiv.org/pdf/2406.10229"],
    "L09": ["https://arxiv.org/pdf/2407.15847"],
    "L10": ["https://arxiv.org/pdf/2408.02871"],
    "L11": ["https://arxiv.org/pdf/2104.10706"],
    "L12": ["https://arxiv.org/pdf/2112.05588"],
    "L13": ["https://arxiv.org/pdf/1706.10268"],
    "L14": ["https://arxiv.org/pdf/1909.01838"],
    "L15": ["https://www.aisnakeoil.com/p/is-gpt-4-getting-worse-over-time"],
    "L16": ["https://www.acm.org/publications/policies/artifact-review-and-badging-current"],
    "L17": ["https://zenodo.org/records/3818627/files/article.pdf",
            "https://rescience.github.io/bibliography/Sinha_2020.html"],
}
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) fathom-lab sand survey pass 3 (research; contact via github.com/fathom-lab)"


def fetch(url, path):
    hdr = path + ".headers"
    r = subprocess.run(["curl", "-sSL", "-A", UA, "--max-time", "120", "-o", path, "-D", hdr,
                        "-w", "%{http_code} %{content_type} %{url_effective}", url],
                       capture_output=True, text=True)
    code, ctype, eff = (r.stdout.strip().split(" ", 2) + ["", ""])[:3]
    return code, ctype, eff, r.stderr.strip()


def pdf_text(path):
    import fitz
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc), doc.page_count


def html_text(raw):
    s = raw.decode("utf-8", "replace")
    s = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h\d>|</li>|</tr>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()


record = {}
for sid, urls in SOURCES.items():
    entry = {"attempts": []}
    for url in urls:
        ext = ".pdf" if url.endswith(".pdf") or "arxiv.org/pdf" in url else ".html"
        path = os.path.join(OUT, sid + ext)
        code, ctype, eff, err = fetch(url, path)
        raw = open(path, "rb").read() if os.path.exists(path) else b""
        att = {"url": url, "url_effective": eff, "http": code, "content_type": ctype, "bytes": len(raw),
               "sha256": hashlib.sha256(raw).hexdigest() if raw else None,
               "fetched_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "error": err or None}
        entry["attempts"].append(att)
        ok = code == "200" and len(raw) > 2000
        if ok:
            try:
                if raw[:5] == b"%PDF-":
                    text, pages = pdf_text(path)
                    att["kind"] = "pdf"; att["pages"] = pages
                else:
                    text = html_text(raw); att["kind"] = "html"
            except Exception as e:  # pragma: no cover
                att["extract_error"] = repr(e); text = ""
            if len(text) > 1500:
                tpath = os.path.join(OUT, sid + ".txt")
                open(tpath, "w", encoding="utf-8", newline="\n").write(text)
                att["fulltext"] = {"file": os.path.basename(tpath), "chars": len(text),
                                   "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
                entry.update({k: v for k, v in att.items() if k != "error"})
                entry["status"] = "FETCHED"
                break
    entry.setdefault("status", "UNFETCHABLE")
    record[sid] = entry
    print(sid, entry["status"], entry.get("http"), entry.get("bytes"), (entry.get("fulltext") or {}).get("chars"), entry.get("kind"))

json.dump(record, open(os.path.join(OUT, "fetch_record_pass3.json"), "w", encoding="utf-8"), indent=1)
print("fetched:", sum(1 for e in record.values() if e["status"] == "FETCHED"), "of", len(record))
