#!/usr/bin/env python3
"""fetch_pass4.py — fetch the sources pass 4 of the sand survey lists, hash the bytes, extract full text, and refuse a
file that is not the listed source.

    python papers/plates/sand_survey_pass4_inputs/fetch_pass4.py <list.json> <out_dir> <fetch_record.json>

<list.json> is a JSON list of {"id", "title", "urls": [...]} — Part A with the arXiv identifiers the lab located by
title, author and year, Part B with the links the search engines returned. For each source the candidate URLs are
tried in order (an arXiv abs link is tried as its PDF first). A candidate is accepted only when it returns more than
1500 characters of extracted text AND at least 60% of the listed title's significant words appear in the first 6000
characters of that text; otherwise the next candidate is tried, and a source with no accepted candidate is
UNFETCHABLE with every attempt recorded. The title check exists because a located identifier can be wrong, and a
survey that reads the wrong paper under the right name prices nothing. The extracted texts are written to <out_dir>
and are not committed (other people's work); their sha256 are in the record, which is.
"""
import datetime as dt
import hashlib
import html
import json
import os
import re
import subprocess
import sys

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) fathom-lab sand survey pass 4 (research; github.com/fathom-lab)"
STOP = {"with", "from", "that", "this", "your", "what", "which", "their", "into", "over", "than", "when", "should",
        "does", "part", "using", "towards", "via", "for", "and", "the", "are", "not"}


def words(title):
    return [w for w in re.findall(r"[a-z0-9]+", title.lower()) if len(w) > 3 and w not in STOP]


def candidates(urls):
    out = []
    for u in urls:
        u = u.strip()
        m = re.match(r"https?://(?:export\.)?arxiv\.org/(?:abs|pdf)/([^?#]+?)(?:v\d+)?(?:\.pdf)?/?$", u)
        if m:
            pdf = f"https://arxiv.org/pdf/{m.group(1)}"
            if pdf not in out:
                out.append(pdf)
            continue
        m = re.match(r"https?://openreview\.net/forum\?id=([^&#]+)", u)
        if m:
            pdf = f"https://openreview.net/pdf?id={m.group(1)}"
            if pdf not in out:
                out.append(pdf)
        if u not in out:
            out.append(u)
    return out


def fetch(url, path):
    r = subprocess.run(["curl", "-sSL", "-A", UA, "--max-time", "120", "-o", path, "-w", "%{http_code} %{content_type} %{url_effective}", url],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    parts = (r.stdout.strip().split(" ", 2) + ["", ""])[:3]
    return parts[0], parts[1], parts[2], r.stderr.strip()


def extract(raw, path):
    if raw[:5] == b"%PDF-":
        import fitz
        doc = fitz.open(path)
        return "\n".join(p.get_text() for p in doc), "pdf", doc.page_count
    s = raw.decode("utf-8", "replace")
    s = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h\d>|</li>|</tr>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n\n", s).strip(), "html", None


def title_match(title, text):
    ws = words(title)
    head = re.sub(r"-\s*\n\s*", "", text[:6000]).lower()
    head = re.sub(r"\s+", " ", head)
    found = [w for w in ws if w in head]
    return (len(found) / len(ws)) if ws else 0.0, [w for w in ws if w not in found]


def main():
    lst = json.load(open(sys.argv[1], encoding="utf-8"))
    out_dir, record_path = sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    record = {}
    for src in lst:
        sid = src["id"]
        entry = {"title": src["title"], "attempts": [], "status": "UNFETCHABLE"}
        for i, url in enumerate(candidates(src.get("urls") or [])):
            path = os.path.join(out_dir, f"{sid}.{i}.bin")
            code, ctype, eff, err = fetch(url, path)
            raw = open(path, "rb").read() if os.path.exists(path) else b""
            att = {"url": url, "url_effective": eff, "http": code, "content_type": ctype, "bytes": len(raw),
                   "sha256": hashlib.sha256(raw).hexdigest() if raw else None,
                   "fetched_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "error": err or None}
            text = ""
            if code == "200" and len(raw) > 2000:
                try:
                    text, kind, pages = extract(raw, path)
                    att["kind"], att["pages"] = kind, pages
                except Exception as e:  # noqa: BLE001
                    att["extract_error"] = repr(e)
            frac, missing = title_match(src["title"], text) if text else (0.0, words(src["title"]))
            att["title_words_found"] = round(frac, 3)
            att["title_words_missing"] = missing
            accepted = len(text) > 1500 and frac >= 0.6
            att["accepted"] = accepted
            entry["attempts"].append(att)
            if accepted:
                tpath = os.path.join(out_dir, f"{sid}.txt")
                with open(tpath, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(text)
                entry.update({k: v for k, v in att.items() if k not in ("error",)})
                entry["fulltext"] = {"file": os.path.basename(tpath), "chars": len(text), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
                entry["status"] = "FETCHED"
                break
        if entry["status"] != "FETCHED":
            entry["reason"] = "no candidate URL returned the listed source (text too short or the title check failed)" if entry["attempts"] else "no URL to try"
        record[sid] = entry
        print(sid, entry["status"], (entry.get("fulltext") or {}).get("chars"), entry.get("title_words_found"), (entry.get("url") or "")[:80], flush=True)
    with open(record_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print("fetched", sum(1 for e in record.values() if e["status"] == "FETCHED"), "of", len(record))


if __name__ == "__main__":
    main()
