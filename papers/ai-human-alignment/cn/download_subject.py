# -*- coding: utf-8 -*-
"""Download one ds004301 subject's preprocessed BOLD + confounds + events (for the GLM)."""
import os, sys, ssl, urllib.request, urllib.parse
from xml.etree import ElementTree as ET

ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"; BASE = "https://s3.amazonaws.com/openneuro.org/"
HERE = os.path.dirname(os.path.abspath(__file__))


def s3all(prefix):
    keys = []; tok = None
    while True:
        url = f"https://s3.amazonaws.com/openneuro.org?list-type=2&prefix={urllib.parse.quote(prefix)}" + (f"&continuation-token={urllib.parse.quote(tok)}" if tok else "")
        x = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), context=ctx, timeout=60).read()
        r = ET.fromstring(x)
        for c in r.findall(ns + "Contents"):
            keys.append((c.find(ns + "Key").text, int(c.find(ns + "Size").text)))
        t = r.find(ns + "NextContinuationToken")
        if t is None:
            break
        tok = t.text
    return keys


def dl(key, out):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return os.path.getsize(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(3):
        try:
            data = urllib.request.urlopen(urllib.request.Request(BASE + urllib.parse.quote(key), headers={"User-Agent": "Mozilla/5.0"}), context=ctx, timeout=300).read()
            open(out, "wb").write(data); return len(data)
        except Exception as e:
            if attempt == 2:
                print(f"  FAIL {key}: {e}"); return 0


def main(subj):
    s = f"sub-{subj}"
    bold = [k for k, sz in s3all(f"ds004301/derivatives/preprocessed_data/{s}/func/") if k.endswith("_bold.nii.gz") or k.endswith("_desc-confounds.tsv")]
    ev = [k for k, sz in s3all(f"ds004301/{s}/func/") if k.endswith("_events.tsv")]
    print(f"{s}: {len(bold)} preprocessed files + {len(ev)} events", flush=True)
    tot = 0
    for k in ev:
        dl(k, os.path.join(HERE, "bold", s, "events", os.path.basename(k)))
    for i, k in enumerate(bold):
        n = dl(k, os.path.join(HERE, "bold", s, os.path.basename(k)))
        tot += n
        if k.endswith("_bold.nii.gz"):
            print(f"  [{i+1}/{len(bold)}] {os.path.basename(k)} ({tot/1e9:.2f} GB)", flush=True)
    print(f"{s} done: {tot/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
