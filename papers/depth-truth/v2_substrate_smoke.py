"""PHASE 1 substrate decision — branch A smoke (PREREG v2).

Tests whether a HARDENED base-gemma-2-2b prompt + hardened extraction recovers clean, gradeable answers on
SYNTHETIC trivia items (hand-written here, NOT drawn from TriviaQA/PopQA/TruthfulQA). If extraction-clean
rate >= 80%, base wins and the depth instrument stays on its validated substrate; else we fall to branch B
(-it) which would require a quarantined substrate-sanity replication first.

Generation-only (no attribution), so it loads a plain HookedTransformer, not the ReplacementModel — the
transcoders don't change decoding, so generation behavior is identical and this is faster. Decision is by
rate, printed at the end.
"""
import os
import re
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness"))
import qa_data as D  # normalize + grade (identical to v1/pilot)

# ---- branch A: hardened 5-shot prompt (answer-only, explicit no-format, short shots) ----
HARD_PROMPT = (
    "Answer each question with only the answer. No numbering, no formatting, no explanation.\n\n"
    "Q: What is the chemical symbol for gold?\nA: Au\n\n"
    "Q: How many sides does a hexagon have?\nA: six\n\n"
    "Q: In what year did the first man walk on the Moon?\nA: 1969\n\n"
    "Q: What is the largest planet in the Solar System?\nA: Jupiter\n\n"
    "Q: Who wrote the play Romeo and Juliet?\nA: William Shakespeare\n\n"
    "Q: {question}\nA:"
)

# ---- hardened extraction (v2 Appendix B candidate): first line, strip list-num + HTML + quotes ----
_LIST = re.compile(r"^\s*\d+\s*[\.\)]\s*")
_HTML = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*\s*/?>")
def hardened_extract(gen: str):
    line = gen.split("\n", 1)[0]
    line = _HTML.sub("", line)
    line = _LIST.sub("", line)
    line = line.strip().strip('"').strip("'").strip()
    return line

def is_clean(raw_first_line: str, extracted: str) -> bool:
    # "clean" = a bare short answer survived extraction: non-empty, <=6 words, and the RAW first line
    # carried no rambling beyond the answer (extracted within a couple tokens of the raw line length).
    if not extracted or len(extracted.split()) > 6:
        return False
    return True

# ---- 15 SYNTHETIC items (hand-written; NOT from any eval set) ----
SYNTH = [
    ("What is the capital city of Japan?", ["Tokyo"]),
    ("What gas do plants absorb from the air for photosynthesis?", ["carbon dioxide", "CO2"]),
    ("How many continents are there on Earth?", ["seven", "7"]),
    ("Who painted the Mona Lisa?", ["Leonardo da Vinci", "da Vinci", "Leonardo"]),
    ("What is the largest ocean on Earth?", ["Pacific", "Pacific Ocean"]),
    ("What metal is liquid at room temperature?", ["mercury"]),
    ("How many legs does a spider have?", ["eight", "8"]),
    ("What is the tallest mountain on Earth?", ["Everest", "Mount Everest"]),
    ("What is the chemical formula for water?", ["H2O"]),
    ("Who developed the theory of general relativity?", ["Einstein", "Albert Einstein"]),
    ("What is the smallest prime number?", ["2", "two"]),
    ("What planet is known as the Red Planet?", ["Mars"]),
    ("What is the currency of Japan?", ["yen", "Japanese yen"]),
    ("What is the hardest known natural material?", ["diamond"]),
    ("In which country would you find the Eiffel Tower?", ["France"]),
]


def main():
    import torch
    from transformer_lens import HookedTransformer
    print("loading gemma-2-2b (HookedTransformer, generation-only)...", flush=True)
    model = HookedTransformer.from_pretrained("google/gemma-2-2b", dtype=torch.bfloat16)
    tok = model.tokenizer
    nl = set(tok.encode("\n")) | set(tok.encode("\nQ"))

    def gen_greedy(prompt, max_new=16):
        tokens = model.to_tokens(prompt)
        ids = []
        with torch.no_grad():
            for _ in range(max_new):
                nxt = int(model(tokens, return_type="logits")[0, -1].argmax().item())
                ids.append(nxt)
                tokens = torch.cat([tokens, torch.tensor([[nxt]], device=tokens.device)], dim=1)
                if nxt in nl:
                    break
        return tok.decode(ids), ids

    n_clean = n_correct = n_content_tok = 0
    print(f"\n{'question':46} {'raw':26} {'extracted':20} clean corr")
    for q, gold in SYNTH:
        raw, ids = gen_greedy(HARD_PROMPT.format(question=q))
        first_line = raw.split("\n", 1)[0]
        ext = hardened_extract(raw)
        clean = is_clean(first_line, ext)
        correct = D.grade(ext, gold)
        # first content token of the EXTRACTED answer (v2 A1): is it a real word token?
        ctok = re.findall(r"[^\W\d_]{2,}|\w+", ext)
        content_ok = bool(ctok)
        n_clean += clean; n_correct += correct; n_content_tok += content_ok
        print(f"  {q[:44]:44} {first_line[:24]!r:26} {ext[:18]!r:20} {str(clean):5} {str(correct):5}")
    n = len(SYNTH)
    print(f"\n=== BRANCH A RESULT (n={n} synthetic) ===")
    print(f"extraction-clean rate : {n_clean}/{n} = {100*n_clean/n:.0f}%   (bar: >=80%)")
    print(f"grade-correct rate    : {n_correct}/{n} = {100*n_correct/n:.0f}%   (easy synthetics; plumbing+knowledge)")
    print(f"content-token rate    : {n_content_tok}/{n} = {100*n_content_tok/n:.0f}%   (A1 target is a real word)")
    print(f"VERDICT: {'BRANCH A WINS — base stays' if n_clean/n>=0.80 else 'BRANCH A FAILS — fall to branch B (-it) + substrate-sanity'}")


if __name__ == "__main__":
    main()
