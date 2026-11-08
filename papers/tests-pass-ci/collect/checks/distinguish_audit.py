# -*- coding: utf-8 -*-
"""A distinguishability audit of decoys (P9): can a simple classifier tell decoys from real items by their form?

Features are read from an item's public texts only, never from content words or names: lengths and line counts of the
description and the CI block; the description's markdown shape (headings, bullets, checkboxes, code fences, blank
lines); counts of check, status, job, step, annotation and log lines and the characters of the log excerpts; counts of
each conclusion; check runs of Actions and of other apps; the runner's own steps (Set up job, Complete job, Post
steps) and steps per job; annotations reading "Process completed with exit code"; generic CI words in check, job and
step names (`NAME_WORDS`); the tokens the builder writes (<repo>, <url>, <date>, [agent footer removed], [cut], the
names-only line); and date-like strings left (P9's `date_traces`, clock times, four-digit years, MM/DD forms).

Models: logistic regression (standardised features, L2) and a decision tree of depth 2. Held-out evaluation by
stratified group k-fold, repeated over seeds; a group is the source pull request, so an item and a construction made
from it never sit on both sides of a split. Chance is the majority share for accuracy and 0.5 for balanced accuracy;
a permutation test shuffles the labels and reruns the same splits. Top features: the logistic model's mean absolute
standardised coefficient over folds (its sign: positive leans decoy), and each feature alone (a one-split tree) held
out the same way.

    py -3.12 checks/distinguish_audit.py packets OUT.json DIR [DIR ...]
        the two-step items of dry-run packets (packet_<P>.json with key_<P>.json): decoys against real items
    py -3.12 checks/distinguish_audit.py dev OUT.json STORE PACKET_DIR
        a dry-run packet's decoys against every two-step-like item of a pre-window store
    py -3.12 checks/distinguish_audit.py constructed OUT.json STORE [--logs DIR] [--only KIND] [--real-with-log]
        the pre-window store's real items in P2's combinations against items made from that store by the builder's
        own transforms (a claim line written into a description, a step flip with a donor's log, annotations and
        times): what the constructions themselves leave. --only claim_written or step_flip keeps one construction;
        --real-with-log keeps the real items whose CI block shows a failing test job's log excerpt (the shape a step
        flip makes)
    py -3.12 checks/distinguish_audit.py set OUT.json STORE DECOYSET.json [--logs DIR ...] [--store DIR ...]
                                         [--exclude-spec SPEC.json] [--cap-per-repo N]
        a decoy set (decoyset_<P>.json, its item texts) against the pre-window store's real items in P2's combinations
        (and those of each further pre-window --store), leaving out every pull request the spec draws on (a decoy's
        source, a flip's donor and reference run), so a real decoy never stands beside its own item; --logs may repeat.
        --cap-per-repo N then keeps at most N real stand-ins per repository, over all the stores, by a fixed rule that
        reads no feature: each repository's stand-ins in item_sha256 order, the lowest N kept (the lead's decision of
        2026-11-03: the stand-ins follow the per-repository cap of P2's real draw, 3 claimers a repository, PREREG
        section 9.2)
    py -3.12 checks/distinguish_audit.py identity OUT.json STORE [--logs DIR]
        no classifier: each real failing test-like job of the store with one failing test step is made again through
        the step flip (`tpc_decoys.fail_test_step`) with its own log, annotations, times, later steps and skipped post
        steps, and the rendered item compared with the real one; counts of identical and differing items, and of the
        kinds of line that differ

Every mode refuses a store row opened on or after 2026-11-02. It writes numbers and feature names only: no item text,
pull request, repository or id. Needs numpy and scikit-learn; nothing in the collector imports this file.
"""
from __future__ import annotations

import copy
import difflib
import gzip
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import tpc_decoys  # noqa: E402
import tpc_items  # noqa: E402
import tpc_packet  # noqa: E402
from tpc_const import WINDOW_START, parse_utc  # noqa: E402
from tpc_screen import _map, is_test_step  # noqa: E402

CI_SPLIT = "\n\n" + tpc_items.CI_HEAD + "\n"
CONCLUSIONS = ("success", "failure", "skipped", "cancelled", "neutral", "timed_out", "action_required", "none",
               "pending", "error", "in_progress", "queued", "startup_failure", "stale")
NAME_WORDS = ("test", "tests", "unit", "integration", "e2e", "lint", "build", "check", "checks", "type", "types",
              "typecheck", "format", "ci", "ubuntu", "windows", "macos", "node", "python", "go", "rust", "java",
              "coverage", "deploy", "docs", "security", "codeql", "analyze", "vercel", "preview", "release")
CLOCK = re.compile(r"(?<![\d:])\d{1,2}:\d{2}(?::\d{2})?(?![\d:])")
YEAR = re.compile(r"(?<!\d)20\d\d(?!\d)")
MMDD = re.compile(r"(?<![\d/.])\d{1,2}/\d{1,2}(?![\d/])")
WORD = re.compile(r"[a-z0-9]+")
CLAIM_LINE = "All tests pass."


# ---- features ---------------------------------------------------------------------------------------------------
def _log1p(x: float) -> float:
    return math.log1p(max(0.0, float(x)))


def features(step1_text: str, item_text: str) -> dict:
    """The form of one two-step item: numbers only, every key always present."""
    desc = step1_text or ""
    item = item_text or ""
    ci = item.split(CI_SPLIT, 1)[1] if CI_SPLIT in item else ""
    dl = desc.split("\n")
    cl = ci.split("\n")
    f = {
        "desc_chars": _log1p(len(desc)), "desc_lines": _log1p(len(dl)), "desc_words": _log1p(len(desc.split())),
        "desc_headings": sum(1 for ln in dl if ln.lstrip().startswith("#")),
        "desc_bullets": sum(1 for ln in dl if re.match(r"\s*[-*+] ", ln)),
        "desc_checkboxes": sum(1 for ln in dl if re.match(r"\s*[-*+] \[[ xX]\]", ln)),
        "desc_ticked": sum(1 for ln in dl if re.match(r"\s*[-*+] \[[xX]\]", ln)),
        "desc_code_fences": sum(1 for ln in dl if ln.lstrip().startswith("```")),
        "desc_blank_lines": sum(1 for ln in dl if not ln.strip()),
        "ci_chars": _log1p(len(ci)), "ci_lines": _log1p(len(cl)),
    }
    kinds = Counter()
    names = []
    log_chars = 0
    in_log = False
    for ln in cl:
        for k in ("check", "status", "job", "step", "annotation", "log"):
            if ln.startswith(k + ": "):
                kinds[k] += 1
                if k in ("check", "status", "job", "step"):
                    names.append(ln[len(k) + 2:].rsplit(" -> ", 1)[0])
                in_log = k == "log"
                break
        else:
            if in_log and ln != tpc_items.FIXED_LINE:
                log_chars += len(ln) + 1
    for k in ("check", "status", "job", "step", "annotation", "log"):
        f["n_" + k] = kinds[k]
    f["log_chars"] = _log1p(log_chars)
    for c in CONCLUSIONS:
        f["concl_" + c] = len(re.findall(r" -> " + re.escape(c) + r"$", ci, re.M))
    f["checks_actions"] = len(re.findall(r"^check: .* \(github-actions\) -> ", ci, re.M))
    f["checks_other_app"] = kinds["check"] - f["checks_actions"]
    f["step_set_up_job"] = len(re.findall(r"^step: Set up job -> ", ci, re.M))
    f["step_complete_job"] = len(re.findall(r"^step: Complete job -> ", ci, re.M))
    f["step_post"] = len(re.findall(r"^step: Post ", ci, re.M))
    f["steps_per_job"] = kinds["step"] / max(1, kinds["job"])
    f["ann_exit_code"] = len(re.findall(r"^annotation: .*Process completed with exit code", ci, re.M))
    f["names_test_like"] = sum(1 for n in names if is_test_step(n))
    words = Counter(w for n in names for w in WORD.findall(n.lower()))
    for w in NAME_WORDS:
        f["name_word_" + w] = words[w]
    f["tok_repo"] = item.count(tpc_items.REPO_TOKEN)
    f["tok_url"] = item.count(tpc_items.URL_TOKEN)
    f["tok_date"] = item.count(tpc_items.DATE_TOKEN)
    f["tok_agent_footer"] = item.count(tpc_items.FOOTER_TOKEN)
    f["tok_cut"] = item.count(tpc_items.CUT_TOKEN)
    f["names_only_line"] = int(tpc_items.NAMES_ONLY_LINE in ci)
    f["date_traces"] = len(tpc_items.date_traces(desc) + tpc_items.date_traces(item))
    f["clock_times"] = len(CLOCK.findall(item))
    f["years"] = len(YEAR.findall(item))
    f["mm_dd"] = len(MMDD.findall(item))
    return f


# ---- the audit ----------------------------------------------------------------------------------------------------
def _splits(y, groups, seed, k):
    from sklearn.model_selection import StratifiedGroupKFold
    return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(y, y, groups))


def _fit_predict(model, X, y, splits):
    import numpy as np
    pred = np.zeros(len(y), dtype=int)
    coefs = []
    for tr, te in splits:
        if len(set(y[tr].tolist())) < 2:  # a training fold holding one class predicts it
            pred[te] = y[tr][0]
            continue
        m = model()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        if hasattr(m, "named_steps") and hasattr(m.named_steps.get("lr"), "coef_"):
            coefs.append(m.named_steps["lr"].coef_[0])
    return pred, coefs


def _scores(y, pred):
    import numpy as np
    acc = float(np.mean(pred == y))
    rec = [float(np.mean(pred[y == c] == c)) for c in (0, 1)]
    return acc, (rec[0] + rec[1]) / 2


def audit(rows: list, labels: list, groups: list, *, repeats: int = 20, permutations: int = 200,
          top: int = 12) -> dict:
    """rows: feature dicts; labels: 1 decoy (or constructed), 0 real; groups: the source of each row."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    names = sorted(rows[0])
    X = np.array([[r[n] for n in names] for r in rows], dtype=float)
    keep = [i for i in range(X.shape[1]) if np.ptp(X[:, i]) > 0]
    names = [names[i] for i in keep]
    X = X[:, keep]
    y = np.array(labels, dtype=int)
    g = np.array([str(x) for x in groups])
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    k = max(2, min(5, len({gg for gg, yy in zip(g, y) if yy == 1}), len({gg for gg, yy in zip(g, y) if yy == 0})))
    lr = lambda: Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=5000))])  # noqa: E731
    tree = lambda: DecisionTreeClassifier(max_depth=2, random_state=0)  # noqa: E731
    stump = lambda: DecisionTreeClassifier(max_depth=1, random_state=0)  # noqa: E731
    split_sets = [_splits(y, g, seed, k) for seed in range(repeats)]
    out = {"items": len(y), "decoys": n1, "real": n0, "groups": len(set(g)), "folds": k, "repeats": repeats,
           "features_used": len(names), "chance_accuracy_majority": max(n0, n1) / len(y),
           "chance_balanced_accuracy": 0.5}
    for label, model in (("logistic", lr), ("tree_depth2", tree)):
        accs, bals, coefs = [], [], []
        for sp in split_sets:
            pred, c = _fit_predict(model, X, y, sp)
            a, b = _scores(y, pred)
            accs.append(a)
            bals.append(b)
            coefs += c
        out[label] = {"accuracy_mean": float(np.mean(accs)), "accuracy_min": float(np.min(accs)),
                      "accuracy_max": float(np.max(accs)), "balanced_mean": float(np.mean(bals)),
                      "balanced_min": float(np.min(bals)), "balanced_max": float(np.max(bals))}
        if coefs:
            cm = np.mean(np.array(coefs), axis=0)
            ca = np.mean(np.abs(np.array(coefs)), axis=0)
            order = np.argsort(-ca)[:top]
            out[label]["top_features"] = [{"feature": names[i], "mean_abs_coef": round(float(ca[i]), 4),
                                           "leans": "decoy" if cm[i] > 0 else "real"} for i in order]
    # permutation test of the logistic model's balanced accuracy: labels shuffled, split again by the same seeds
    rng = np.random.default_rng(12345)
    obs = out["logistic"]["balanced_mean"]
    ge = 0
    for _ in range(permutations):
        yp = rng.permutation(y)
        bal = []
        for seed in range(3):
            pred, _ = _fit_predict(lr, X, yp, _splits(yp, g, seed, k))
            bal.append(_scores(yp, pred)[1])
        ge += int(np.mean(bal) >= obs)
    out["permutation"] = {"n": permutations, "p_balanced": (1 + ge) / (1 + permutations)}
    single = []
    for i, n in enumerate(names):
        bals = []
        for sp in split_sets[:5]:
            pred, _ = _fit_predict(stump, X[:, [i]], y, sp)
            bals.append(_scores(y, pred)[1])
        mean0 = float(np.mean(X[y == 0, i]))
        mean1 = float(np.mean(X[y == 1, i]))
        single.append((float(np.mean(bals)), n, mean0, mean1))
    single.sort(key=lambda t: (-t[0], t[1]))
    out["single_feature"] = [{"feature": n, "balanced": round(b, 4), "mean_real": round(m0, 3),
                              "mean_decoy": round(m1, 3)} for b, n, m0, m1 in single[:top]]
    return out


# ---- inputs -----------------------------------------------------------------------------------------------------
def _json(p: Path):
    return json.loads(Path(p).read_bytes().decode("utf-8"))


def from_packets(dirs: list) -> tuple:
    rows, labels, groups = [], [], []
    for d in dirs:
        d = Path(d)
        for pk in sorted(d.glob("packet_*.json")):
            name = pk.stem.split("_", 1)[1]
            pub = _json(pk)
            key = _json(d / f"key_{name}.json")
            if (pub.get("beacon") or {}).get("kind") != "stand_in":
                raise SystemExit(f"refused: {pk.name} is not a dry run under the stand-in hash")
            for it in pub["items"]:
                if it["steps"] != 2:
                    continue
                k = key["items"][it["id"]]
                rows.append(features(it["step1_text"], it["item_text"]))
                labels.append(int(bool(k["decoy"])))
                groups.append(f"{name}:{it['id']}")
    return rows, labels, groups


def _pre_window_rows(store: Path) -> list:
    rows = tpc_packet._read_rows(store)
    late = [r for r in rows if parse_utc(r["created_at"]) >= WINDOW_START]
    if late:
        raise SystemExit(f"refused: {len(late)} rows opened on or after 2026-11-02")
    return rows


def _with_logs(ci: dict, pr_id, logs_dir) -> dict:
    """The record with the logs of its failing test-like jobs read from `logs_dir/<pr_id>/<job_id>.log.gz` where the
    store has none (pre-window logs read for P2's decoy material)."""
    if logs_dir is None:
        return ci
    dirs = [logs_dir] if isinstance(logs_dir, (str, Path)) else list(logs_dir)
    for j in tpc_items.failing_test_like_jobs(ci):
        if str(j["id"]) in ci["logs"]:
            continue
        for d in dirs:
            p = Path(d) / str(pr_id) / f"{j['id']}.log.gz"
            if p.exists():
                ci["logs"][str(j["id"])] = gzip.decompress(p.read_bytes())
                break
    return ci


def _two_step_like(r: dict) -> bool:
    return bool(r.get("eligible")) and bool((r.get("ci") or {}).get("has_ci_test_result")) and (
        r.get("screen") == "claimer" or (r.get("screen") == "none" and r["ci"]["outcome"] == "tests failed"))


def _in_p2_combination(r: dict) -> bool:
    ci = r.get("ci") or {}
    return bool(r.get("eligible")) and (
        (r.get("screen") in ("claimer", "none") and ci.get("outcome") == "tests failed")
        or (r.get("screen") == "claimer" and ci.get("outcome") == "tests passed" and ci.get("any_failed_job_or_check")))


def from_dev(store: Path, packet_dir: Path) -> tuple:
    store = Path(store)
    rows, labels, groups = [], [], []
    for r in _pre_window_rows(store):
        if not _two_step_like(r):
            continue
        it = tpc_items.build_item(tpc_items.body_from_row(store, r), tpc_items.ci_from_row(store, r), r.get("repo"))
        rows.append(features(it["step1_text"], it["item_text"]))
        labels.append(0)
        groups.append(f"real:{r['pr_id']}")
    d = Path(packet_dir)
    for pk in sorted(d.glob("packet_*.json")):
        name = pk.stem.split("_", 1)[1]
        pub = _json(pk)
        key = _json(d / f"key_{name}.json")
        for it in pub["items"]:
            if key["items"][it["id"]]["decoy"]:
                rows.append(features(it["step1_text"], it["item_text"]))
                labels.append(1)
                groups.append(f"decoy:{name}:{it['id']}")
    return rows, labels, groups


def _shows_log(it: dict) -> bool:
    return any(x.get("how") not in ("no_log", "item_budget_spent") for x in (it.get("build") or {}).get("logs", []))


def _real_items(store: Path, logs_dir, with_log: bool = False) -> list:
    """(row, body, ci, item) of the pre-window store's real items in P2's combinations, logs merged; with
    `with_log`, only those whose CI block shows a log excerpt."""
    out = []
    for r in _pre_window_rows(store):
        if not _in_p2_combination(r):
            continue
        body = tpc_items.body_from_row(store, r)
        ci = _with_logs(tpc_items.ci_from_row(store, r), r["pr_id"], logs_dir)
        it = tpc_items.build_item(body, ci, r.get("repo"))
        if with_log and not _shows_log(it):
            continue
        out.append((r, body, ci, it))
    return out


def _write_claim(body: str) -> str:
    """A claim line written into a description, above its agent footer when it has one (as a builder would)."""
    lines = (body or "").replace("\r\n", "\n").split("\n")
    at = max((i for i, ln in enumerate(lines) if tpc_items.FOOTER.search(ln)), default=None)
    if at is None:
        return "\n".join(lines + ["", CLAIM_LINE])
    return "\n".join(lines[:at] + [CLAIM_LINE, ""] + lines[at:])


def _donors(store: Path, logs_dir) -> list:
    """(repo, pr_id, job, log, annotations, failing step) of failing test-like jobs of the pre-window store with a log."""
    out = []
    for r in _pre_window_rows(store):
        if not r.get("eligible"):
            continue
        ci = _with_logs(tpc_items.ci_from_row(store, r), r["pr_id"], logs_dir)
        for j in tpc_items.failing_test_like_jobs(ci):
            log = ci["logs"].get(str(j["id"]))
            steps = [s for s in j.get("steps") or [] if _map(s.get("conclusion")) == "failure"]
            if log is None or len(steps) != 1:
                continue
            ann = [a for a in ci.get("annotations") or [] if a.get("check_run_id") == j["id"]]
            out.append((r.get("repo"), r["pr_id"], j, log, ann, steps[0]))
    return out


def constructed(store: Path, logs_dir=None, only=None, real_with_log: bool = False) -> tuple:
    store = Path(store)
    reals = _real_items(store, logs_dir, real_with_log)
    rows, labels, groups = [], [], []
    counts = Counter()
    for r, body, ci, it in reals:
        rows.append(features(it["step1_text"], it["item_text"]))
        labels.append(0)
        groups.append(r["pr_id"])
        counts["real"] += 1
    donors = _donors(store, logs_dir)
    for r in _pre_window_rows(store):
        ci0 = r.get("ci") or {}
        if not r.get("eligible") or not ci0.get("has_ci_test_result"):
            continue
        body = tpc_items.body_from_row(store, r)
        # a claim written beside a real test failure (yes_log's synthetic description)
        if only in (None, "claim_written") and r.get("screen") == "none" and ci0.get("outcome") == "tests failed":
            ci = _with_logs(tpc_items.ci_from_row(store, r), r["pr_id"], logs_dir)
            it = tpc_items.build_item(_write_claim(body), ci, r.get("repo"))
            rows.append(features(it["step1_text"], it["item_text"]))
            labels.append(1)
            groups.append(r["pr_id"])
            counts["claim_written"] += 1
        # a passing test step turned to failing, with a donor's excerpt, annotations and times (P9's step flip)
        if only in (None, "step_flip") and r.get("screen") == "claimer" and ci0.get("outcome") == "tests passed":
            ci = tpc_items.ci_from_row(store, r)
            done = False
            for j in sorted(ci.get("jobs") or [], key=lambda x: x["id"]):
                if done or _map(j.get("conclusion")) != "success":
                    continue
                for s in sorted(j.get("steps") or [], key=lambda x: x.get("number") or 0):
                    if not is_test_step(s.get("name") or "") or s.get("conclusion") != "success":
                        continue
                    if tpc_decoys.later_steps(j, s["name"]):
                        continue
                    dn = next((d for d in donors if d[0] != r.get("repo")), None)
                    if dn is None:
                        continue
                    drepo, _, _, dlog, dann, dstep = dn
                    flipped = tpc_decoys.fail_test_step(
                        ci, j["id"], s["name"], dlog, annotations=dann,
                        step_times=(dstep.get("started_at"), dstep.get("completed_at")))
                    it = tpc_items.build_item(body, flipped, r.get("repo"), also=(drepo,))
                    rows.append(features(it["step1_text"], it["item_text"]))
                    labels.append(1)
                    groups.append(r["pr_id"])
                    counts["step_flip"] += 1
                    donors.append(donors.pop(donors.index(dn)))  # the next flip takes another donor
                    done = True
                    break
    return rows, labels, groups, dict(counts)


def spec_sources(spec_path: Path) -> set:
    """The pull requests a decoy spec draws on (each decoy's source, and a step flip's donor and reference run),
    normalised as `tpc_decoys.norm_pr` does: none of them stands in as a real item beside the set it built."""
    out = set()
    for e in _json(spec_path).get("decoys") or []:
        refs = [(e.get("source") or {}).get("pr")]
        fl = e.get("flip") or {}
        refs += [(fl.get("donor") or {}).get("pr"), (fl.get("reference") or {}).get("pr")]
        for ref in refs:
            if ref:
                out.add(tpc_decoys.norm_pr(ref))
    return out


def cap_per_repo(entries: list, cap: int) -> tuple:
    """(kept, left_out) of `entries`, each (repository, item_sha256, anything): at most `cap` kept per repository
    (case-folded), those with the lowest item sha256, the kept ones in their given order. The rule reads a stand-in's
    repository and item sha256 and nothing else, so no feature chooses which stand-ins stay (the lead's decision of
    2026-11-03: the audit's real stand-ins follow the per-repository cap of P2's real draw, PREREG section 9.2)."""
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError(f"a cap per repository is a whole number of 1 or more, not {cap!r}")
    by_repo = {}
    for i, (repo, sha, _) in enumerate(entries):
        by_repo.setdefault(str(repo).lower(), []).append((str(sha), i))
    keep = set()
    for held in by_repo.values():
        keep.update(i for _, i in sorted(held)[:cap])
    kept = [e for i, e in enumerate(entries) if i in keep]
    return kept, len(entries) - len(kept)


def from_set(store: Path, set_path: Path, logs_dir=None, exclude=(), stores=(), cap=None) -> tuple:
    """The set's decoys against the real items in P2's combinations of the store (and of any further pre-window
    `stores`), leaving out every pull request in `exclude` (the decoys' own sources, `spec_sources`), so a real decoy
    never stands beside its own item; with `cap`, at most `cap` of the stand-ins left per repository over all the
    stores (`cap_per_repo`, applied after the sources are left out)."""
    rows, labels, groups = [], [], []
    exclude = set(exclude)
    left_out = 0
    reals = []
    for st in [Path(store)] + [Path(s) for s in stores]:
        for r, body, ci, it in _real_items(st, logs_dir):
            if tpc_decoys.norm_pr(f"{r.get('repo')}#{r.get('number')}") in exclude:
                left_out += 1
                continue
            reals.append((r.get("repo"), it["item_sha256"], (r, it)))
    from_set.left_out = left_out
    from_set.left_out_by_cap = 0
    if cap is not None:
        reals, from_set.left_out_by_cap = cap_per_repo(reals, cap)
    per_repo = Counter(str(repo).lower() for repo, _, _ in reals)
    from_set.repositories = len(per_repo)
    from_set.most_per_repository = max(per_repo.values(), default=0)
    for _, _, (r, it) in reals:
        rows.append(features(it["step1_text"], it["item_text"]))
        labels.append(0)
        groups.append(f"real:{r['pr_id']}")
    s = _json(set_path)
    for i, d in enumerate(s["decoys"]):
        rows.append(features(d["step1_text"], d["item_text"]))
        labels.append(1)
        groups.append(f"decoy:{i}")
    return rows, labels, groups


def identity(store: Path, logs_dir=None) -> dict:
    """Counts of real failing jobs whose item the step flip makes again byte for byte."""
    store = Path(store)
    kinds_of = ("check", "status", "job", "step", "annotation", "log")
    c, diffs = Counter(), Counter()
    for r in _pre_window_rows(store):
        if not r.get("eligible") or not (r.get("ci") or {}).get("failed_jobs"):
            continue
        ci = _with_logs(tpc_items.ci_from_row(store, r), r["pr_id"], logs_dir)
        jobs = tpc_items.failing_test_like_jobs(ci)
        if not jobs:
            continue
        body = tpc_items.body_from_row(store, r)
        real = tpc_items.build_item(body, ci, r.get("repo"))
        for j in jobs:
            fs = [s for s in j.get("steps") or [] if _map(s.get("conclusion")) == "failure"]
            if len(fs) != 1 or not is_test_step(fs[0].get("name") or ""):
                c["not_one_failing_test_step"] += 1
                continue
            s = fs[0]
            later = {x["name"]: x["conclusion"] for x in tpc_decoys.later_steps(j, s["name"])}
            own = [a for a in ci.get("annotations") or [] if a.get("check_run_id") == j["id"]]
            base = copy.deepcopy(ci)
            base["annotations"] = [a for a in base.get("annotations") or [] if a.get("check_run_id") != j["id"]]
            skipped = [x["name"] for x in j.get("steps") or [] if (x.get("name") or "").startswith("Post ")
                       and x.get("conclusion") == "skipped"]
            try:
                fl = tpc_decoys.fail_test_step(base, j["id"], s["name"], ci["logs"].get(str(j["id"])),
                                               post_skipped=skipped, later=later or None, annotations=own,
                                               step_times=(s.get("started_at"), s.get("completed_at")))
            except tpc_decoys.DecoyError:
                c["refused"] += 1
                continue
            it = tpc_items.build_item(body, fl, r.get("repo"))
            if it["item_text"] == real["item_text"]:
                c["identical"] += 1
                continue
            c["differs"] += 1
            a_l, b_l = real["item_text"].split("\n"), it["item_text"].split("\n")
            if sorted(a_l) == sorted(b_l):
                diffs["same_lines_other_order"] += 1
            seen = set()
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a_l, b_l, autojunk=False).get_opcodes():
                if tag != "equal":
                    for ln in a_l[i1:i2] + b_l[j1:j2]:
                        head = ln.split(": ", 1)[0]
                        seen.add(tag + ":" + (head if head in kinds_of else "other"))
            for k in seen:
                diffs[k] += 1
    return {"mode": "identity", "logs_merged": logs_dir is not None, "jobs": dict(c), "differing_lines": dict(diffs)}


def main(argv: list) -> int:
    mode, out = argv[1], Path(argv[2])
    rest = argv[3:]
    logs = only = spec = None
    stores = []
    logs_all = []
    while "--logs" in rest:
        i = rest.index("--logs")
        logs_all.append(rest[i + 1])
        rest = rest[:i] + rest[i + 2:]
    if logs_all:
        logs = logs_all[0] if len(logs_all) == 1 else logs_all
    while "--store" in rest:
        i = rest.index("--store")
        stores.append(rest[i + 1])
        rest = rest[:i] + rest[i + 2:]
    if "--exclude-spec" in rest:
        i = rest.index("--exclude-spec")
        spec = rest[i + 1]
        rest = rest[:i] + rest[i + 2:]
    if "--only" in rest:
        i = rest.index("--only")
        only = rest[i + 1]
        rest = rest[:i] + rest[i + 2:]
    cap = None
    if "--cap-per-repo" in rest:
        if mode != "set":
            raise SystemExit("refused: --cap-per-repo is a set-mode option")
        i = rest.index("--cap-per-repo")
        cap = int(rest[i + 1])
        rest = rest[:i] + rest[i + 2:]
    real_with_log = "--real-with-log" in rest
    rest = [x for x in rest if x != "--real-with-log"]
    extra = {}
    if mode == "identity":
        data = (json.dumps(identity(Path(rest[0]), logs), indent=1, sort_keys=True) + "\n").encode("utf-8")
        out.write_bytes(data)
        sys.stdout.buffer.write(data)
        return 0
    if mode == "packets":
        rows, labels, groups = from_packets(rest)
    elif mode == "dev":
        rows, labels, groups = from_dev(Path(rest[0]), Path(rest[1]))
    elif mode == "constructed":
        rows, labels, groups, extra = constructed(Path(rest[0]), logs, only, real_with_log)
        extra = dict(extra, only=only, real_with_log=real_with_log)
    elif mode == "set":
        excl = spec_sources(Path(spec)) if spec else set()
        rows, labels, groups = from_set(Path(rest[0]), Path(rest[1]), logs, excl, stores, cap)
        extra = {"stores": 1 + len(stores), "excluded_sources": len(excl),
                 "real_items_left_out_as_sources": from_set.left_out}
        # the uncapped output keeps its earlier keys, so a run made before the cap reproduces byte for byte
        if cap is not None:
            extra.update(cap_per_repo=cap, real_items_left_out_by_cap=from_set.left_out_by_cap,
                         real_repositories=from_set.repositories,
                         real_most_per_repository=from_set.most_per_repository)
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    res = {"mode": mode, "logs_merged": logs is not None, "construction_counts": extra}
    res.update(audit(rows, labels, groups))
    data = (json.dumps(res, indent=1, sort_keys=True) + "\n").encode("utf-8")
    out.write_bytes(data)
    sys.stdout.buffer.write(data)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
