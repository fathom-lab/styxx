# RECEIPT v0 check page

Status: DRAFT, unsigned. RECEIPT v0 (`receipt/v0/RECEIPT_v0.md`) is a lab draft, nothing in it is frozen, and this page is not signed by anyone. The checker inside the page is the lab's own `receipt/v0/check_v0.js`, so the page is not independent of the lab.

`receipt_check_v0.html` is one HTML file. It checks a document, optionally its manifest, and a receipt against RECEIPT v0, in a browser, with no network of its own: a Content-Security-Policy of `default-src 'none'`, Trusted Types, the checker embedded as text and run in a fresh worker with a time limit of 10 seconds, and the checker pinned by its sha256.

## The digest of the committed page

sha256 of receipt_check_v0.html: `577bfeb10865dce7afb9c261683b0cab967c7c1bb6ca52e3f6b996023018b507`

Size: 163,704 bytes.

This digest is written here, outside the page, because a page cannot carry its own digest. `tests/test_receipt_v0_page.py` checks that this line names the digest of the committed file. Compare it with the file you have:

```
sha256sum receipt_check_v0.html
```

On Windows: `certutil -hashfile receipt_check_v0.html SHA256`.

Read this digest from the repository at a commit you chose (for example this file on GitHub), not from a README that came with the page: a copy of the page can come with a copy of this README, and then the two agree with each other and tie nothing to the lab.

The page also shows the sha256 of the checker inside it (`cbdae74e95cbfb0280d38b0d38380dcbf2f32caf12d12c860c483aa2ab9e70a3`, the sha256 of `receipt/v0/check_v0.js` in this tree). That hash only shows the checker element is intact. A copy of the page can keep the checker and change the code that draws the verdict, so only the digest of the whole file, compared with this line, ties a copy to this tree.

## Open it from disk

Save `receipt_check_v0.html` anywhere and open it in a browser (double-click it, or File, Open). Nothing needs installing and the page fetches nothing. The page has three example buttons; each loads a document, a manifest and the receipt the command line checker printed for them.

Opening from disk has not been tried in any browser. If the browser will not start a worker from a file page, every run says "no verdict". Serving the folder from 127.0.0.1 is the setup the lead tried.

The page reads a file or a pasted text of at most 16,777,216 bytes (16 MiB); a larger input is "too large for this page" and is not read. The page's own list of limits does not state this one.

## What the page says about its checker

- "where the build took it from": `receipt/v0/check_v0.js of the tree this page is committed in; that file last changed at commit 53bbf47ed5304f047398d940db35f0ff3bea33cb (Fathom Lab)`. A page cannot name the commit that contains it. It names the checker by sha256, and names the commit that last changed the checker file. When history is present, `tests/test_receipt_v0_page.py` checks that this is still that commit. CI checks out one commit, but `tests/conftest.py` unshallows the checkout first, so the check runs on CI too. Pull request 270 was merged with a merge commit: in the copy of `origin/main` this branch's checkout held when this line was written, `git log -1 -- receipt/v0/check_v0.js` names 53bbf47e. A squash or rebase merge of pull request 270 would have given the checker a new last commit. This branch does not change the checker, so a squash or rebase merge of its own pull request leaves that answer as it is. Any later change to the checker makes the page's label false, and the test then fails until the page is rebuilt with `--origin` naming the new commit.
- "repository": `github.com/fathom-lab/styxx`, as plain text, not a link.
- "is that commit published": "not recorded when this page was built." The build makes no claim about what a remote holds, because such a claim can stop being true after the page is committed.

## Rebuild it

From the root of a checkout, with Python 3.9 or newer and nothing else:

```
python receipt/v0/page/build_page.py receipt/v0/check_v0.js --repo github.com/fathom-lab/styxx --origin "receipt/v0/check_v0.js of the tree this page is committed in; that file last changed at commit 53bbf47ed5304f047398d940db35f0ff3bea33cb (Fathom Lab)"
```

It writes `receipt/v0/page/receipt_check_v0.html`, or prints `unchanged:` when the file already holds those bytes. The build reads no clock and no network, so the same checker, sources and examples give the same bytes. It refuses, and writes no page, for the reasons listed in the docstring of `build_page.py`; each listed refusal has a test. One-line mutants of `build_page.py` were run against those tests: all but one were caught. The one that was not makes the builder's own check cut a script element case-sensitively; every text that goes into a script element is already refused for `</script` in any case, so that mutant cannot change a build.

The build reads:

- `receipt/v0/check_v0.js` (the checker, embedded byte for byte);
- `page_src/page.html`, `page_src/page.css`, `page_src/page_main.js` (the page);
- `examples/`: three example documents, the shared manifest, and `expected_cli_2026-11-13T153310Z.json`, the command line checker's answers on them with this checker, plus the receipts it printed under `receipts_cbdae74e95cbfb02/`. The build refuses examples recorded with another checker. The record's `checker.path` names the copy of the checker it was made with outside this repository; that copy has the same sha256 as `receipt/v0/check_v0.js` here. The record's `argv` lists hold absolute paths of the machine that made it; they mean nothing elsewhere. The record is pinned (the page is built from it), so it stays as it is; `make_examples.py` now writes paths relative to its own folder in any new record.

When the checker changes, the page must be rebuilt in a later commit, with `make_examples.py` run first for the new checker and `--origin` naming the commit that changed the checker. The byte test fails until then.

`--from-git REPO --commit COMMIT` builds from a commit of any checkout instead. It keeps write-once copies of the checker and the vectors under `src/`, which git ignores here: the tree already holds both.

## Tests

`tests/test_receipt_v0_page.py` (no browser):

- the committed page is byte for byte what `build_page.py` builds from this tree's checker with the command above;
- the checker embedded in the page, cut out by the HTML rule for script elements, is `receipt/v0/check_v0.js` byte for byte;
- the digest line above is the committed page's sha256;
- every refusal the builder's docstring lists, run on a copy of this folder in a temporary directory;
- every vector of `receipt/v0/vectors/receipt_v0_vectors.json` through the page's own code (`tests/prove_page2.js`, in Node worker threads), when Node 18 or newer is on PATH. With no Node, or an older one, that test is skipped and says why; the byte tests above do not need Node.

## Where it came from

The page was written and attacked outside the repository on 2026-11-13, and the lead opened a build of it in a real browser (Chrome in the desktop app, served from 127.0.0.1). That build, sha256 `a1e1a025f97207bd7d7a08780c38ed2301ef39a3dd64cc206517619cdf1e13e6`, 163,783 bytes, came from the same checker bytes, the same `page_src` and the same examples as this one. The two files differ in two labels the build is given, and in what follows from them:

- "where the build took it from": that build said `receipt/v0/check_v0.js at commit fd650bf0407bb0b860459bf80adc299d9fbafcb7 (Fathom Lab)`; this one names the tree and commit 53bbf47e, as above;
- "is that commit published": that build said "not known to be", from the remote branches its checkout knew; this one says "not recorded";
- the CSP `script-src` hash, in the meta tag and in the policy text the page prints, because the origin label sits inside the page script.

Built with the lead build's labels, this folder's builder and sources give the lead-verified file byte for byte. No line of `page_src` was changed for the move.

This file (`577bfeb1...`) has not been opened in a browser. The lead opened `a1e1a025...`, which differs from it only in the lines above, and on that build looked at the checker hash line, the origin, example 1 and a 375-pixel width. The probes of the policy (a fetch, an image, `innerHTML`, a worker from a data: address) and the document full of markup were run on an earlier build, `c631c9ee...`, made from the same `page_src` with an earlier checker (`006a71c7...`) and that checker's examples record.

## What the page does not prove

- A HELD verdict says one comparison passed. It does not say a sentence is true, that the manifest is honest, or that the right sentences were bound. The page prints the nine items of RECEIPT_v0.md section 12 and the gist of section 14; read the text itself for the rest (section 12 also has a paragraph on a third, Go, checker that the page leaves out).
- The checker is lab-written. Agreement between this page and the lab's command line, or with the lab's vectors, is agreement inside one lab. It does not show the checker is right.
- The page is not signed. The digest above ties a file to this tree; it does not say who wrote the tree.
- The CSP and Trusted Types protect only in browsers that enforce them. The lead's check saw Chrome refuse a fetch, an image, `innerHTML` and a data: worker, on build `c631c9ee...` (see above). It did not open the page from disk (`file://`), did not choose local files, and did not test Firefox, Safari or a phone. Nobody has checked from inside the worker that the worker inherits the page's policy.
- The page's own text says the lab "has not yet watched the page run in a browser". For this exact file that is still so: the lead watched `a1e1a025...`, which differs from it in the lines listed above. The sentence was left as it was, because changing it changes the page the lead checked.
