# styxx.v8 foundations — interface contract (frozen for the build agents, 2026-09-07)

Three pure modules. No module here imports torch/transformers or anything outside the standard
library, `cryptography`, and (optionally) `rfc8785`. Every function is deterministic. Every module
has a test file `tests/test_v8_<module>.py` that runs under `python -m pytest tests/test_v8_<module>.py -q`
from the repository root (`C:\Users\heyzo\clawd\wt\v8`). Tests must not skip; if a dependency is
missing they fail with the reason.

Repository conventions that apply (violations are defects):
- Files are written UTF-8, LF line endings, no BOM. Python files pass `python -m py_compile`.
- Nothing hashes working-tree text through CRLF: hash `bytes` you constructed, never `read_text()`.
- No claim words in docstrings or output: never "immutable", "tamper-proof", "self-verifying", "first".
- A test that pins a number must assert it; a comment is not a test.

---

## `styxx/v8/jcs.py` — RFC 8785 canonical bytes

```python
def canonical_bytes(obj) -> bytes
    # RFC 8785 JCS. Backend: `rfc8785.dumps(obj)` when importable (it is installed here, 0.1.4),
    # else `styxx.attestation.jcs(obj).encode("utf-8")`. Both must produce identical bytes on the
    # domain the certs use; the differential test below is the receipt for that.
    # Raises TypeError for non-JSON types (bytes, set, datetime, custom objects), ValueError for
    # NaN/Infinity, non-str dict keys, and ints outside the IEEE-754 exactly-representable range
    # (|n| > 2**53) — the envelope never stores such ints; refusing is the honest behaviour.
def sha256_hex(b: bytes) -> str          # 64 lowercase hex
def digest(obj) -> str                   # sha256_hex(canonical_bytes(obj)); NO "sha256:" prefix
def backend() -> str                     # "rfc8785" | "styxx.attestation"
```

Tests (`tests/test_v8_jcs.py`):
1. The RFC 8785 Appendix example: input
   `{"numbers": [333333333.33333329, 1E30, 4.50, 2e-3, 0.000000000000000000000000001], "string": "\u20ac$\u000F\u000aA'\u0042\u0022\u005c\\\"\u002f", "literals": [null, true, false]}`
   canonicalizes to exactly
   `{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],"string":"€$\u000f\nA'B\"\\\\\"/"}`
   (as UTF-8 bytes; the `€` is the raw UTF-8 character, `\u000f` and `\n` are the two-character escapes
   as RFC 8785 §3.2.2.2 prescribes, `\\\\` is two backslashes). Check the hash the RFC gives if you can
   recall it with confidence; otherwise pin the bytes only.
2. ES6 number formatting cases: `1e21 → 1e+21`, `1e-7 → 1e-7`, `0.000001 → 0.000001`, `-0.0 → 0`,
   `1.0 → 1`, `100 → 100`, `1.5e300`, subnormal `5e-324`, `123456789012345680000 → 123456789012345680000`,
   `0.1+0.2 → 0.30000000000000004`.
3. Differential (calibrated, per the lab's own rule): hypothesis-generated JSON values (nested
   dicts/lists, unicode strings including astral characters, all C0 controls, `"\u2028"`, lone
   surrogates EXCLUDED as invalid, floats over the full double range excluding NaN/Inf, ints within
   ±2**53, bools, None) — assert `rfc8785.dumps(x) == styxx.attestation.jcs(x).encode("utf-8")`.
   Then a MUTATION CHECK in the same test file: apply, in-process via monkeypatching a copied
   function, at least five semantic mutations to the styxx implementation (e.g. drop key sorting;
   emit `1e21` instead of `1e+21`; escape `/`; emit `-0` for negative zero; use `ensure_ascii=True`)
   and assert that the differential generator CATCHES each within `max_examples=300`. A mutation
   that the generator cannot catch must be listed in the test as `KNOWN_MISSES` with an assertion
   on the list's contents — the miss list is the finding, not a failure.
4. Refusals: NaN, inf, `2**53 + 1`, bytes, set, a dict with an int key → the documented exceptions.

---

## `styxx/v8/merkle.py` — RFC 6962 Merkle tree (as restated in RFC 9162 §2.1)

```python
EMPTY_ROOT: bytes      # sha256(b"") = e3b0c442...b855
def leaf_hash(entry: bytes) -> bytes           # sha256(b"\x00" + entry)
def node_hash(left: bytes, right: bytes) -> bytes   # sha256(b"\x01" + left + right)
def root(leaf_hashes: Sequence[bytes]) -> bytes     # MTH over ALREADY-HASHED leaves; MTH([]) = EMPTY_ROOT
def inclusion_proof(leaf_hashes, index: int, tree_size: int | None = None) -> list[bytes]   # PATH(index, D[0:tree_size])
def verify_inclusion(leaf_hash: bytes, index: int, tree_size: int, proof: Sequence[bytes], expected_root: bytes) -> bool
def consistency_proof(leaf_hashes, first: int, second: int | None = None) -> list[bytes]     # PROOF(first, D[0:second]); [] when first == second or first == 0
def verify_consistency(first: int, second: int, first_root: bytes, second_root: bytes, proof: Sequence[bytes]) -> bool

class MerkleTree:
    def __init__(self, leaf_hashes: Iterable[bytes] = ()): ...
    def append(self, leaf_hash: bytes) -> int      # returns the new leaf index
    @property
    def size(self) -> int
    def root(self, tree_size: int | None = None) -> bytes
    def inclusion_proof(self, index: int, tree_size: int | None = None) -> list[bytes]
    def consistency_proof(self, first: int, second: int | None = None) -> list[bytes]
```

Rules: inputs are leaf HASHES (32 bytes); `root`/proofs over raw entries are the caller's job via
`leaf_hash`. Every function validates its arguments (index < tree_size, 0 <= first <= second <= size,
32-byte hashes) and raises `ValueError`. Verification functions return `False`, never raise, on a
malformed proof (wrong length, wrong byte length) — a stranger's verifier must not crash on hostile
input. Verification follows the RFC 9162 §2.1.3.2 and §2.1.4.2 algorithms literally.

Tests (`tests/test_v8_merkle.py`):
1. The published CT test vectors. Leaves (raw entries, to be leaf-hashed):
   `b""`, `b"\x00"`, `b"\x10"`, `b"\x20\x21"`, `b"\x30\x31"`, `b"\x40\x41\x42\x43"`,
   `b"\x50\x51\x52\x53\x54\x55\x56\x57"`, `b"\x60\x61\x62\x63\x64\x65\x66\x67\x68\x69\x6a\x6b\x6c\x6d\x6e\x6f"`.
   Expected roots for tree sizes 1..8 (from the certificate-transparency `merkle` test file):
   1 `6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d`
   2 `fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125`
   3 `aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77`
   4 `d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7`
   5 `4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4`
   6 `76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef`
   7 `ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c`
   8 `5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328`
   Empty tree: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
   These vectors are quoted from memory by the orchestrator: COMPUTE them; if your implementation
   disagrees with any, do not "fix" the vector — report the disagreement in your return value and
   pin the size-1 and empty-tree values (which are derivable by hand) plus whatever agrees.
   Known inclusion-proof vector: leaf index 0, tree size 8 →
   `[96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7, 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e, 6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4]`;
   leaf 5, size 8 → `[bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b, ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0, d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7]`.
   Known consistency-proof vector: first=2, second=8 →
   `[5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e, 6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4]`; first=6, second=8 →
   `[0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a, ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0, d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7]`.
   Same rule: compute, compare, report disagreement rather than editing the vector.
2. Properties (hypothesis, sizes 0..96, random 32-byte leaves): every inclusion proof verifies against
   the root at its tree size; every consistency proof (all pairs m ≤ n) verifies; `MerkleTree`
   incremental roots equal functional `root()` at every size; a proof for size n verifies against
   root(n) and not against root(n+1) unless the tree is unchanged by definition (first == second).
3. Mutations: flipping any single byte of any proof element, changing the index, changing the
   tree size, or swapping two proof elements makes verification return `False`; a truncated or
   extended proof returns `False`; non-32-byte hashes return `False` from verify_* and raise
   ValueError from the generators.

---

## `styxx/v8/keys.py` — ed25519 via `cryptography`

```python
def generate() -> tuple[bytes, bytes]                 # (private_seed_32, public_32)
def public_from_private(private_seed_32: bytes) -> bytes
def encode_public(public_32: bytes) -> str             # "ed25519:" + base64url without padding
def decode_public(s: str) -> bytes                     # ValueError on bad prefix / length / padding present
def encode_signature(sig_64: bytes) -> str             # "ed25519:" + base64url without padding
def decode_signature(s: str) -> bytes
def sign(private_seed_32: bytes, message: bytes) -> bytes        # 64 bytes, pure Ed25519 (RFC 8032), no prehash
def verify(public_32: bytes, message: bytes, sig_64: bytes) -> bool   # False on any failure, never raises
def tagged(tag: str, digest_32: bytes) -> bytes        # tag.encode("ascii") + b"\x00" + digest_32  — the domain-separated preimage.
                                                       # Tags are fixed by the spec (v0.2): "styxx.v8/cert/1" for cert ids, "styxx.v8/sth/1" for tree heads.
                                                       # This module only provides the framing; it must refuse tags that are empty or contain NUL.
def save_private_pem(private_seed_32: bytes, path) -> None     # PKCS8 PEM, unencrypted, LF; best-effort 0600
def load_private_pem(path) -> bytes
def save_public(public_32: bytes, path) -> None        # the encode_public string + "\n", LF, no BOM
def load_public(path) -> bytes
```

Tests (`tests/test_v8_keys.py`):
1. RFC 8032 §7.1 test vector 1: secret `9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60`,
   public `d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a`, message empty, signature
   `e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b`.
   Also test vector 2 (message `72`, secret `4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb`,
   public `3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c`, signature
   `92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00`).
2. Round trips: encode/decode public and signature; padding present → ValueError; wrong length → ValueError.
3. `verify` returns False (does not raise) on: wrong key, wrong message, a flipped signature byte,
   a 63-byte signature, a 31-byte key.
4. `tagged`: distinct tags give distinct preimages for the same digest; empty tag and a tag containing
   NUL raise ValueError; the preimage of ("styxx.v8/cert/1", d) is exactly `b"styxx.v8/cert/1\x00" + d`.
5. Node interop (node 24 is installed at `node`): sign a message in Python, then run
   `node -e` with `crypto.verify(null, msg, {key: Buffer.concat([Buffer.from("302a300506032b6570032100","hex"), pub]), format: "der", type: "spki"}, sig)`
   and assert it prints `true`; the same with a flipped byte prints `false`. Fail (not skip) if node is missing.
6. PEM round trip through a temp dir; the public file is exactly the encoded string plus one LF.
