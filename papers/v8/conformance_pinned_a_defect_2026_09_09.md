# FINDING — the conformance set was pinning two of the defects the adversary exploited

Fathom Lab · 2026-09-09 · **A finding about `conformance/v8`, produced while repairing two rules
the set defended.** Five committed vectors stopped reproducing when the rules were repaired. Each
one is named below with the repair that moved it and with what answers for it now.

This is not a claim that the two repairs are complete, and not a measurement of how many other
vectors carry the same shape of mistake. It reports what five vectors did and what property of
conformance sets that is an instance of. Not sworn.

## What happened

Two rules were repaired in `styxx.v8` on 2026-09-09. Both close an attack an adversary had
demonstrated against a running system, both are recorded in
`papers/v8/challenge_and_attack_2026_09_09`.

**(a) A challenge must declare what produced its distances.** `styxx/v8/schema/challenge.json` now
requires `body.subject` and `body.recipe_core`. Attack C3: every challenge the ladder produced
carried `subject: {}` and `recipe: {}` with a body of `{per_channel, coverage, environment}`, so a
reader of the cert could not tell what had been run to obtain the numbers it carried.

**(b) A ref role is named once.** `cert.duplicate_roles` refuses a repeated role, excepting
`result`, `robustness` and `run`, which are plural by construction. Attack C-DUPREF: duplicate
`own` refs on a challenge were resolved last-wins by the log's loop, while a reader walking `refs`
in order saw the earlier one, so one cert had two readings.

Both repairs are correct. Both broke the conformance set.

## The five vectors

`python -m pytest tests/test_v8_conformance.py -q` reported five of 107 `cert` vectors not
reproducing. Every one of the five was replayed by hand against the repaired tree, and the reasons
`cert.check` now emits were read one by one:

| retired id | what it pinned | what the repaired rules say | repair |
|---|---|---|---|
| `41a682e2…4cb5cfbc` | `ok: true` | `body: 'recipe_core' is a required property`, `body: 'subject' is a required property` | (a) |
| `b8777ac4…ab637ffe` | `ok: true` | the same two | (a) |
| `4bbbb3cb…6a8eafb0` | `reason_kinds: ["schema"]` | those two, plus the `coverage` enum violation the vector was about | (a) |
| `6e1391de…3663fe281b` | `reason_kinds: ["schema"]` | those two, plus the missing `own` ref the vector was about | (a) |
| `a6f0718b…6e6228dd` | `ok: true` | `refs: role 'battery' is named more than once` | (b) |

Nothing else moved. Of the 89 addresses this regeneration retired, 84 still reproduce against the
tree that retired them — they retired because a source now calls with different arguments, not
because an answer changed — and the five above are the whole of the remainder. That split is the
test that separates a rewritten test from a moved answer, and it was run before anything was
regenerated.

## What the set was doing

Three of the five pinned `ok: true` on a cert the repaired rules refuse. `41a682e2…` and
`b8777ac4…` are challenges that assert nothing about what produced their distances: exactly the
object C3 describes. `a6f0718b…` is a fingerprint naming the `battery` role twice: the same
construction as C-DUPREF, one role short of the attack.

So the set held a recorded, addressed, replayed statement that the implementation accepts each of
those. When the rules were repaired, those statements became false, and the replay reported the
repair as a failure of the set.

The mechanism is plain and has nothing to do with these two rules. A vector's `expect` is whatever
the implementation returned while the source test ran. Nothing in the pipeline that produces a
vector asks whether the answer was right — the recorder records, the fold addresses, the replay
compares. A defect present on the day the set was generated is recorded with the same standing as a
behaviour anyone intended, and from then on the set fails whenever that defect is removed.

The generator's moved-core refusal points the same way. It refuses to write when a committed
address carries a different outcome, prints both, and calls that a finding about `styxx.v8`. That
is right when the committed outcome was right. When the committed outcome was the defect, the same
rule stands in the way of the repair, and the rule cannot tell the two cases apart, because the
information that would tell them apart is not in the set. Here the refusal never fired: the sources
were repaired at the same time, so the inputs changed, so the addresses changed, and the set failed
at replay instead. Same signal, different door.

## An acceptance and a refusal are not the same kind of pin

The two shapes among the five behave differently, and the difference is general.

A vector pinning `ok: true` asserts that **no** rule refuses this input. It is a statement
quantified over the whole rule set, and every rule added later can falsify it. That is what
happened to three of the five.

A vector pinning a refusal asserts that **these** reasons fire. Adding a rule usually leaves it
refusing. `4bbbb3cb…` and `6e1391de…` did keep refusing; they failed only because this set pins the
multiset of reason kinds, and each gained two `schema` reasons. A set that pinned `ok: false` alone
would have replayed both of them green through repair (a) and reported nothing.

Two consequences follow. A vector that pins an acceptance is the shape that can silently encode a
hole, because the hole is the absence of a rule and the vector asserts exactly that absence. And a
vector that pins a refusal is worth more when it pins the reasons and not only the verdict — this
set does, and that is why two of the five surfaced at all.

The committed set today, by that division:

```
assert an acceptance   240   cert.check ok:true 180, log.verify_* 33, merkle.verify_* 25,
                             verify.diff exit 0  2
assert a refusal       177   cert.check 39, log.verify_* 61, merkle.verify_* 33,
                             verify.diff 12, a raised exception 32
neither                942   a distance, a root, a proof, a floor block, a verdict
```

240 of 1359 vectors are the ones that can carry a hole of this kind. The set does not currently
name them anywhere.

## What follows for practice

1. **The index should name the acceptances.** `index.json` records the entrypoint census and the
   honest remainder; it does not record which vectors assert that an input was accepted. That list
   is what a reviewer needs after any rule is tightened, and it is derivable from the vectors, so
   it costs nothing but the writing.
2. **An acceptance vector that a repair retires is a finding, not a regeneration.** The lab's rule
   already says a moved core is a finding about the implementation. The case here is the mirror of
   it — a moved core that is a finding about the *set* — and it needs its own note in the commit,
   which is what this document and the set's README now are.
3. **The generator's drop notice is weaker than its refusal, and today the interesting event
   arrived as a drop.** `DROPPED <id> -- the run no longer produces it` reads the same for a
   rewritten test and for a repaired defect. Replaying each dropped vector against the tree that
   dropped it separates the two — 84 and 5 here — and the generator could print that split rather
   than leaving it to be run by hand. Owed.
4. **A hole that no rule and no test reaches is not visible by any of this.** Both repairs came
   from an adversarial pass, not from the set. The set recorded the defects; it did not find them,
   and nothing here suggests it could have.

## What the repairs did to what the set can see

`conformance/v8/mutation_coverage.json` was rerun against the regenerated set and the repaired
tree. The catalogue is unchanged, and so is every number that matters: 22 viable mutants, 17
caught, 5 missed, 4 controls, 0 controls caught, 15 regions, detection rate 0.7727. **The miss list
is identical** — `merkle-leaf-prefix-dropped`, `keys-small-order-accepted`,
`cert-small-order-key-accepted`, `cert-noncanonical-point-accepted`, `log-duplicate-id-accepted` —
in both runs.

The set grew from 877 vectors to 1359 in the same regeneration. Those 482 additional vectors caught
no mutation the old set missed. What they changed is the number of witnesses per catch
(`floor-alpha-off-by-one` from 61 vectors to 296, `keys-tag-separator-dropped` from 191 to 303),
which is redundancy and not detection power. A set can grow by half again and see exactly what it
saw before.

## Limits

One pair of repairs, one set, one day. This document does not estimate how many of the 240
acceptance vectors pin something a later rule will refuse; that number is unknown and nothing here
bounds it.

The 84-of-89 split was measured against the set as it stood before the regeneration. That set is no
longer on disk, so the split is re-derivable only from the diff of this commit, not from the
committed set alone.

The mutation run says the repairs did not change what the vectors can detect, for the 26 edits in
this catalogue. It says nothing about behaviours the catalogue does not name, and the catalogue was
written by the same lab as the implementation.

The sources are the builder's own tests. A defect no test in `tests/test_v8_*.py` reaches was never
recorded as a vector, so it is not pinned, not defended, and not visible here either.

## Appendix — the 89 retired addresses

Retired because an answer moved (5), all `cert.check`:

```
41a682e2176a7895f62be318a3d85f17741073ad89b375e355a28d234cb5cfbc
4bbbb3cbbfbbf19fe3745fdb7d50682794c68ed7a31c968bbcc058766a8eafb0
6e1391de536c7ea6afc72fa89035f80e3e571b80ea4a3cc0df437c3663fe281b
a6f0718baadacc91c75b06e3a1f4ebe013a74d1c38a784f32064da466e6228dd
b8777ac429346978c297cac977bd379adfa548111a10a4c68c07e507ab637ffe
```

What answers for each of them now:

```
41a682e2... -> 7b803597f0b0d9b44378f221a1163d817741a9ffdf88a159e9777baa05a3b29e  ok: true
b8777ac4... -> 0166328dabb5ea1fb762b74505550178b035b43a1efbe7edb75b12048f5549ac  ok: true
4bbbb3cb... -> b3c888c1b1994d68e86b4cf9e860b99d1c24ce51d7e1bfa7f1c0d8d6d5eb628a  ["schema"]
6e1391de... -> 63c69306e462d385eda70b45c43a4ec7202faedcfb19a951c87ed6d8620d492d  ["schema"]
a6f0718b... -> 24e41a00714a2815a0c60b30d1de8c1c64473cc99ee3f148ea53478256a26891  ok: true
```

The last of those is an address the set already held. Removing the duplicate `battery` ref makes
`test_a_role_allowed_on_a_fingerprint_is_accepted[battery]` build the same cert as the plain
fingerprint round trip, so the call folds into the existing vector and adds a source to it. The set
is one address smaller for it.

Retired because a source now calls with different arguments (84). Each of these still reproduced
against the tree that retired it:

```
cert.check (20)
104d319a337bf0a97e5982fb55deb8a4d0aae6fbc8622e6d0fa8d31982fb6b8d
1352b457b7a68d0011db6a4f00ffe92a7b53d8fc6e2f88366132ce5214fd8b97
1c5cffdc54762632d07d898dd157177a6261b9a886b53f696bb2f456a58c5351
340098aa33f24627d2af8e62f0b288a938fe71588996b505e65feca52bc5f41b
5bffecb43aaf0fad196872361343c3912d7267eb3b4e01ac984485f777ea684b
62317da64042039b6d82e66216c9971f2e518d73c92197c7f6778a40c47de53d
6bdea48fc963c5b514d725a9ba45d2ecbe22a1135e7d6c6bec64deca4d0bb4d4
722eb70a84e92cfc943019507316dfefc43b67beea19079eb952de86c127323e
761213c8f0cc12232e137599b2efa54682c0c72fb77a59c66304844e02e75e4a
780df0e646705edfb3a30cb372bf822553e1b0ccf21f96fe5a5c86a1a8969bb0
80bc1c073800ab1b73ebdd72c7356ab71c6557879ed6359bfac131a4660138c9
925abb0c50cd57f57aa9016f5379add5a02c6b99d308af672c8b173e6969e9f2
a652ade76b590ce2a6921efb109d33eeda4802ebe5fd7aeb7e6a7a237d037c1b
abc0b8821925a1affde4fc6199b1808098b24c5a908a17b6f000a083e5cb2919
b1b63e094009f8cadc11062e3bb2032abc13925497310dcb85a37f26ca39ae6c
bb4b724ea882a5daabce34897fb98f49d5db0fad6905d2f33ca93235e19b336e
bbb0e47465de85cead0c0403ef2ccb42e86d4c9bdb5c3de571c923da9f699162
caaaa50afcbe37528c6b64184a573b8426ea3c099875868c03fe1ab823b08730
d7ead28d09702c946ec956bc9060bbb15ee24700ac980ffe2055069f1531d137
f8d2f63f89907512e87367b15585fa98ef1e585f77090b5d408399190daf7b91

floor.pairwise (50)
0682793fe2caf4723afe2eb9ccbadb11a8b3673f6fe809f8138888f05b927030
0cd9b86317c459e9f33744b651a53c2887958d772cbfcaf3fc66f6997427d4e0
169bc74b9213329cc6c8ec44c4b78c45f3c40f2a3f6c0be6040da61b01e8aa4a
1a616d17a9faa61b51db3d9ffe9bb5e462ae20a030bee1702cc2e6d373e45d92
1e6ed6b59def1d7b3996118265f5bb2a5303518288bc14130e601f8cfa006093
272ac8b134d1c0023ddf20ee73414b7a1dea10ac1dd8043cb45c82fd52d77e43
2d33b898c91d58a01602fd0e51e65e25c8deef83457717cfbc5273511fbad0a1
2e29ee4a3bfc6cf9e97aa389c0a8ba53c13d76f6561e9a8617e3042da4212649
2e943eb432b612332faf950a7595335283d4df3212045d55c24f3873d02f8e34
372d1a56963e3081717cce1b73eebfc4dd391a4a02869010375281accc928657
3833d376b4be9dfeae4325bd28da1713c2339de505872b7087ecc4b63550dc68
3a72d8e423bbcc9509867ec2cb8f712738c1716fc53ea75efff4687df904951c
40268f214282a13ac9d9622218a50af6fc1fa134d15a5a88255e387f6a0d03d3
4831a3a8df5d9adfc0c6496aa74676ce5efdf91d416320ff1767176a62644055
4a25d5ddc074be74de0c15f3ff44418b1f95032075f5c0b10909becf0aad666a
500bb1482b7836e83dafe91eab45935e45295256322e516ce566bcd56880df36
560e154d2ff35710479ee47903f7860dbca91b78bf0dc9b86b0b55ecf91378b9
5800f71e1bdc344ba0c74ebbb5e0e131c647905d34a695c8588e737845d8a334
5dfbe2ac42317690bd2af93c99cab02bc45abdf70b364ed46a92c1799959f897
6579ccd3646e9c12373d85a54d36c53ff03f52e8dfbfde2da8ef89599f943dc7
671eebb1ec1fffa6d2803f450cde987a93db5d63b5e830d74ce5fe0bc48ba860
70d7576edb3612c1a1503fce789acadb4b5d0c2a06e93dc6700e1e633216cfdb
72d8399d864702a48aef2ac2b3c437fe587e1f527df88998aa1e92a052db53a8
7d163e8973d4d6ed4c486d3431108083eacaf93aa15f6ce4e0269032f4e4cee4
87dd92ccb6beaeec1c83fb7769f26f29575ee2f724e5a1334f8f751a5c50e087
8ce00f6bd76d2f341098ed07d5dd3afab145f14b93b949d3d770ec60e10b6225
8e26e3d0b7e5f9f2c809ea2692c7ad1cdf4bb373d68b143840b217f7037d26e9
96e0997bc591d60d3cc690a898687c5bd48bb62f58643d5a8f5e53fb4ce25830
a20b27d3ae749f5c74bce66220de046a3186fca747c455f5959ba2460687596f
ad290a9beabdf0432daf64de7ce986ae8299abce241a7cbf8c817238fd94b531
b2408cf6a4627f4d7f4a01531fd1384d7e6f8be4e5726e7d5a324e3d42858973
ba51bd4e669c80021e82a784e0255030c83083df180c9119b68b9a906f5a076c
beab2571deee0fcb3814bbd6ce5bed62841720754682e8b50633165bf613288a
bf387460f3a9d5a32b442e7eb4c2b52d34fa4450bbd8415eb0b53b0e6e54c5b0
c0456c843216d416f3787b77112787a97203476be906c147fefe0a29246c8640
c956cc7a5340e4d8004f2b174ba1e87f493b67111234861f2683a6640b6985b0
cda6d3249d16659445999618b3a23d8cd89ccc6afe4b64eafad8c8909cb5629f
d2c702a700317e23e38025bfac02c7e6c4a52248800200305cf6780174d59787
d3e36ec8d560ce8ac0ecb85cbfa6cb874d613c71edcbb7432a602496e02f504a
dac7ab6e7ff9e8d31a42e549b77ece601212ae658270d3c30446ea71cf0f9db0
dd88c263faf5d7cc1804a6db871bb285c7c9ea175aede45dd3f2c81a52cf32d1
defecda8b1c70c67c2e564a6c3feaf365ad2ff3c62b3b5b1c9dc368c204c05a1
df6fa4393fb60807a46f8d7e2ef08d324a042854981a1c8f1635f66f740c9fc3
e018286932cbf4d9fc169d5b0da4cc8f190c90b125f47cc2560376f8c8f48298
e758bca6982734700182e27621632112d7135fabe20ebcff33de31754878f64c
e99eb01fa2bb6272e6f64e36170b0cf85ba220f44693db8489af591bcf84921e
f2158f1a34dffa9569ccfe8af9a68eb5337798e5a8401af08bb9f20917556b7c
fd742d74313d9a0e814ccf467861d6380f8d051bd9aabeade119f8fc3f02eee0
fdcd6190c6b55780463d9b83a0faacb8c3e53c59c377d5c00d49935bf398d907
ffe1f3db245fb3883bf0cea74e3540cbd3d7011ed2603864b6e6aa067aab2085

verify.diff (14)
1212116399fc02b94431da3c7ee57d6444ef7c22d0b0c88168662e5269ce27b9
13aec11dc79c4bc0b13d96927217363da136e02df87d4f8d74467052417265aa
1e7b0caecf1b1d2986c26d054ccad4cb20719d5a8387080a713c8c9762cea6db
25574d9129412f9e3416a8511c0e2d8d4c6df057ed5dd13fcc2657a28f565132
3aa2c923230629dc1062d12d801624afa1f6eafc8de408f725983efb4ecded53
4ef0ffda6092f96507321eaa79d1b98141c1c2e4810b5c6991192dcb126252ef
7c95bb85536e50a10f594c52d2d64b8b4c8220f3cabd19eb3adf0ab5c9f1dcd7
7f6dae4d8a01aec5ac6afd61f2efe46032c391b43764a55435bf151ebfb64f1e
80a1414d82fcf9463a1b2bf294d44df6cecbd6b75eaa06c53f0dc6291141622d
87482a84acbbcc15ef3218704ca0ea8375a4d3be27b33f0568d87422b61f5582
892ad8a7693105b6e2438e4f6488c5cfecad3c0767d5d612635b6bc4ab3f6386
8f62bcba92e1551ef7603f93c96fbef1ba78c601e510ad31bb9c1adec8d72072
b9324586e254260d8e8e6eaf59de966c403fdb3875759498e797f6faeb8c054f
d54264b4d5ac1446ced2b991df9a682356a163a8b4766954a9ce596398e03c17
```

The set moved from `set_sha256 cbc213f3cdfd145c5be6a49fafcfe4d06dec615b694e1c71f9e5675109263d1f`
to `6df4c39aa44db83ed8b341e017470e7a702d3ddf968f539e82739981a1a0c858` in this regeneration, and
571 addresses were added — most of them from tests written in `tests/test_v8_log.py` and
`tests/test_v8_verify.py` today, which is ordinary growth and not part of this finding.
