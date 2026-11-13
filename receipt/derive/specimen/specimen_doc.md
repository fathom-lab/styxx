# EXTERNAL-1 headline, bound by derive (specimen, DRAFT profile derive/v0-draft)

Published sentence (README.md on origin/main): a blind three-seat panel upheld <sworn r="r1#upheld" k="derive">23</sworn> of <sworn r="r1#arm" k="derive">100</sworn> sampled accusations, precision <sworn r="r1#precision" k="derive">0.23</sworn>.

r1 is papers/closed-model-frontier/external1_answers.json as committed (git blob 8b578a16a63d16a5a0e94253531dd9dc2dec5503).
The accusation arm is the keys E1-000 to E1-099: a fact of the lab's recipe (external1_packet.py), not of the bytes.

```derive
{
  "arm": {
    "rows": {"format": "json", "at": ""},
    "where": [{"row": "key", "ge": "E1-000"}, {"row": "key", "le": "E1-099"}],
    "op": "count"
  },
  "upheld": {
    "rows": {"format": "json", "at": ""},
    "where": [{"row": "key", "ge": "E1-000"}, {"row": "key", "le": "E1-099"},
              {"row": "value", "eq": "CONTRADICTED"}],
    "op": "count",
    "within": "arm"
  },
  "precision": {
    "rows": {"format": "json", "at": ""},
    "where": [{"row": "key", "ge": "E1-000"}, {"row": "key", "le": "E1-099"}],
    "match": [{"row": "value", "eq": "CONTRADICTED"}],
    "op": "share"
  }
}
```
