package main

// 11.3: checking a receipt against a document and a manifest.

func checkReceipt(rbytes []byte, doc []byte, haveDoc bool, man []byte, haveMan bool) (string, *Refusal) {
	if alt["G17"] && !haveDoc {
		return "", &Refusal{"nothing_to_verify"}
	}
	v, anyDup, err := parseJSON(rbytes)
	if err != nil || anyDup || !validateReceipt(v) {
		return "", &Refusal{"receipt_shape"}
	}
	if len(v.Map["attestations"].Arr) != 0 {
		return "", &Refusal{"attestations_not_empty"}
	}
	core := v.Map["core"]
	ct, cerr := canonical(core)
	if cerr != nil || shaHex([]byte(ct)) != v.Map["digest"].Str {
		return "", &Refusal{"receipt_digest"}
	}
	if core.Map["profile"].Str != "rn-inline/v0-draft" {
		return "", &Refusal{"profile_not_checkable"}
	}
	if !haveDoc {
		return "", &Refusal{"nothing_to_verify"} // GUESS G-R3
	}
	docObj := core.Map["document"]
	if shaHex(doc) != docObj.Map["inline_sha256"].Str {
		return "", &Refusal{"document_mismatch"}
	}
	mine, ref := checkDocument(Inputs{Doc: doc, HaveDoc: true, Name: docObj.Map["name"],
		Commit: core.Map["commit"], Manifest: man, HaveMan: haveMan})
	if ref != nil {
		if alt["G18"] {
			return "DOES-NOT-REDERIVE", nil
		}
		return "", ref
	}
	_, dg := receiptOf(mine)
	if dg == v.Map["digest"].Str {
		return "REDERIVES", nil
	}
	return "DOES-NOT-REDERIVE", nil
}
