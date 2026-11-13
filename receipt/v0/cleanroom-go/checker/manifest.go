package main

// Section 5: the manifest.

import (
	"sort"
	"strings"
)

type Manifest struct {
	Root     *JVal
	Harness  string
	Receipts *JVal          // object
	Authored []string       // rebuilt: lowercased, sorted
	Rung     *JVal          // as given (nil == null)
	Intact   bool
	Digest   *string        // recomputed digest, nil when the core has no canonical form
}

type Refusal struct{ Code string }

func (r *Refusal) Error() string { return r.Code }

func isHex64Any(s string) bool { return len(s) == 64 && isHexStr(s) }

func isHex64Lower(s string) bool {
	if len(s) != 64 {
		return false
	}
	for i := 0; i < 64; i++ {
		c := s[i]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return false
		}
	}
	return true
}

// readManifest applies 5.5 and computes 5.4 and 5.6.
func readManifest(b []byte) (*Manifest, *Refusal) {
	root, anyDup, err := parseJSON(b)
	if err != nil {
		return nil, &Refusal{"manifest_spec"}
	}
	if anyDup {
		return nil, &Refusal{"manifest_form"}
	}
	if root.K != JObj {
		return nil, &Refusal{"manifest_spec"}
	}
	if s, ok := root.Get("spec"); !ok || s.K != JStr || s.Str != "sworn/manifest/0.2" {
		return nil, &Refusal{"manifest_spec"}
	}
	receipts, ok := root.Get("receipts")
	if !ok || receipts.K == JNull {
		receipts = newObj().val()
	}
	if receipts.K != JObj {
		return nil, &Refusal{"manifest_receipts_shape"}
	}
	auth, ok := root.Get("authored_sha256")
	if !ok || auth.K == JNull {
		auth = jArr()
	}
	if auth.K != JArr {
		return nil, &Refusal{"manifest_authored_shape"}
	}
	var authored []string
	for _, a := range auth.Arr {
		if a.K != JStr || !isHex64Any(a.Str) {
			return nil, &Refusal{"manifest_authored_shape"}
		}
		authored = append(authored, strings.ToLower(a.Str)) // ASCII only here
	}
	sort.Strings(authored)
	for _, k := range []string{"harness", "turn", "minted_at"} {
		if v, ok := root.Get(k); !ok || v.K != JStr {
			return nil, &Refusal{"manifest_form"}
		}
	}
	m := &Manifest{Root: root, Receipts: receipts, Authored: authored}
	m.Harness = root.Map["harness"].Str
	rung, ok := root.Get("rung")
	if !ok {
		rung = jNull()
	}
	m.Rung = rung
	// 5.4 the core, rebuilt
	aArr := jArr()
	for _, a := range authored {
		aArr.Arr = append(aArr.Arr, jStr(a))
	}
	core := newObj().
		set("spec", root.Map["spec"]).
		set("harness", root.Map["harness"]).
		set("turn", root.Map["turn"]).
		set("minted_at", root.Map["minted_at"]).
		set("authored_sha256", aArr).
		set("receipts", receipts).
		set("rung", rung).val()
	if alt["G11"] {
		for _, k := range []string{"authored_sha256", "receipts", "rung"} {
			if x, ok := root.Get(k); !ok {
				delete(core.Map, k)
				var ks []string
				for _, kk := range core.Keys {
					if kk != k {
						ks = append(ks, kk)
					}
				}
				core.Keys = ks
			} else if x.K == JNull {
				core.Map[k] = x
			}
		}
	}
	if ct, err := canonical(core); err == nil {
		d := shaHex([]byte(ct))
		m.Digest = &d
		m.Intact = true
		if dg, ok := root.Get("digest"); ok && dg.K != JNull {
			if dg.K != JStr || (dg.Str != d && !(alt["G12"] && strings.ToLower(dg.Str) == d)) {
				m.Intact = false
			}
		}
	}
	return m, nil
}

func (m *Manifest) authoredHas(s string) bool {
	i := sort.SearchStrings(m.Authored, s)
	return i < len(m.Authored) && m.Authored[i] == s
}
