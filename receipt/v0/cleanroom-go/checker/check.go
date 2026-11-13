package main

// Sections 6, 7, 8 and 9: judging each declaration and building the core.

import (
	"bytes"
	"fmt"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

type Row struct {
	At       int
	Start    *int
	End      *int
	Receipt  *string
	Kind     *string
	ClaimSha *string
	Claim    *string
	Verdict  string
	Reason   *string
	Detail   *JVal
	Resolved *string
	Prov     *JVal
}

func (r *Row) set(verdict, reason string, detail *JVal) {
	if alt["G7"] && verdict == "HELD" {
		detail = nil
	}
	r.Verdict = verdict
	if reason == "" {
		r.Reason = nil
	} else {
		rs := reason
		r.Reason = &rs
	}
	if detail == nil {
		detail = newObj().val()
	}
	r.Detail = detail
}

func (r *Row) toJ() *JVal {
	o := newObj().set("at", jInt(r.At))
	if r.Start == nil {
		o.set("start", jNull()).set("end", jNull())
	} else {
		o.set("start", jInt(*r.Start)).set("end", jInt(*r.End))
	}
	o.set("receipt", jStrOrNull(r.Receipt)).set("kind", jStrOrNull(r.Kind))
	o.set("claim_sha256", jStrOrNull(r.ClaimSha)).set("claim", jStrOrNull(r.Claim))
	o.set("verdict", jStr(r.Verdict)).set("reason", jStrOrNull(r.Reason)).set("detail", r.Detail)
	if r.Resolved != nil {
		o.set("resolved_sha256", jStr(*r.Resolved))
		o.set("provenance", r.Prov)
	}
	return o.val()
}

var externalKinds = map[string]bool{"tool_stdout": true, "tool_stderr": true, "file_read": true,
	"http_fetch": true, "harness_note": true, "test_report": true, "attestation": true}
var authorKinds = map[string]bool{"agent_output": true, "agent_file_write": true, "agent_message": true}

func isASCIIWS(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == 0x0B || c == 0x0C || c == '\r'
}

func allASCIIWS(b []byte) bool {
	for _, c := range b {
		if !isASCIIWS(c) {
			return false
		}
	}
	return true
}

// ---- 7.2 numeric form ----

var dashSet = map[rune]bool{0x00AD: true, 0x058A: true, 0x05BE: true, 0x1400: true, 0x1806: true,
	0x2010: true, 0x2011: true, 0x2012: true, 0x2013: true, 0x2014: true, 0x2015: true, 0x2E17: true,
	0x2E1A: true, 0x2E3A: true, 0x2E3B: true, 0x2E40: true, 0x2E5D: true, 0x301C: true, 0x3030: true,
	0x30A0: true, 0xFE31: true, 0xFE32: true, 0xFE58: true, 0xFE63: true, 0xFF0D: true, 0x10EAD: true}

func isTokenRune(r rune) bool {
	if isLN(r) || dashSet[r] || r == 0x2212 || r == 0x00B1 {
		return true
	}
	return strings.ContainsRune("_.,+-%/:", r)
}

var numGrammar = regexp.MustCompile(`^[-+\x{2212}]?(?:(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?|\.[0-9]+)%?$`)

// numericForm returns the stripped token, or a failure reason and detail.
func numericForm(inner string) (string, string, *JVal) {
	var tokens []string
	var cur []rune
	flush := func() {
		if len(cur) > 0 {
			hasNd := false
			for _, r := range cur {
				if isNd(r) {
					hasNd = true
					break
				}
			}
			if hasNd {
				tokens = append(tokens, string(cur))
			}
			cur = cur[:0]
		}
	}
	for _, r := range inner {
		if isTokenRune(r) {
			cur = append(cur, r)
		} else {
			flush()
		}
	}
	flush()
	det := func() *JVal {
		a := jArr()
		for _, t := range tokens {
			a.Arr = append(a.Arr, jStr(t))
		}
		return newObj().set("digit_bearing_tokens", a).val()
	}
	if len(tokens) != 1 {
		return "", "number_count", det()
	}
	t := strings.TrimRight(tokens[0], ".,:")
	if !numGrammar.MatchString(t) {
		return "", "number_grammar", det()
	}
	return t, "", nil
}

// printedDec reads a stripped token that passed the grammar.
func printedDec(tok string) (Dec, int) {
	s := strings.ReplaceAll(tok, ",", "")
	s = strings.ReplaceAll(s, "%", "")
	s = strings.ReplaceAll(s, "−", "-")
	d := Dec{}
	if strings.HasPrefix(s, "-") {
		d.Neg = true
		s = s[1:]
	} else if strings.HasPrefix(s, "+") {
		s = s[1:]
	}
	ip, fp, _ := strings.Cut(s, ".")
	c := strings.TrimLeft(ip+fp, "0")
	if c == "" {
		c = "0"
	}
	d.Coef = c
	d.Exp = bigFromInt(-len(fp))
	if alt["G21"] {
		d.Lead = len(ip) - len(strings.TrimLeft(ip, "0"))
		if strings.TrimLeft(ip, "0") == "" && ip != "" {
			d.Lead = len(ip) - 1 // plain notation writes one 0 itself
		}
	}
	return d, len(fp)
}

// ---- 7.4 needle ----

func findNeedles(inner []byte) [][]byte {
	var out [][]byte
	for p := 0; p < len(inner); {
		if inner[p] == '`' {
			next, closed, cs, ce := inlineCode(inner, p)
			if closed {
				out = append(out, inner[cs:ce])
			}
			p = next
			continue
		}
		p++
	}
	return out
}

// ---- 7.6 hex runs ----

func isAlnumU(c byte) bool {
	return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_'
}

func hexRuns(b []byte) (lens []int, starts []int) {
	for i := 0; i < len(b); {
		if hexv(b[i]) < 0 {
			i++
			continue
		}
		j := i
		for j < len(b) && hexv(b[j]) >= 0 {
			j++
		}
		glued := (i > 0 && isAlnumU(b[i-1])) || (j < len(b) && isAlnumU(b[j]))
		if alt["G26"] && !glued {
			if pr, _ := utf8.DecodeLastRune(b[:i]); i > 0 && (unicode.IsLetter(pr) || unicode.IsDigit(pr)) {
				glued = true
			}
			if nr, _ := utf8.DecodeRune(b[j:]); j < len(b) && (unicode.IsLetter(nr) || unicode.IsDigit(nr)) {
				glued = true
			}
		}
		if !glued {
			lens = append(lens, j-i)
			starts = append(starts, i)
		}
		i = j
	}
	return
}

// ---- base64 (6.10) ----

func b64val(c byte) int {
	switch {
	case c >= 'A' && c <= 'Z':
		return int(c - 'A')
	case c >= 'a' && c <= 'z':
		return int(c-'a') + 26
	case c >= '0' && c <= '9':
		return int(c-'0') + 52
	case c == '+':
		return 62
	case c == '/':
		return 63
	}
	return -1
}

func strictB64(s string) ([]byte, bool) {
	if len(s)%4 != 0 {
		return nil, false
	}
	out := make([]byte, 0, len(s)/4*3)
	for i := 0; i < len(s); i += 4 {
		q := s[i : i+4]
		last := i+4 == len(s)
		pad := 0
		if last {
			if q[3] == '=' {
				pad = 1
				if q[2] == '=' {
					pad = 2
				}
			}
		}
		var v [4]int
		for k := 0; k < 4-pad; k++ {
			v[k] = b64val(q[k])
			if v[k] < 0 {
				return nil, false
			}
		}
		n := v[0]<<18 | v[1]<<12 | v[2]<<6 | v[3]
		switch pad {
		case 0:
			out = append(out, byte(n>>16), byte(n>>8), byte(n))
		case 1:
			out = append(out, byte(n>>16), byte(n>>8))
		case 2:
			out = append(out, byte(n>>16))
		}
	}
	return out, true
}

// ---- receipt bytes helpers ----

func lineSlice(b []byte, a, z int) ([]byte, bool) {
	ls := splitLines(b)
	if z > len(ls) {
		return nil, false
	}
	s := ls[a-1].start
	e := ls[z-1].end
	if e > s && b[e-1] == '\n' {
		e--
	}
	return b[s:e], true
}

func leafText(v *JVal) string {
	var s string
	switch v.K {
	case JStr:
		s = v.Str
	case JBool:
		if v.B {
			s = "True"
		} else {
			s = "False"
		}
	case JSpecial:
		s = v.Num
	case JNum:
		s = parseJSONNumber(v.Num).SciString()
	}
	rs := wtf8Runes(s)
	if len(rs) > 80 && !(alt["G15"] && v.K == JNum) {
		rs = rs[:80]
	}
	for i, r := range rs {
		if r >= 0xD800 && r <= 0xDFFF {
			rs[i] = '?'
		}
	}
	return string(rs)
}

func leafTypeName(v *JVal) string {
	switch v.K {
	case JObj:
		return "object"
	case JArr:
		return "array"
	}
	return "null"
}

// ---- the judge ----

type CheckResult struct {
	Core *JVal
}

func judge(d *Decl, m *Manifest, haveManifest bool) *Row {
	r := &Row{At: d.At, Receipt: d.Receipt, Kind: d.Kind}
	if d.Mark != "" {
		det := newObj()
		if d.Raw != nil {
			det.set("raw", jStr(*d.Raw))
		}
		r.set("MALFORMED", d.Mark, det.val())
		return r
	}
	inner := d.Inner
	innerS := string(inner)
	if allASCIIWS(inner) {
		r.set("MALFORMED", "empty_span", nil)
		return r
	}
	if cp := utf8.RuneCount(inner); cp > 300 {
		r.set("MALFORMED", "length_cap", newObj().set("code_points", jInt(cp)).set("bytes", jInt(len(inner))).set("cap", jInt(300)).val())
		return r
	}
	kind := *d.Kind
	switch kind {
	case "numeric", "quote", "hash", "absent":
	case "exec":
		r.set("MALFORMED", "kind_reserved", newObj().set("kind", jStr(kind)).val())
		return r
	default:
		r.set("MALFORMED", "kind_unknown", newObj().set("kind", jStr(kind)).val())
		return r
	}
	ref := parseRef(*d.Receipt)
	if ref == nil {
		r.set("MALFORMED", "receipt_form", newObj().set("receipt", jStr(*d.Receipt)).val())
		return r
	}
	if ref.HasFrag && kind == "absent" {
		r.set("MALFORMED", "absent_over_partial", newObj().set("receipt", jStr(*d.Receipt)).val())
		return r
	}
	if ref.HasFrag && kind == "hash" {
		r.set("MALFORMED", "hash_over_partial", newObj().set("receipt", jStr(*d.Receipt)).val())
		return r
	}
	for _, c := range innerS {
		if c == 0x202D || c == 0x202E {
			r.set("MALFORMED", "directional_override", newObj().set("code_point", jStr(fmt.Sprintf("U+%04X", c))).val())
			return r
		}
	}
	// h. form
	var numTok string
	var needle []byte
	var stated string
	switch kind {
	case "numeric":
		t, reason, det := numericForm(innerS)
		if reason != "" {
			r.set("MALFORMED", reason, det)
			return r
		}
		numTok = t
	case "quote", "absent":
		ns := findNeedles(inner)
		if len(ns) != 1 {
			if alt["G5"] {
				r.set("MALFORMED", "needle_count", newObj().set("occurrences", jInt(len(ns))).val())
				return r
			}
			r.set("MALFORMED", "needle_count", nil)
			return r
		}
		if allASCIIWS(ns[0]) || (alt["G10"] && strings.TrimFunc(string(ns[0]), unicode.IsSpace) == "") {
			r.set("MALFORMED", "needle_empty", nil)
			return r
		}
		needle = ns[0]
	case "hash":
		lens, starts := hexRuns(inner)
		n64, bad := 0, false
		at := -1
		for i, l := range lens {
			switch l {
			case 64:
				n64++
				at = starts[i]
			case 32, 40, 96, 128:
				bad = true
			}
		}
		if n64 != 1 || bad {
			a := jArr()
			for _, l := range lens {
				a.Arr = append(a.Arr, jInt(l))
			}
			r.set("MALFORMED", "digest_form", newObj().set("hex_runs", a).val())
			return r
		}
		stated = strings.ToLower(innerS[at : at+64])
	}
	// i. resolution
	if ref.Form != "rn" {
		r.set("UNRESOLVED", "no_repository", nil)
		return r
	}
	if !haveManifest {
		r.set("UNRESOLVED", "manifest_absent", nil)
		return r
	}
	if !m.Intact {
		r.set("UNRESOLVED", "manifest_integrity", nil)
		return r
	}
	rungName := ""
	switch {
	case m.Rung.K == JNull:
		rungName = "undeclared"
	case m.Rung.K == JStr && (m.Rung.Str == "L1" || m.Rung.Str == "L2"):
		rungName = m.Rung.Str
	case alt["G13"] && m.Rung.K != JStr:
		r.set("UNRESOLVED", "manifest_integrity", nil)
		return r
	default:
		r.set("UNRESOLVED", "rung_unknown", nil)
		return r
	}
	entry, ok := m.Receipts.Get(ref.ID)
	if !ok {
		r.set("UNRESOLVED", "manifest_id_missing", nil)
		return r
	}
	if entry.K != JObj {
		r.set("UNRESOLVED", "manifest_integrity", nil)
		return r
	}
	esha, ok := entry.Get("sha256")
	if !ok || esha.K != JStr || !isHex64Lower(esha.Str) {
		r.set("UNRESOLVED", "manifest_integrity", nil)
		return r
	}
	kos, ok := entry.Get("kind_of_source")
	if !ok || kos.K != JStr {
		r.set("MALFORMED", "kind_of_source_unknown", nil)
		return r
	}
	if authorKinds[kos.Str] || m.authoredHas(esha.Str) {
		r.set("MALFORMED", "receipt_author_minted", nil)
		return r
	}
	if !externalKinds[kos.Str] {
		r.set("MALFORMED", "kind_of_source_unknown", nil)
		return r
	}
	prov := newObj().set("form", jStr("rn")).set("harness", jStr(m.Harness)).
		set("rung", jStr(rungName)).set("kind_of_source", jStr(kos.Str)).val()
	resolved := esha.Str
	withProv := func() { r.Resolved = &resolved; r.Prov = prov }
	var rb []byte
	hasBytes := false
	if bv, ok := entry.Get("bytes"); ok {
		if bv.K != JStr {
			r.set("UNRESOLVED", "manifest_integrity", nil)
			return r
		}
		dec, ok := strictB64(bv.Str)
		if !ok || shaHex(dec) != esha.Str {
			r.set("UNRESOLVED", "manifest_integrity", nil)
			return r
		}
		rb, hasBytes = dec, true
	} else if kind != "hash" {
		r.set("UNRESOLVED", "manifest_bytes_absent", nil)
		return r
	}
	_ = hasBytes
	// fragments (6.11, 6.12)
	var hay []byte
	var leaf *JVal
	haveLeaf := false
	if ref.HasFrag && ref.IsLine {
		s, ok := lineSlice(rb, ref.LA, ref.LB)
		if !ok {
			if alt["G6"] {
				r.set("MALFORMED", "anchor_out_of_range", newObj().set("receipt", jStr(*d.Receipt)).val())
				return r
			}
			r.set("MALFORMED", "anchor_out_of_range", nil)
			return r
		}
		hay = s
	} else if ref.HasFrag {
		root, _, err := parseJSON(rb)
		if err != nil {
			r.set("MALFORMED", "receipt_not_json", nil)
			return r
		}
		cur := root
		for _, tok := range ref.Pointer {
			switch cur.K {
			case JObj:
				x, ok := cur.Map[tok]
				if !ok {
					r.set("MALFORMED", "pointer_unresolvable", nil)
					return r
				}
				if cur.Dups[tok] {
					r.set("MALFORMED", "pointer_ambiguous", nil)
					return r
				}
				cur = x
			case JArr:
				if !(tok == "0" || isAllDigitsNoLeadingZero(tok)) {
					r.set("MALFORMED", "pointer_unresolvable", nil)
					return r
				}
				idx, _, _ := posDecimal(tok)
				if tok == "0" {
					idx = 0
				}
				if idx >= len(cur.Arr) {
					r.set("MALFORMED", "pointer_unresolvable", nil)
					return r
				}
				cur = cur.Arr[idx]
			default:
				r.set("MALFORMED", "pointer_unresolvable", nil)
				return r
			}
		}
		leaf, haveLeaf = cur, true
	} else {
		hay = rb
	}
	// j. absent completeness
	if kind == "absent" {
		cv, ok := entry.Get("complete")
		if !ok || cv.K != JBool {
			r.set("UNRESOLVED", "manifest_no_completeness", nil)
			return r
		}
		if !cv.B {
			withProv()
			r.set("MALFORMED", "absent_over_partial", newObj().set("complete", jBool(false)).val())
			return r
		}
	}
	// k. comparison
	withProv()
	switch kind {
	case "hash":
		det := newObj().set("stated", jStr(stated)).set("receipt_sha256", jStr(esha.Str)).val()
		if stated == esha.Str {
			r.set("HELD", "", det)
		} else {
			r.set("FAILED", "digest_mismatch", det)
		}
	case "absent":
		n := bytes.Count(rb, needle)
		det := newObj().set("needle_bytes", jInt(len(needle))).set("haystack_bytes", jInt(len(rb))).set("complete", jBool(true)).val()
		if n > 0 {
			r.set("FAILED", "needle_present", det)
		} else {
			r.set("HELD", "", det)
		}
	case "quote":
		if haveLeaf {
			if leaf.K != JStr {
				r.set("MALFORMED", "leaf_not_string", nil)
				return r
			}
			if hasLoneSurrogate(leaf.Str) {
				r.set("MALFORMED", "leaf_not_string", newObj().set("note", jStr("leaf is not encodable as UTF-8")).val())
				return r
			}
			hay = []byte(leaf.Str)
		} else {
			narrows := false
			if ref.HasFrag && ref.IsLine {
				trimmed := bytes.TrimRight(rb, "\n")
				if len(trimmed) == 0 && !alt["G8"] {
					narrows = len(rb) < 16
				} else {
					narrows = !bytes.Equal(hay, trimmed) || len(rb) < 16
				}
			}
			if len(needle) < 16 && !narrows && !alt["C3"] {
				r.set("MALFORMED", "short_needle", newObj().set("needle_bytes", jInt(len(needle))).set("minimum_bytes", jInt(16)).val())
				return r
			}
		}
		n := bytes.Count(hay, needle)
		det := newObj().set("needle_bytes", jInt(len(needle))).set("haystack_bytes", jInt(len(hay))).set("occurrences", jInt(n)).val()
		if n > 0 {
			r.set("HELD", "", det)
		} else {
			r.set("FAILED", "needle_missing", det)
		}
	case "numeric":
		if !haveLeaf {
			v, _, err := parseJSON(hay)
			if err != nil {
				r.set("MALFORMED", "receipt_not_json", nil)
				return r
			}
			leaf = v
		}
		switch leaf.K {
		case JObj, JArr, JNull:
			r.set("MALFORMED", "leaf_not_scalar", newObj().set("leaf_type", jStr(leafTypeName(leaf))).val())
			return r
		case JStr, JBool, JSpecial:
			r.set("MALFORMED", "leaf_not_numeric", newObj().set("leaf", jStr(leafText(leaf))).val())
			return r
		}
		rd := parseJSONNumber(leaf.Num)
		if rd.Adjusted().Cmp(bigFromInt(320)) > 0 {
			r.set("MALFORMED", "leaf_not_numeric", newObj().set("leaf", jStr(leafText(leaf))).val())
			return r
		}
		pd, f := printedDec(numTok)
		printed, _ := pd.roundPlain(f)
		rr, rzero := rd.roundPlain(f)
		det := newObj().set("printed_token", jStr(numTok)).set("printed", jStr(printed)).
			set("receipt", jStr(rd.SciString())).set("receipt_rounded", jStr(rr)).
			set("fractional_digits", jInt(f)).set("rounding", jStr("ROUND_HALF_EVEN")).val()
		switch {
		case printed != rr:
			r.set("FAILED", "value_mismatch", det)
		case !rd.IsZero() && rzero && !alt["C2"]:
			r.set("FAILED", "rounds_to_zero", det)
		default:
			r.set("HELD", "", det)
		}
	}
	return r
}

// ---- 9: the core ----

type Inputs struct {
	Doc      []byte
	HaveDoc  bool
	Name     *JVal // string (WTF-8) expected
	Commit   *JVal // null or string
	Manifest []byte
	HaveMan  bool
}

var commitRe = regexp.MustCompile(`^([0-9a-f]{40}|[0-9a-f]{64})$`)

func checkDocument(in Inputs) (*JVal, *Refusal) {
	if !in.HaveDoc {
		return nil, &Refusal{"nothing_to_verify"}
	}
	if alt["G23"] && (in.Name == nil || in.Name.K == JNull) {
		return nil, &Refusal{"nothing_to_verify"}
	}
	if in.Name == nil || in.Name.K != JStr || hasLoneSurrogate(in.Name.Str) {
		return nil, &Refusal{"name_form"}
	}
	commit := jNull()
	if in.Commit != nil && in.Commit.K != JNull {
		if in.Commit.K != JStr || !commitRe.MatchString(in.Commit.Str) {
			return nil, &Refusal{"commit_form"}
		}
		commit = in.Commit
	}
	var m *Manifest
	if in.HaveMan {
		var ref *Refusal
		m, ref = readManifest(in.Manifest)
		if ref != nil {
			return nil, ref
		}
	}
	ds := scanDocument(in.Doc)
	var rows []*Row
	for _, d := range ds.Decls {
		row := judge(d, m, in.HaveMan)
		if ds.HasCanon {
			s, e := d.Start, d.End
			row.Start, row.End = &s, &e
			cs := shaHex(d.Inner)
			row.ClaimSha = &cs
			if utf8.RuneCount(d.Inner) <= 300 {
				cl := string(d.Inner)
				row.Claim = &cl
			}
		}
		rows = append(rows, row)
	}
	counts := map[string]int{"HELD": 0, "FAILED": 0, "UNRESOLVED": 0, "MALFORMED": 0}
	rungs := newObj()
	rungN := map[string]int{}
	var rungOrder []string
	spans := jArr()
	for _, row := range rows {
		counts[row.Verdict]++
		k := "unresolved"
		if row.Prov != nil {
			k = row.Prov.Map["rung"].Str
		}
		if _, ok := rungN[k]; !ok {
			rungOrder = append(rungOrder, k)
		}
		rungN[k]++
		spans.Arr = append(spans.Arr, row.toJ())
	}
	for _, k := range rungOrder {
		rungs.set(k, jInt(rungN[k]))
	}
	verdict := ""
	switch {
	case ds.Malformed != nil:
		verdict = "SWORN-FAILED"
	case len(rows) == 0:
		verdict = "UNSWORN"
	case counts["FAILED"] > 0 || counts["MALFORMED"] > 0:
		verdict = "SWORN-FAILED"
	case counts["HELD"] == 0:
		verdict = "SWORN-UNCHECKED"
	default:
		verdict = "SWORN-HELD"
	}
	canon := jNull()
	if ds.HasCanon && ds.Malformed == nil {
		canon = jStr(shaHex(ds.Canonical))
	}
	mdig := jNull()
	if m != nil && m.Digest != nil {
		mdig = jStr(*m.Digest)
	}
	dm := jNull()
	if ds.Malformed != nil {
		dm = ds.Malformed
	}
	core := newObj().
		set("schema", jStr("receipt/v0-draft")).
		set("profile", jStr("rn-inline/v0-draft")).
		set("format", jStr("sworn/0.1")).
		set("document", newObj().set("name", in.Name).set("inline_sha256", jStr(shaHex(in.Doc))).set("canonical_sha256", canon).val()).
		set("commit", commit).
		set("manifest_digest", mdig).
		set("spans", spans).
		set("counts", newObj().set("HELD", jInt(counts["HELD"])).set("FAILED", jInt(counts["FAILED"])).set("UNRESOLVED", jInt(counts["UNRESOLVED"])).set("MALFORMED", jInt(counts["MALFORMED"])).val()).
		set("sworn_total", jInt(len(rows))).
		set("unresolved", jInt(counts["UNRESOLVED"])).
		set("document_verdict", jStr(verdict)).
		set("document_malformed", dm).
		set("rungs", rungs.val()).val()
	// 11.2 guard
	if _, err := canonical(core); err != nil {
		return nil, &Refusal{err.(*CanonError).Code}
	}
	return core, nil
}

// receiptOf wraps a core as a verdict receipt (9.4) and returns its canonical text.
func receiptOf(core *JVal) (string, string) {
	ct, _ := canonical(core)
	dg := shaHex([]byte(ct))
	rc := newObj().set("core", core).set("digest", jStr(dg)).set("attestations", jArr()).val()
	t, _ := canonical(rc)
	return t, dg
}
