package main

// Section 2: the span grammar.

import (
	"unicode/utf8"
)

type Decl struct {
	At       int
	Receipt  *string
	Kind     *string
	Mark     string // "" or a section-2 reason
	Raw      *string
	OpenEnd  int // byte offset just past the opener
	Paired   bool
	Inner    []byte
	CloserAt int
	Start    int
	End      int
}

type DocScan struct {
	Malformed *JVal // document_malformed object, or nil
	Decls     []*Decl
	Canonical []byte // nil when there is no canonical text
	HasCanon  bool
}

// firstInvalidUTF8 returns the offset of the first ill-formed sequence or -1.
func firstInvalidUTF8(b []byte) int {
	for i := 0; i < len(b); {
		r, sz := utf8.DecodeRune(b[i:])
		if r == utf8.RuneError && sz <= 1 {
			return i
		}
		i += sz
	}
	return -1
}

type lineSpan struct{ start, end int } // end exclusive, LF included when present

func splitLines(b []byte) []lineSpan {
	var out []lineSpan
	s := 0
	for i := 0; i < len(b); i++ {
		if b[i] == '\n' {
			out = append(out, lineSpan{s, i + 1})
			s = i + 1
		}
	}
	if s < len(b) {
		out = append(out, lineSpan{s, len(b)})
	}
	return out
}

func isFenceLine(line []byte) bool {
	i := 0
	for i < len(line) && i < 3 && line[i] == ' ' {
		i++
	}
	n := 0
	for i+n < len(line) && line[i+n] == '`' {
		n++
	}
	return n >= 3
}

func isWordByte(c byte) bool {
	return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-'
}

func lower(c byte) byte {
	if c >= 'A' && c <= 'Z' {
		return c + 32
	}
	return c
}

// inlineCodeEnd: at p is a backtick. Returns the new position, and whether a
// code span closed (with the content range).
func inlineCode(b []byte, p int) (next int, closed bool, cs, ce int) {
	n := 0
	for p+n < len(b) && b[p+n] == '`' {
		n++
	}
	q := p + n
	for q < len(b) && b[q] != '\n' {
		if b[q] == '`' {
			m := 0
			for q+m < len(b) && b[q+m] == '`' {
				m++
			}
			if m == n {
				return q + m, true, p + n, q
			}
			q += m
			continue
		}
		q++
	}
	return p + n, false, 0, 0
}

func indexFrom(b []byte, from int, pat string) int {
	if from > len(b) {
		return -1
	}
	for i := from; i+len(pat) <= len(b); i++ {
		if string(b[i:i+len(pat)]) == pat {
			return i
		}
	}
	return -1
}

func refByteOK(c byte) bool { return c != '"' && c != '<' && c != '>' && c != '\r' && c != '\n' }

// matchOpener returns (end, ref, kind, ok).
func matchOpener(b []byte, p int) (int, string, string, bool) {
	const a = `<sworn r="`
	if p+len(a) > len(b) || string(b[p:p+len(a)]) != a {
		return 0, "", "", false
	}
	q := p + len(a)
	rs := q
	for q < len(b) && refByteOK(b[q]) {
		q++
	}
	const m = `" k="`
	if q+len(m) > len(b) || string(b[q:q+len(m)]) != m {
		return 0, "", "", false
	}
	ref := string(b[rs:q])
	q += len(m)
	ks := q
	for q < len(b) && refByteOK(b[q]) {
		q++
	}
	const z = `">`
	if q+len(z) > len(b) || string(b[q:q+len(z)]) != z {
		return 0, "", "", false
	}
	return q + len(z), ref, string(b[ks:q]), true
}

func scanDocument(b []byte) *DocScan {
	ds := &DocScan{}
	if off := firstInvalidUTF8(b); off >= 0 {
		ds.Malformed = newObj().set("reason", jStr("invalid_utf8")).set("at", jInt(off)).val()
		return ds
	}
	// 2.2 fences
	lines := splitLines(b)
	var delim []int // line indices (0-based)
	for i, ln := range lines {
		if isFenceLine(b[ln.start:ln.end]) {
			delim = append(delim, i)
		}
	}
	if len(delim)%2 == 1 {
		arr := jArr()
		for _, d := range delim {
			arr.Arr = append(arr.Arr, jInt(d+1))
		}
		ds.Malformed = newObj().set("reason", jStr("unbalanced_fences")).set("delimiter_lines", arr).val()
		return ds
	}
	type region struct{ s, e int }
	var regions []region
	for i := 0; i < len(delim); i += 2 {
		regions = append(regions, region{lines[delim[i]].start, lines[delim[i+1]].end})
	}
	ri := 0
	// 2.3 scan
	p, c := 0, 0
	var stack []*Decl
	for p < len(b) {
		for ri < len(regions) && regions[ri].e <= p {
			ri++
		}
		if ri < len(regions) && p >= regions[ri].s && p < regions[ri].e {
			p = regions[ri].e
			continue
		}
		if p+4 <= len(b) && string(b[p:p+4]) == "<!--" && p >= c {
			if j := indexFrom(b, p+4, "-->"); j >= 0 {
				c = j + 3
			} else {
				c = len(b)
			}
			p += 4
			continue
		}
		if b[p] == '`' {
			p, _, _, _ = inlineCode(b, p)
			continue
		}
		if b[p] == '<' && isCandidate(b, p) {
			if end, ref, kind, ok := matchOpener(b, p); ok {
				r, k := ref, kind
				d := &Decl{At: p, Receipt: &r, Kind: &k, OpenEnd: end}
				if len(stack) > 0 {
					for _, o := range stack {
						if o.Mark == "" || (alt["G9"] && o.Mark == "hidden_commitment") {
							o.Mark = "nesting"
						}
					}
					d.Mark = "nesting"
				} else if p < c {
					d.Mark = "hidden_commitment"
				}
				stack = append(stack, d)
				ds.Decls = append(ds.Decls, d)
				p = end
				continue
			}
			if p+8 <= len(b) && string(b[p:p+8]) == "</sworn>" {
				if len(stack) > 0 {
					d := stack[len(stack)-1]
					stack = stack[:len(stack)-1]
					d.Paired = true
					d.Inner = b[d.OpenEnd:p]
					d.CloserAt = p
					if len(d.Inner) == 0 && (d.Mark == "" || d.Mark == "hidden_commitment") {
						d.Mark = "empty_span"
					}
				} else {
					raw := "</sworn>"
					ds.Decls = append(ds.Decls, &Decl{At: p, Mark: "stray_closer", Raw: &raw})
				}
				p += 8
				continue
			}
			e := len(b)
			for q := p; q < len(b); q++ {
				if b[q] == '>' {
					e = q + 1
					break
				}
				if b[q] == '\n' {
					e = q
					break
				}
			}
			raw := string(b[p:e])
			ds.Decls = append(ds.Decls, &Decl{At: p, Mark: "tag_syntax", Raw: &raw})
			p = e
			continue
		}
		p++
	}
	// 2.4
	for _, o := range stack {
		if o.Mark == "" || o.Mark == "hidden_commitment" {
			o.Mark = "unclosed"
		}
	}
	// 2.5 canonical text
	blocking := false
	for _, d := range ds.Decls {
		switch d.Mark {
		case "tag_syntax", "nesting", "stray_closer", "unclosed":
			blocking = true
		}
	}
	if !blocking {
		var out []byte
		last := 0
		for _, d := range ds.Decls { // all paired, none nested, in order of at
			out = append(out, b[last:d.At]...)
			d.Start = len(out)
			out = append(out, d.Inner...)
			d.End = len(out)
			last = d.CloserAt + 8
		}
		out = append(out, b[last:]...)
		if out == nil {
			out = []byte{}
		}
		ds.Canonical = out
		ds.HasCanon = true
	}
	return ds
}

func isCandidate(b []byte, p int) bool {
	q := p + 1
	if q < len(b) && b[q] == '/' {
		q++
	}
	const w = "sworn"
	if q+len(w) > len(b) {
		return false
	}
	for i := 0; i < len(w); i++ {
		if lower(b[q+i]) != w[i] {
			return false
		}
	}
	q += len(w)
	return q >= len(b) || !isWordByte(b[q])
}
