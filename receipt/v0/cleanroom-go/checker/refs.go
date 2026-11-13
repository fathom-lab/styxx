package main

// Section 4: REF.

import (
	"math"
	"math/big"
	"strings"
)

type Ref struct {
	Form    string // "rn", "path", "prereg"
	ID      string // for rn: the head, e.g. "r1"
	HasFrag bool
	IsLine  bool
	LA, LB  int
	Pointer []string // unescaped tokens
}

// posDecimal reads a positive decimal without a leading zero. Values beyond
// int range are clamped (they are out of range of any receipt anyway); the
// big value is returned for ordering.
func posDecimal(s string) (int, *big.Int, bool) {
	if !isAllDigitsNoLeadingZero(s) {
		return 0, nil, false
	}
	bv, _ := new(big.Int).SetString(s, 10)
	if alt["G25"] && (!bv.IsInt64() || bv.Int64() > math.MaxInt32) {
		return 0, nil, false
	}
	if !bv.IsInt64() || bv.Int64() > math.MaxInt32 {
		return math.MaxInt32, bv, true
	}
	return int(bv.Int64()), bv, true
}

func parseFragment(f string, r *Ref) bool {
	if strings.HasPrefix(f, "/") {
		for i := 0; i < len(f); i++ {
			if f[i] == '~' && (i+1 >= len(f) || (f[i+1] != '0' && f[i+1] != '1')) {
				return false
			}
		}
		for _, t := range strings.Split(f[1:], "/") {
			t = strings.ReplaceAll(t, "~1", "/")
			t = strings.ReplaceAll(t, "~0", "~")
			r.Pointer = append(r.Pointer, t)
		}
		return true
	}
	if !strings.HasPrefix(f, "L") {
		return false
	}
	body := f[1:]
	if i := strings.IndexByte(body, '-'); i >= 0 {
		a, ba, ok1 := posDecimal(body[:i])
		rest := body[i+1:]
		if !strings.HasPrefix(rest, "L") {
			return false
		}
		b, bb, ok2 := posDecimal(rest[1:])
		if !ok1 || !ok2 || bb.Cmp(ba) < 0 {
			return false
		}
		r.IsLine, r.LA, r.LB = true, a, b
		return true
	}
	a, _, ok := posDecimal(body)
	if !ok {
		return false
	}
	r.IsLine, r.LA, r.LB = true, a, a
	return true
}

var pathBad = map[rune]bool{'\\': true, '*': true, '?': true, '[': true, ']': true,
	0x7F: true, 0x85: true, 0xA0: true, 0x1680: true, 0x2028: true, 0x2029: true,
	0x202F: true, 0x205F: true, 0x3000: true}

func pathTargetOK(t string) bool {
	if t == "" || t[0] == '/' || t[0] == ':' {
		return false
	}
	for _, seg := range strings.Split(t, "/") {
		if seg == "" || seg == "." || seg == ".." {
			return false
		}
	}
	for _, c := range t {
		if c <= 0x20 || pathBad[c] || (c >= 0x2000 && c <= 0x200A) {
			return false
		}
	}
	return true
}

func isHexStr(s string) bool {
	for i := 0; i < len(s); i++ {
		if hexv(s[i]) < 0 {
			return false
		}
	}
	return true
}

// parseRef returns nil when REF fails section 4 (receipt_form).
func parseRef(ref string) *Ref {
	head, frag, has := strings.Cut(ref, "#")
	r := &Ref{}
	if has {
		if frag == "" {
			return nil
		}
		r.HasFrag = true
		if !parseFragment(frag, r) {
			return nil
		}
	}
	switch {
	case strings.HasPrefix(head, "r") && isAllDigitsNoLeadingZero(head[1:]):
		r.Form, r.ID = "rn", head
	case strings.HasPrefix(head, "path:") && pathTargetOK(head[5:]):
		r.Form = "path"
	case strings.HasPrefix(head, "prereg:") && len(head) == 7+64 && isHexStr(head[7:]):
		r.Form = "prereg"
	default:
		return nil
	}
	return r
}

func isAllDigitsNoLeadingZero(s string) bool {
	if s == "" || s[0] == '0' {
		return false
	}
	for i := 0; i < len(s); i++ {
		if !isDig(s[i]) {
			return false
		}
	}
	return true
}
