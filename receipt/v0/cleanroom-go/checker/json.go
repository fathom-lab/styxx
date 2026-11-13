package main

// 6.13: JSON read strictly. UTF-8 without a BOM, RFC 8259, plus the bare
// tokens NaN, Infinity, -Infinity. Numbers keep their text. Strings are held
// as WTF-8 (an unpaired surrogate from a \uD800 escape is kept as its 3-byte
// generalised encoding), so "has a lone surrogate" == !utf8.ValidString.
// A repeated key keeps its last value and is remembered as repeated.

import (
	"errors"
	"strings"
	"unicode/utf8"
)

type JKind int

const (
	JNull JKind = iota
	JBool
	JNum
	JSpecial // NaN, Infinity, -Infinity (Num holds the token)
	JStr
	JArr
	JObj
)

type JVal struct {
	K    JKind
	B    bool
	Num  string // number text, or the special token
	Str  string // WTF-8
	Arr  []*JVal
	Keys []string // unique keys, first-insertion order
	Map  map[string]*JVal
	Dups map[string]bool // keys written more than once in this object
}

func (v *JVal) Get(k string) (*JVal, bool) {
	if v == nil || v.K != JObj {
		return nil, false
	}
	x, ok := v.Map[k]
	return x, ok
}

type jparser struct {
	b      []byte
	i      int
	anyDup bool
	depth  int
}

var errJSON = errors.New("not json")

// parseJSON parses b under 6.13. anyDup reports a key written twice in any object.
func parseJSON(b []byte) (v *JVal, anyDup bool, err error) {
	if !utf8.Valid(b) {
		return nil, false, errJSON
	}
	p := &jparser{b: b}
	p.ws()
	v, err = p.value()
	if err != nil {
		return nil, false, err
	}
	p.ws()
	if p.i != len(p.b) {
		return nil, false, errJSON
	}
	return v, p.anyDup, nil
}

func (p *jparser) ws() {
	for p.i < len(p.b) {
		c := p.b[p.i]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			p.i++
		} else {
			return
		}
	}
}

func (p *jparser) lit(s string) bool {
	if p.i+len(s) <= len(p.b) && string(p.b[p.i:p.i+len(s)]) == s {
		p.i += len(s)
		return true
	}
	return false
}

func (p *jparser) value() (*JVal, error) {
	if p.i >= len(p.b) {
		return nil, errJSON
	}
	if alt["G1"] {
		p.depth++
		defer func() { p.depth-- }()
		if p.depth > 1000 {
			return nil, errJSON
		}
	}
	switch c := p.b[p.i]; {
	case c == '{':
		return p.object()
	case c == '[':
		return p.array()
	case c == '"':
		s, err := p.str()
		if err != nil {
			return nil, err
		}
		return &JVal{K: JStr, Str: s}, nil
	case c == 't':
		if p.lit("true") {
			return &JVal{K: JBool, B: true}, nil
		}
	case c == 'f':
		if p.lit("false") {
			return &JVal{K: JBool, B: false}, nil
		}
	case c == 'n':
		if p.lit("null") {
			return &JVal{K: JNull}, nil
		}
	case c == 'N':
		if p.lit("NaN") {
			return &JVal{K: JSpecial, Num: "NaN"}, nil
		}
	case c == 'I':
		if p.lit("Infinity") {
			return &JVal{K: JSpecial, Num: "Infinity"}, nil
		}
	case c == '-' || (c >= '0' && c <= '9'):
		if c == '-' && p.lit("-Infinity") {
			return &JVal{K: JSpecial, Num: "-Infinity"}, nil
		}
		return p.number()
	}
	return nil, errJSON
}

func isDig(c byte) bool { return c >= '0' && c <= '9' }

func (p *jparser) number() (*JVal, error) {
	s := p.i
	if p.b[p.i] == '-' {
		p.i++
	}
	if p.i >= len(p.b) {
		return nil, errJSON
	}
	if p.b[p.i] == '0' {
		p.i++
	} else if p.b[p.i] >= '1' && p.b[p.i] <= '9' {
		for p.i < len(p.b) && isDig(p.b[p.i]) {
			p.i++
		}
	} else {
		return nil, errJSON
	}
	if p.i < len(p.b) && p.b[p.i] == '.' {
		p.i++
		if p.i >= len(p.b) || !isDig(p.b[p.i]) {
			return nil, errJSON
		}
		for p.i < len(p.b) && isDig(p.b[p.i]) {
			p.i++
		}
	}
	if p.i < len(p.b) && (p.b[p.i] == 'e' || p.b[p.i] == 'E') {
		p.i++
		if p.i < len(p.b) && (p.b[p.i] == '+' || p.b[p.i] == '-') {
			p.i++
		}
		if p.i >= len(p.b) || !isDig(p.b[p.i]) {
			return nil, errJSON
		}
		for p.i < len(p.b) && isDig(p.b[p.i]) {
			p.i++
		}
	}
	return &JVal{K: JNum, Num: string(p.b[s:p.i])}, nil
}

func hexv(c byte) int {
	switch {
	case c >= '0' && c <= '9':
		return int(c - '0')
	case c >= 'a' && c <= 'f':
		return int(c-'a') + 10
	case c >= 'A' && c <= 'F':
		return int(c-'A') + 10
	}
	return -1
}

func (p *jparser) hex4() (rune, bool) {
	if p.i+4 > len(p.b) {
		return 0, false
	}
	var r rune
	for k := 0; k < 4; k++ {
		h := hexv(p.b[p.i+k])
		if h < 0 {
			return 0, false
		}
		r = r*16 + rune(h)
	}
	p.i += 4
	return r, true
}

// appendWTF8 encodes r, writing a surrogate code point in its 3-byte form.
func appendWTF8(sb *strings.Builder, r rune) {
	if r >= 0xD800 && r <= 0xDFFF {
		sb.WriteByte(byte(0xE0 | (r >> 12)))
		sb.WriteByte(byte(0x80 | ((r >> 6) & 0x3F)))
		sb.WriteByte(byte(0x80 | (r & 0x3F)))
		return
	}
	sb.WriteRune(r)
}

func (p *jparser) str() (string, error) {
	p.i++ // opening quote
	var sb strings.Builder
	for {
		if p.i >= len(p.b) {
			return "", errJSON
		}
		c := p.b[p.i]
		switch {
		case c == '"':
			p.i++
			return sb.String(), nil
		case c < 0x20:
			return "", errJSON
		case c == '\\':
			p.i++
			if p.i >= len(p.b) {
				return "", errJSON
			}
			e := p.b[p.i]
			p.i++
			switch e {
			case '"':
				sb.WriteByte('"')
			case '\\':
				sb.WriteByte('\\')
			case '/':
				sb.WriteByte('/')
			case 'b':
				sb.WriteByte(8)
			case 'f':
				sb.WriteByte(12)
			case 'n':
				sb.WriteByte(10)
			case 'r':
				sb.WriteByte(13)
			case 't':
				sb.WriteByte(9)
			case 'u':
				r, ok := p.hex4()
				if !ok {
					return "", errJSON
				}
				if r >= 0xD800 && r <= 0xDBFF && p.i+1 < len(p.b) && p.b[p.i] == '\\' && p.b[p.i+1] == 'u' {
					save := p.i
					p.i += 2
					r2, ok2 := p.hex4()
					if ok2 && r2 >= 0xDC00 && r2 <= 0xDFFF {
						appendWTF8(&sb, 0x10000+((r-0xD800)<<10)+(r2-0xDC00))
						continue
					}
					p.i = save
				}
				appendWTF8(&sb, r)
			default:
				return "", errJSON
			}
		default:
			// raw UTF-8 (input validated as a whole)
			_, sz := utf8.DecodeRune(p.b[p.i:])
			sb.Write(p.b[p.i : p.i+sz])
			p.i += sz
		}
	}
}

func (p *jparser) array() (*JVal, error) {
	p.i++
	v := &JVal{K: JArr, Arr: []*JVal{}}
	p.ws()
	if p.i < len(p.b) && p.b[p.i] == ']' {
		p.i++
		return v, nil
	}
	for {
		p.ws()
		x, err := p.value()
		if err != nil {
			return nil, err
		}
		v.Arr = append(v.Arr, x)
		p.ws()
		if p.i >= len(p.b) {
			return nil, errJSON
		}
		if p.b[p.i] == ',' {
			p.i++
			continue
		}
		if p.b[p.i] == ']' {
			p.i++
			return v, nil
		}
		return nil, errJSON
	}
}

func (p *jparser) object() (*JVal, error) {
	p.i++
	v := &JVal{K: JObj, Map: map[string]*JVal{}, Dups: map[string]bool{}}
	rawSeen := map[string]bool{}
	p.ws()
	if p.i < len(p.b) && p.b[p.i] == '}' {
		p.i++
		return v, nil
	}
	for {
		p.ws()
		if p.i >= len(p.b) || p.b[p.i] != '"' {
			return nil, errJSON
		}
		rawStart := p.i
		k, err := p.str()
		rawRepeat := true
		if err == nil && alt["G30"] {
			raw := string(p.b[rawStart:p.i])
			rawRepeat = rawSeen[raw]
			rawSeen[raw] = true
		}
		if err != nil {
			return nil, err
		}
		p.ws()
		if p.i >= len(p.b) || p.b[p.i] != ':' {
			return nil, errJSON
		}
		p.i++
		p.ws()
		x, err := p.value()
		if err != nil {
			return nil, err
		}
		if _, seen := v.Map[k]; seen {
			if rawRepeat {
				v.Dups[k] = true
				p.anyDup = true
			}
		} else {
			v.Keys = append(v.Keys, k)
		}
		v.Map[k] = x
		p.ws()
		if p.i >= len(p.b) {
			return nil, errJSON
		}
		if p.b[p.i] == ',' {
			p.i++
			continue
		}
		if p.b[p.i] == '}' {
			p.i++
			return v, nil
		}
		return nil, errJSON
	}
}

// wtf8Runes decodes a WTF-8 string into code points (surrogates kept).
func wtf8Runes(s string) []rune {
	out := make([]rune, 0, len(s))
	for i := 0; i < len(s); {
		c := s[i]
		if c == 0xED && i+2 < len(s) && s[i+1] >= 0xA0 && s[i+1] <= 0xBF {
			r := rune(c&0x0F)<<12 | rune(s[i+1]&0x3F)<<6 | rune(s[i+2]&0x3F)
			out = append(out, r)
			i += 3
			continue
		}
		r, sz := utf8.DecodeRuneInString(s[i:])
		out = append(out, r)
		i += sz
	}
	return out
}

func hasLoneSurrogate(s string) bool { return !utf8.ValidString(s) }
