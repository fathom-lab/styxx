package main

// Section 3: the canonical form. JCS restricted to null, booleans, integers
// with |n| <= 2^53-1 written without fraction or exponent, strings, arrays,
// objects with keys in U+0000..U+007E, depth <= 32.

import (
	"math/big"
	"sort"
	"strings"
)

type CanonError struct{ Code string }

func (e *CanonError) Error() string { return e.Code }

var maxSafe = new(big.Int).SetInt64(9007199254740991)

func canonical(v *JVal) (string, error) {
	var sb strings.Builder
	if err := canonWrite(&sb, v, 1); err != nil {
		return "", err
	}
	return sb.String(), nil
}

func canonInt(num string) (string, *CanonError) {
	if strings.ContainsAny(num, ".eE") {
		return "", &CanonError{"float"}
	}
	n, ok := new(big.Int).SetString(num, 10)
	if !ok {
		return "", &CanonError{"float"}
	}
	if new(big.Int).Abs(n).Cmp(maxSafe) > 0 {
		return "", &CanonError{"int_range"}
	}
	if alt["G4"] && strings.HasPrefix(num, "-") && n.Sign() == 0 {
		return "-0", nil
	}
	return n.String(), nil // -0 prints as 0
}

func canonWrite(sb *strings.Builder, v *JVal, depth int) error {
	if alt["G29"] && depth > 32 {
		return &CanonError{"depth"}
	}
	switch v.K {
	case JNull:
		sb.WriteString("null")
	case JBool:
		if v.B {
			sb.WriteString("true")
		} else {
			sb.WriteString("false")
		}
	case JSpecial:
		return &CanonError{"float"}
	case JNum:
		s, err := canonInt(v.Num)
		if err != nil {
			return err
		}
		sb.WriteString(s)
	case JStr:
		if hasLoneSurrogate(v.Str) {
			return &CanonError{"lone_surrogate"}
		}
		writeCanonString(sb, v.Str)
	case JArr:
		if depth > 32 {
			return &CanonError{"depth"}
		}
		sb.WriteByte('[')
		for i, x := range v.Arr {
			if i > 0 {
				sb.WriteByte(',')
			}
			if err := canonWrite(sb, x, depth+1); err != nil {
				return err
			}
		}
		sb.WriteByte(']')
	case JObj:
		if depth > 32 {
			return &CanonError{"depth"}
		}
		vals := make(map[string]string, len(v.Keys))
		if alt["G2"] {
			ks := append([]string(nil), v.Keys...)
			sort.Strings(ks)
			for _, k := range ks {
				if e := keyCheck(k); e != nil {
					return e
				}
				var sub strings.Builder
				if err := canonWrite(&sub, v.Map[k], depth+1); err != nil {
					return err
				}
				vals[k] = sub.String()
			}
		}
		// check keys and values in insertion order first (GUESS G2), then emit sorted
		for _, k := range v.Keys {
			if alt["G2"] {
				break
			}
			if alt["G3"] && hasLoneSurrogate(k) {
				return &CanonError{"lone_surrogate"}
			}
			for i := 0; i < len(k); i++ {
				if k[i] > 0x7E {
					return &CanonError{"non_ascii_key"}
				}
			}
		}
		for _, k := range v.Keys {
			if alt["G2"] {
				break
			}
			var sub strings.Builder
			if err := canonWrite(&sub, v.Map[k], depth+1); err != nil {
				return err
			}
			vals[k] = sub.String()
		}
		keys := append([]string(nil), v.Keys...)
		sort.Strings(keys) // keys are ASCII here, so byte order == code point order
		parts := make([]string, 0, len(keys))
		for _, k := range keys {
			var sub strings.Builder
			writeCanonString(&sub, k)
			sub.WriteByte(':')
			sub.WriteString(vals[k])
			parts = append(parts, sub.String())
		}
		sb.WriteByte('{')
		sb.WriteString(strings.Join(parts, ","))
		sb.WriteByte('}')
	default:
		return &CanonError{"type"}
	}
	return nil
}

func writeCanonString(sb *strings.Builder, s string) {
	const hx = "0123456789abcdef"
	sb.WriteByte('"')
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c == '"':
			sb.WriteString(`\"`)
		case c == '\\':
			sb.WriteString(`\\`)
		case c == 8:
			sb.WriteString(`\b`)
		case c == 9:
			sb.WriteString(`\t`)
		case c == 10:
			sb.WriteString(`\n`)
		case c == 12:
			sb.WriteString(`\f`)
		case c == 13:
			sb.WriteString(`\r`)
		case c < 0x20:
			sb.WriteString(`\u00`)
			sb.WriteByte(hx[c>>4])
			sb.WriteByte(hx[c&15])
		default:
			sb.WriteByte(c)
		}
	}
	sb.WriteByte('"')
}

// ---- building values in Go code (for the verdict core) ----

func jNull() *JVal            { return &JVal{K: JNull} }
func jBool(b bool) *JVal      { return &JVal{K: JBool, B: b} }
func jInt(n int) *JVal        { return &JVal{K: JNum, Num: big.NewInt(int64(n)).String()} }
func jStr(s string) *JVal     { return &JVal{K: JStr, Str: s} }
func jArr(xs ...*JVal) *JVal  { return &JVal{K: JArr, Arr: append([]*JVal{}, xs...)} }
func jStrOrNull(s *string) *JVal {
	if s == nil {
		return jNull()
	}
	return jStr(*s)
}

type objB struct{ v *JVal }

func newObj() *objB {
	return &objB{&JVal{K: JObj, Map: map[string]*JVal{}, Dups: map[string]bool{}}}
}
func (o *objB) set(k string, x *JVal) *objB {
	if _, ok := o.v.Map[k]; !ok {
		o.v.Keys = append(o.v.Keys, k)
	}
	o.v.Map[k] = x
	return o
}
func (o *objB) val() *JVal { return o.v }

func keyCheck(k string) error {
	if alt["G3"] && hasLoneSurrogate(k) {
		return &CanonError{"lone_surrogate"}
	}
	for i := 0; i < len(k); i++ {
		if k[i] > 0x7E {
			return &CanonError{"non_ascii_key"}
		}
	}
	return nil
}
