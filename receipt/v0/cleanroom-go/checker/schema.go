package main

// A small evaluator for the JSON Schema keywords the given schema lists:
// type, const, enum, pattern, minimum, maxLength, maxItems, required,
// properties, additionalProperties, items, anyOf, allOf, if, then, else, $ref.
// Numbers are read as written (11.3): "integer" means a number token with no
// fraction and no exponent.

import (
	_ "embed"
	"math/big"
	"regexp"
	"strings"
	"sync"
)

//go:embed receipt-v0.schema.json
var schemaBytes []byte

var (
	schemaRoot *JVal
	schemaOnce sync.Once
	reCache    = map[string]*regexp.Regexp{}
)

func loadSchema() *JVal {
	schemaOnce.Do(func() {
		v, _, err := parseJSON(schemaBytes)
		if err != nil {
			panic("embedded schema does not parse")
		}
		schemaRoot = v
		// GUESS G-R1: attestations_not_empty (11.3, second step) must be
		// reachable, so the shape step does not apply maxItems to attestations.
		rc := v.Map["$defs"].Map["receipt"].Map["properties"].Map["attestations"]
		if alt["G16"] {
			return
		}
		delete(rc.Map, "maxItems")
		var keys []string
		for _, k := range rc.Keys {
			if k != "maxItems" {
				keys = append(keys, k)
			}
		}
		rc.Keys = keys
	})
	return schemaRoot
}

func resolveRef(ref string) *JVal {
	root := loadSchema()
	if !strings.HasPrefix(ref, "#/") {
		panic("unsupported $ref " + ref)
	}
	cur := root
	for _, t := range strings.Split(ref[2:], "/") {
		cur = cur.Map[t]
	}
	return cur
}

func isIntegerWritten(v *JVal) bool {
	return v.K == JNum && !strings.ContainsAny(v.Num, ".eE")
}

func typeMatches(v *JVal, t string) bool {
	switch t {
	case "object":
		return v.K == JObj
	case "array":
		return v.K == JArr
	case "string":
		return v.K == JStr
	case "integer":
		return isIntegerWritten(v)
	case "number":
		return v.K == JNum
	case "boolean":
		return v.K == JBool
	case "null":
		return v.K == JNull
	}
	return false
}

func jEqual(a, b *JVal) bool {
	if a.K != b.K {
		return false
	}
	switch a.K {
	case JNull:
		return true
	case JBool:
		return a.B == b.B
	case JNum:
		if alt["G19"] {
			x, y := parseJSONNumber(a.Num), parseJSONNumber(b.Num)
			xs, _ := x.roundPlain(400)
			ys, _ := y.roundPlain(400)
			return xs == ys
		}
		if !isIntegerWritten(a) || !isIntegerWritten(b) {
			return false
		}
		x, _ := new(big.Int).SetString(a.Num, 10)
		y, _ := new(big.Int).SetString(b.Num, 10)
		return x.Cmp(y) == 0
	case JSpecial:
		return false
	case JStr:
		return a.Str == b.Str
	case JArr:
		if len(a.Arr) != len(b.Arr) {
			return false
		}
		for i := range a.Arr {
			if !jEqual(a.Arr[i], b.Arr[i]) {
				return false
			}
		}
		return true
	case JObj:
		if len(a.Keys) != len(b.Keys) {
			return false
		}
		for _, k := range a.Keys {
			y, ok := b.Map[k]
			if !ok || !jEqual(a.Map[k], y) {
				return false
			}
		}
		return true
	}
	return false
}

func validate(v, s *JVal) bool {
	if s.K == JBool {
		return s.B
	}
	if s.K != JObj {
		return true
	}
	if r, ok := s.Map["$ref"]; ok {
		if !validate(v, resolveRef(r.Str)) {
			return false
		}
	}
	if t, ok := s.Map["type"]; ok {
		okT := false
		if t.K == JStr {
			okT = typeMatches(v, t.Str)
		} else {
			for _, x := range t.Arr {
				if typeMatches(v, x.Str) {
					okT = true
				}
			}
		}
		if !okT {
			return false
		}
	}
	if c, ok := s.Map["const"]; ok && !jEqual(v, c) {
		return false
	}
	if e, ok := s.Map["enum"]; ok {
		found := false
		for _, x := range e.Arr {
			if jEqual(v, x) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	if v.K == JStr {
		if p, ok := s.Map["pattern"]; ok {
			re, ok := reCache[p.Str]
			if !ok {
				re = regexp.MustCompile(p.Str)
				reCache[p.Str] = re
			}
			if hasLoneSurrogate(v.Str) {
				// replace surrogates by U+FFFD: none of the schema's patterns admit it
				if !re.MatchString(string(wtf8Runes(v.Str))) {
					return false
				}
			} else if !re.MatchString(v.Str) {
				return false
			}
		}
		if ml, ok := s.Map["maxLength"]; ok {
			n, _ := new(big.Int).SetString(ml.Num, 10)
			if big.NewInt(int64(len(wtf8Runes(v.Str)))).Cmp(n) > 0 {
				return false
			}
		}
	}
	if v.K == JNum {
		if mn, ok := s.Map["minimum"]; ok {
			if isIntegerWritten(v) {
				x, _ := new(big.Int).SetString(v.Num, 10)
				y, _ := new(big.Int).SetString(mn.Num, 10)
				if x.Cmp(y) < 0 {
					return false
				}
			} else if strings.HasPrefix(v.Num, "-") && parseJSONNumber(v.Num).Coef != "0" {
				return false
			}
		}
	}
	if v.K == JArr {
		if mi, ok := s.Map["maxItems"]; ok {
			n, _ := new(big.Int).SetString(mi.Num, 10)
			if big.NewInt(int64(len(v.Arr))).Cmp(n) > 0 {
				return false
			}
		}
		if it, ok := s.Map["items"]; ok {
			for _, x := range v.Arr {
				if !validate(x, it) {
					return false
				}
			}
		}
	}
	if v.K == JObj {
		if rq, ok := s.Map["required"]; ok {
			for _, k := range rq.Arr {
				if _, ok := v.Map[k.Str]; !ok {
					return false
				}
			}
		}
		props, hasProps := s.Map["properties"]
		if hasProps {
			for _, k := range props.Keys {
				if x, ok := v.Map[k]; ok && !validate(x, props.Map[k]) {
					return false
				}
			}
		}
		if ap, ok := s.Map["additionalProperties"]; ok {
			for _, k := range v.Keys {
				if hasProps {
					if _, ok := props.Map[k]; ok {
						continue
					}
				}
				if !validate(v.Map[k], ap) {
					return false
				}
			}
		}
	}
	if a, ok := s.Map["allOf"]; ok {
		for _, x := range a.Arr {
			if !validate(v, x) {
				return false
			}
		}
	}
	if a, ok := s.Map["anyOf"]; ok {
		any := false
		for _, x := range a.Arr {
			if validate(v, x) {
				any = true
				break
			}
		}
		if !any {
			return false
		}
	}
	if c, ok := s.Map["if"]; ok {
		if validate(v, c) {
			if t, ok := s.Map["then"]; ok && !validate(v, t) {
				return false
			}
		} else if e, ok := s.Map["else"]; ok && !validate(v, e) {
			return false
		}
	}
	return true
}

func validateReceipt(v *JVal) bool { return validate(v, loadSchema()) }
