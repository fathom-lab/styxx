package main

// 7.3: the receipt number (coefficient, exponent, adjusted exponent), its
// scientific string, and rounding to f fractional digits, half to even,
// in exact decimal arithmetic. Exponents are big integers: a written
// exponent may have thousands of digits.

import (
	"math/big"
	"strings"
)

type Dec struct {
	Neg  bool
	Coef string   // digits, no leading zeros ("0" if zero)
	Exp  *big.Int // exponent
	Lead int      // G21-alt only: leading zeros written in the printed token
}

var bigOne = big.NewInt(1)

// parseJSONNumber reads a JSON number token (already validated by the parser).
func parseJSONNumber(t string) Dec {
	d := Dec{}
	if strings.HasPrefix(t, "-") {
		d.Neg = true
		t = t[1:]
	}
	expPart := ""
	if i := strings.IndexAny(t, "eE"); i >= 0 {
		expPart = t[i+1:]
		t = t[:i]
	}
	intPart, frac := t, ""
	if i := strings.IndexByte(t, '.'); i >= 0 {
		intPart, frac = t[:i], t[i+1:]
	}
	c := strings.TrimLeft(intPart+frac, "0")
	if c == "" {
		c = "0"
	}
	d.Coef = c
	e := new(big.Int)
	if expPart != "" {
		e.SetString(strings.TrimPrefix(expPart, "+"), 10)
	}
	e.Sub(e, big.NewInt(int64(len(frac))))
	d.Exp = e
	return d
}

func (d Dec) Adjusted() *big.Int {
	a := new(big.Int).Add(d.Exp, big.NewInt(int64(len(d.Coef))))
	return a.Sub(a, bigOne)
}

func (d Dec) IsZero() bool { return d.Coef == "0" }

// SciString: the General Decimal Arithmetic to-scientific-string of (coef, exp).
func (d Dec) SciString() string {
	var sb strings.Builder
	if d.Neg {
		sb.WriteByte('-')
	}
	adj := d.Adjusted()
	if d.Exp.Sign() <= 0 && adj.Cmp(big.NewInt(-6)) >= 0 {
		n := int(-d.Exp.Int64()) // bounded: adj >= -6 and exp <= 0
		c := d.Coef
		if n == 0 {
			sb.WriteString(c)
		} else if len(c) > n {
			sb.WriteString(c[:len(c)-n])
			sb.WriteByte('.')
			sb.WriteString(c[len(c)-n:])
		} else {
			sb.WriteString("0.")
			sb.WriteString(strings.Repeat("0", n-len(c)))
			sb.WriteString(c)
		}
		return sb.String()
	}
	sb.WriteByte(d.Coef[0])
	if len(d.Coef) > 1 {
		sb.WriteByte('.')
		sb.WriteString(d.Coef[1:])
	}
	sb.WriteByte('E')
	if adj.Sign() >= 0 {
		sb.WriteByte('+')
		sb.WriteString(adj.String())
	} else {
		sb.WriteByte('-')
		sb.WriteString(new(big.Int).Neg(adj).String())
	}
	return sb.String()
}

func pow10(n int64) *big.Int { return new(big.Int).Exp(big.NewInt(10), big.NewInt(n), nil) }

// roundPlain rounds d to f fractional digits, ROUND_HALF_EVEN, and writes it in
// plain notation with exactly f fractional digits and an unsigned zero.
// It also reports whether the rounded value is zero.
// The caller guarantees the adjusted exponent is at most 320.
func (d Dec) roundPlain(f int) (string, bool) {
	coef, _ := new(big.Int).SetString(d.Coef, 10)
	var q *big.Int
	adj := d.Adjusted()
	// |v| < 10^(adj+1); if adj <= -f-2 then |v|*10^f < 0.1, rounds to 0
	if adj.Cmp(big.NewInt(int64(-f-2))) <= 0 {
		q = new(big.Int)
	} else {
		e2 := new(big.Int).Add(d.Exp, big.NewInt(int64(f))) // bounded here
		if e2.Sign() >= 0 {
			q = new(big.Int).Mul(coef, pow10(e2.Int64()))
		} else {
			k := -e2.Int64()
			div := pow10(k)
			qq, r := new(big.Int).QuoRem(coef, div, new(big.Int))
			r2 := new(big.Int).Lsh(r, 1)
			switch c := r2.Cmp(div); {
			case c > 0:
				qq.Add(qq, bigOne)
			case c == 0:
				if qq.Bit(0) == 1 || alt["C1"] {
					qq.Add(qq, bigOne)
				}
			}
			q = qq
		}
	}
	zero := q.Sign() == 0
	digits := q.String()
	if len(digits) < f+1 {
		digits = strings.Repeat("0", f+1-len(digits)) + digits
	}
	out := digits
	if f > 0 {
		out = digits[:len(digits)-f] + "." + digits[len(digits)-f:]
	}
	if d.Lead > 0 {
		out = strings.Repeat("0", d.Lead) + out
	}
	if d.Neg && !zero {
		out = "-" + out
	}
	return out, zero
}

func bigFromInt(n int) *big.Int { return big.NewInt(int64(n)) }
