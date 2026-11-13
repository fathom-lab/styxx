package main

// Section 7.2 pins Unicode 15.0.0 for the classes L*, N* and Nd.
// Go's unicode package carries one version of the tables (unicode.Version).
// selfTestUnicode recomputes the two range lists in the text's stated form
// and compares them with the digests the text gives, so a host whose tables
// differ is caught before any verdict depends on them.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"unicode"
)

const (
	digestLN = "4ca0e4c9044e908691020e69f4bd19bea0dbb8f67fb7c5488aa1e70f19121bf5"
	digestNd = "ef10fe9b74cef6d6af8debed5957b7c2b2af8909759ddabff75d8c34f95f222c"
)

func isLN(r rune) bool { return unicode.IsLetter(r) || unicode.IsNumber(r) }
func isNd(r rune) bool { return unicode.Is(unicode.Nd, r) }

func rangeText(pred func(rune) bool) (string, int) {
	var parts []string
	in := false
	var first rune
	flush := func(last rune) {
		if first == last {
			parts = append(parts, fmt.Sprintf("%X", first))
		} else {
			parts = append(parts, fmt.Sprintf("%X-%X", first, last))
		}
	}
	for r := rune(0); r <= 0x10FFFF; r++ {
		p := pred(r)
		if p && !in {
			in = true
			first = r
		} else if !p && in {
			in = false
			flush(r - 1)
		}
	}
	if in {
		flush(0x10FFFF)
	}
	return strings.Join(parts, " "), len(parts)
}

func shaHex(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

// selfTestUnicode returns an error when the host tables are not the pinned ones.
func selfTestUnicode() error {
	ln, nln := rangeText(isLN)
	nd, nnd := rangeText(isNd)
	gl, gd := shaHex([]byte(ln)), shaHex([]byte(nd))
	if gl != digestLN || gd != digestNd {
		return fmt.Errorf("unicode tables (go %s) differ from 15.0.0: L/N %d ranges %s, Nd %d ranges %s", unicode.Version, nln, gl, nnd, gd)
	}
	return nil
}
