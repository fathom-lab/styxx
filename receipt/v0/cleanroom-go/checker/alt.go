package main

// Alternate readings, switched on by RV0_ALT=G7,G16,... so that each guess in
// GUESSES.md can be run against the vectors both ways. The default build
// (RV0_ALT unset) is the checker as first run.

import (
	"os"
	"strings"
)

var alt = func() map[string]bool {
	m := map[string]bool{}
	for _, g := range strings.Split(os.Getenv("RV0_ALT"), ",") {
		if g = strings.TrimSpace(g); g != "" {
			m[g] = true
		}
	}
	return m
}()
