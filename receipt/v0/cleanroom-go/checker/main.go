package main

import (
	"fmt"
	"os"
	"sort"
	"strings"
)

func main() {
	if err := selfTestUnicode(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: checker selftest | shape <file> | run <vector file>")
		os.Exit(2)
	}
	switch os.Args[1] {
	case "selftest":
		os.Exit(runSelfTest())
	case "shape":
		b, err := os.ReadFile(os.Args[2])
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		v, _, err := parseJSON(b)
		if err != nil {
			fmt.Fprintln(os.Stderr, "vector file does not parse:", err)
			os.Exit(2)
		}
		paths := map[string]int{}
		shapeWalk(v, "$", paths)
		var ks []string
		for k := range paths {
			ks = append(ks, k)
		}
		sort.Strings(ks)
		for _, k := range ks {
			fmt.Printf("%6d  %s\n", paths[k], k)
		}
	case "meta":
		printMeta(os.Args[2])
	case "run":
		os.Exit(runVectors(os.Args[2]))
	default:
		fmt.Fprintln(os.Stderr, "unknown mode")
		os.Exit(2)
	}
}

// shapeWalk records key paths and value kinds only, never values, so the
// runner can be written without reading any expected answer.
func shapeWalk(v *JVal, path string, out map[string]int) {
	kind := []string{"null", "bool", "number", "special", "string", "array", "object"}[v.K]
	out[path+" : "+kind]++
	switch v.K {
	case JArr:
		for _, x := range v.Arr {
			shapeWalk(x, path+"[]", out)
		}
	case JObj:
		for _, k := range v.Keys {
			if strings.Count(path, ".") > 3 {
				continue
			}
			shapeWalk(v.Map[k], path+"."+k, out)
		}
	}
}
