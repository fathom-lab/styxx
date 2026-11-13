package main

import (
	"fmt"
	"os"
)

// printMeta prints the vector file's top-level descriptive strings and the
// distinct op values and row widths: the format, not any expected answer.
func printMeta(path string) {
	b, _ := os.ReadFile(path)
	v, _, _ := parseJSON(b)
	for _, k := range []string{"schema", "status", "text", "id_rule"} {
		fmt.Printf("%s: %s\n", k, v.Map[k].Str)
	}
	er := v.Map["expect_rule"]
	for _, k := range er.Keys {
		fmt.Printf("expect_rule.%s: %s\n", k, er.Map[k].Str)
	}
	fmt.Printf("built_from.note: %s\n", v.Map["built_from"].Map["note"].Str)
	ops := map[string]int{}
	widths := map[int]int{}
	srcs := map[string]int{}
	for _, x := range v.Map["vectors"].Arr {
		ops[x.Map["op"].Str]++
		srcs[x.Map["source"].Str]++
		if er, ok := x.Map["expect_rows"]; ok {
			for _, r := range er.Arr {
				widths[len(r.Arr)]++
			}
		}
	}
	fmt.Println("ops:", ops)
	fmt.Println("sources:", srcs)
	fmt.Println("expect_rows widths:", widths)
}
