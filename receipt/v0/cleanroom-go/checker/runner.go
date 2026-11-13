package main

// The vector runner: per vector, prints the canonical receipt (op check),
// the canonical text (op jcs) or the answer (op recheck), or the refusal
// code; then compares with the vector's expectation and prints a summary.

import (
	"bufio"
	"encoding/base64"
	"fmt"
	"os"
	"sort"
	"strings"
)

func b64in(v *JVal, key string) ([]byte, bool, error) {
	x, ok := v.Map[key]
	if !ok || x.K == JNull {
		return nil, false, nil
	}
	b, err := base64.StdEncoding.DecodeString(x.Str)
	return b, true, err
}

func expStr(e *JVal, k string) (string, bool) {
	x, ok := e.Map[k]
	if !ok || x.K != JStr {
		return "", false
	}
	return x.Str, true
}

func runVectors(path string) int {
	b, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	root, _, err := parseJSON(b)
	if err != nil {
		fmt.Fprintln(os.Stderr, "vector file does not parse")
		return 2
	}
	w := bufio.NewWriterSize(os.Stdout, 1<<20)
	defer w.Flush()
	total, pass := 0, 0
	bySrc := map[string][2]int{}
	failKinds := map[string]int{}
	var failIDs []string
	for _, vec := range root.Map["vectors"].Arr {
		total++
		id := vec.Map["id"].Str
		op := vec.Map["op"].Str
		src := vec.Map["source"].Str
		in := vec.Map["inputs"]
		exp := vec.Map["expect"]
		result := ""
		ok := true
		why := ""
		func() {
			defer func() {
				if r := recover(); r != nil {
					result = fmt.Sprintf("CRASH %v", r)
					ok = false
					why = "crash"
				}
			}()
			switch op {
			case "jcs":
				vb, _, _ := b64in(in, "value_b64")
				v, _, perr := parseJSON(vb)
				if perr != nil {
					result = "REFUSED not_json"
				} else if ct, cerr := canonical(v); cerr != nil {
					result = "REFUSED " + cerr.Error()
				} else {
					result = ct
				}
				if r, has := expStr(exp, "refusal"); has {
					ok = result == "REFUSED "+r
				} else if c, has := expStr(exp, "canonical"); has {
					ok = result == c
				}
				if !ok {
					why = "jcs"
				}
			case "recheck":
				rb, _, _ := b64in(in, "receipt_b64")
				db, hd, _ := b64in(in, "document_b64")
				mb, hm, _ := b64in(in, "manifest_b64")
				ans, rf := checkReceipt(rb, db, hd, mb, hm)
				if rf != nil {
					result = "REFUSED " + rf.Code
				} else {
					result = ans
				}
				if r, has := expStr(exp, "refusal"); has {
					ok = result == "REFUSED "+r
				} else if a, has := expStr(exp, "answer"); has {
					ok = result == a
				}
				if !ok {
					why = "recheck"
				}
			case "check":
				db, hd, _ := b64in(in, "document_b64")
				mb, hm, _ := b64in(in, "manifest_b64")
				name, hasName := in.Map["name"]
				if !hasName {
					name = nil
				}
				core, rf := checkDocument(Inputs{Doc: db, HaveDoc: hd, Name: name, Commit: in.Map["commit"], Manifest: mb, HaveMan: hm})
				if rf != nil {
					result = "REFUSED " + rf.Code
					if r, has := expStr(exp, "refusal"); has {
						ok = r == rf.Code
					} else {
						ok = false
					}
					if !ok {
						why = "check:refusal"
					}
					return
				}
				rt, dg := receiptOf(core)
				result = rt
				if _, has := expStr(exp, "refusal"); has {
					ok, why = false, "check:expected_refusal"
					return
				}
				rowsOK := true
				if er, has := vec.Map["expect_rows"]; has {
					spans := core.Map["spans"].Arr
					if len(spans) != len(er.Arr) {
						rowsOK = false
					} else {
						for i, r := range er.Arr {
							gv := spans[i].Map["verdict"].Str
							gr := spans[i].Map["reason"]
							if r.Arr[0].Str != gv {
								rowsOK = false
							}
							if r.Arr[1].K == JNull {
								if gr.K != JNull {
									rowsOK = false
								}
							} else if gr.K != JStr || gr.Str != r.Arr[1].Str {
								rowsOK = false
							}
						}
					}
					if dv, has := expStr(vec, "expect_document_verdict"); has && dv != core.Map["document_verdict"].Str {
						rowsOK = false
					}
				}
				digOK := true
				if cd, has := expStr(exp, "core_digest"); has {
					digOK = cd == dg
				}
				ok = rowsOK && digOK
				switch {
				case !rowsOK:
					why = "check:rows"
				case !digOK:
					why = "check:digest_only"
				}
			}
		}()
		st := bySrc[src]
		st[1]++
		mark := "FAIL"
		if ok {
			pass++
			st[0]++
			mark = "PASS"
		} else {
			failKinds[why]++
			failIDs = append(failIDs, id)
		}
		bySrc[src] = st
		fmt.Fprintf(w, "%s %s %s %s :: %s\n", mark, id, op, src, result)
		if !ok {
			ej, _ := canonical(exp)
			extra := ""
			if er, has := vec.Map["expect_rows"]; has {
				t, _ := canonical(er)
				extra = " expect_rows=" + t
			}
			fmt.Fprintf(w, "     expected %s%s (%s)\n", ej, extra, why)
		}
	}
	fmt.Fprintf(w, "\nSUMMARY: %d of %d vectors pass\n", pass, total)
	var srcs []string
	for s := range bySrc {
		srcs = append(srcs, s)
	}
	sort.Strings(srcs)
	for _, s := range srcs {
		fmt.Fprintf(w, "  %-12s %d / %d\n", s, bySrc[s][0], bySrc[s][1])
	}
	var ks []string
	for k := range failKinds {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	for _, k := range ks {
		fmt.Fprintf(w, "  fail kind %-24s %d\n", k, failKinds[k])
	}
	_ = strings.Join
	if pass == total {
		return 0
	}
	return 1
}
