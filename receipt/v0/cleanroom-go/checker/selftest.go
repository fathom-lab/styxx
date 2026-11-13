package main

// Checks drawn only from examples written in RECEIPT_v0.md itself.

import (
	"fmt"
	"strings"
)

func runSelfTest() int {
	fails := 0
	eq := func(name, got, want string) {
		if got != want {
			fails++
			fmt.Printf("FAIL %s: got %q want %q\n", name, got, want)
		} else {
			fmt.Printf("ok   %s\n", name)
		}
	}
	// 2.1 empty document
	core, ref := checkDocument(Inputs{Doc: []byte{}, HaveDoc: true, Name: jStr("x")})
	if ref != nil {
		eq("2.1 empty", ref.Code, "no refusal")
	} else {
		eq("2.1 empty canonical", core.Map["document"].Map["canonical_sha256"].Str, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
		eq("2.1 empty verdict", core.Map["document_verdict"].Str, "UNSWORN")
	}
	// 2.5 offsets
	ds := scanDocument([]byte("é<sworn r=\"r1\" k=\"numeric\">n 7</sworn>"))
	if len(ds.Decls) == 1 {
		d := ds.Decls[0]
		eq("2.5 at/start/end", fmt.Sprint(d.At, d.Start, d.End), "2 2 5")
	} else {
		eq("2.5 decls", fmt.Sprint(len(ds.Decls)), "1")
	}
	// 3.2 repeated key
	v, _, _ := parseJSON([]byte(`{"a": 1, "a": 2}`))
	c, _ := canonical(v)
	eq("3.2 last wins", c, `{"a":2}`)
	for _, tc := range [][2]string{{"9007199254740993", "int_range"}, {"1.0", "float"}} {
		v, _, _ := parseJSON([]byte(tc[0]))
		_, err := canonical(v)
		got := "none"
		if err != nil {
			got = err.Error()
		}
		eq("3.1 "+tc[0], got, tc[1])
	}
	// 6.10 base64
	for _, tc := range [][2]string{{"QQ==", "A"}, {"QR==", "A"}, {"", ""}} {
		b, ok := strictB64(tc[0])
		eq("6.10 "+tc[0], fmt.Sprint(ok, string(b)), fmt.Sprint(true, tc[1]))
	}
	for _, s := range []string{"QQ", "QQ=", "Q Q==", "QQ==QQ==", "QQ==\n"} {
		_, ok := strictB64(s)
		eq("6.10 bad "+strings.ReplaceAll(s, "\n", "\\n"), fmt.Sprint(ok), "false")
	}
	// 7.3 receipt numbers
	for _, tc := range [][4]string{
		{"0.4211", "4211", "-4", "-1"}, {"1.50", "150", "-2", "0"}, {"100e1", "100", "1", "3"},
		{"0.001e322", "1", "319", "319"}, {"0.000e323", "0", "320", "320"}, {"0.0001e324", "1", "320", "320"},
		{"1e321", "1", "321", "321"}, {"0e400", "0", "400", "400"},
	} {
		d := parseJSONNumber(tc[0])
		eq("7.3 "+tc[0], fmt.Sprint(d.Coef, " ", d.Exp, " ", d.Adjusted()), tc[1]+" "+tc[2]+" "+tc[3])
	}
	for _, tc := range [][2]string{{"0.4211", "0.4211"}, {"1.50", "1.50"}, {"1e-7", "1E-7"}, {"100e1", "1.00E+3"},
		{"0.0001e322", "1E+318"}, {"0.000e323", "0E+320"}, {"-0", "-0"}, {"1e321", "1E+321"}, {"0e400", "0E+400"}} {
		eq("7.3 sci "+tc[0], parseJSONNumber(tc[0]).SciString(), tc[1])
	}
	for _, tc := range [][2]string{{"42.", "42"}, {"+42", "+42"}, {"42%", "42%"}, {"−3", "−3"}} {
		t, reason, _ := numericForm("n " + tc[0])
		eq("7.3 printed_token "+tc[0], t+reason, tc[1])
	}
	for _, tc := range [][2]string{{".5", "0.5"}, {"+42", "42"}, {"1,234.5", "1234.5"}, {"−3", "-3"}} {
		d, f := printedDec(tc[0])
		p, _ := d.roundPlain(f)
		eq("7.3 printed "+tc[0], p, tc[1])
	}
	for _, s := range []string{"r1", "v0.1", "1e-5", "3/4", "2026-09-01", "٧"} {
		_, reason, _ := numericForm(s)
		eq("7.2 malformed "+s, fmt.Sprint(reason != ""), "true")
	}
	_, reason, _ := numericForm("n is 7 \U00011BF0")
	eq("7.2 U+11BF0", reason, "")
	_, reason, _ = numericForm("7Ᲊ")
	eq("7.2 U+1C89", reason, "")
	_, reason, _ = numericForm("7ᲀ")
	eq("7.2 U+1C80", reason, "number_grammar")
	// 7.3 comparisons from the text
	cmp := func(printed, receipt string) string {
		pd, f := printedDec(printed)
		ps, _ := pd.roundPlain(f)
		rd := parseJSONNumber(receipt)
		rs, z := rd.roundPlain(f)
		if ps != rs {
			return "FAILED"
		}
		if !rd.IsZero() && z {
			return "rounds_to_zero"
		}
		return "HELD"
	}
	eq("7.3 0.001e322 vs 0", cmp("0", "0.001e322"), "FAILED")
	eq("7.3 0.000e323 vs 0", cmp("0", "0.000e323"), "HELD")
	eq("7.3 0.0001e324 vs 1", cmp("1", "0.0001e324"), "FAILED")
	eq("7.3 42% vs 42", cmp("42", "42"), "HELD")
	eq("7.3a 0 vs 0.001", cmp("0", "0.001"), "rounds_to_zero")
	eq("7.3 huge negative exponent", cmp("0", "1e-"+strings.Repeat("9", 4001)), "rounds_to_zero")
	// 7.3 leaf
	lv, _, _ := parseJSON([]byte(`"\ud800x"`))
	eq("7.3 leaf surrogate", leafText(lv), "?x")
	// 7.6
	h64 := strings.Repeat("ab", 32)
	l, _ := hexRuns([]byte("sha256 " + h64))
	eq("7.6 a", fmt.Sprint(l), "[64]")
	l, _ = hexRuns([]byte("sha256 " + h64 + " and " + h64[:32]))
	eq("7.6 b", fmt.Sprint(l), "[64 32]")
	l, _ = hexRuns([]byte("g" + h64[:32]))
	eq("7.6 c", fmt.Sprint(l), "[]")
	// 5.4 authored rebuild example
	a, b := strings.Repeat("a", 64), strings.Repeat("b", 64)
	m, rf := readManifest([]byte(`{"spec":"sworn/manifest/0.2","harness":"h","turn":"t","minted_at":"m","authored_sha256":["` + strings.ToUpper(b) + `","` + a + `","` + a + `"],"receipts":{},"rung":null}`))
	if rf != nil {
		eq("5.4", rf.Code, "no refusal")
	} else {
		eq("5.4 authored", strings.Join(m.Authored, ","), a+","+a+","+b)
	}
	// 5.5 examples
	for _, tc := range [][2]string{
		{`{"spec":"sworn/manifest/0.2","spec":"sworn/manifest/0.2","harness":"h","turn":"t","minted_at":"m"}`, "manifest_form"},
		{`{"spec":"sworn/manifest/0.1","harness":"h","harness":"h","turn":"t","minted_at":"m"}`, "manifest_form"},
		{string([]byte{0xEF, 0xBB, 0xBF}) + "{}", "manifest_spec"},
		{`{"spec":"sworn/manifest/0.2","harness":"h","turn":"t","minted_at":"m"} x`, "manifest_spec"},
		{`{"spec":"sworn/manifest/0.2","harness":"h","turn":"t"}`, "manifest_form"},
	} {
		_, rf := readManifest([]byte(tc[0]))
		got := "none"
		if rf != nil {
			got = rf.Code
		}
		eq("5.5 "+tc[1], got, tc[1])
	}
	// 5.6 digest null is intact
	m, _ = readManifest([]byte(`{"spec":"sworn/manifest/0.2","harness":"h","turn":"t","minted_at":"m","digest":null}`))
	eq("5.6 digest null intact", fmt.Sprint(m.Intact), "true")
	// 9.1 rungs example: one tag_syntax row, no manifest
	core, _ = checkDocument(Inputs{Doc: []byte("<sworn>"), HaveDoc: true, Name: jStr("d")})
	ct, _ := canonical(core.Map["rungs"])
	eq("9.1 rungs tag_syntax", ct, `{"unresolved":1}`)
	eq("13.12 malformed tag verdict", core.Map["document_verdict"].Str, "SWORN-FAILED")
	// end-to-end HELD with an L2 manifest
	rb := []byte(`{"p95": 240}`)
	sha := shaHex(rb)
	man := `{"spec":"sworn/manifest/0.2","harness":"h","turn":"t","minted_at":"m","authored_sha256":[],"rung":"L2","receipts":{"r1":{"id":"r2","sha256":"` + sha + `","kind_of_source":"tool_stdout","captured_at":"c","complete":true,"bytes":"` + b64enc(rb) + `"}}}`
	core, ref = checkDocument(Inputs{Doc: []byte(`Latency fell to <sworn r="r1#/p95" k="numeric">240</sworn> ms. <sworn>`), HaveDoc: true, Name: jStr("d"), Manifest: []byte(man), HaveMan: true})
	if ref != nil {
		eq("e2e", ref.Code, "none")
	} else {
		ct, _ := canonical(core.Map["rungs"])
		eq("9.1 rungs example", ct, `{"L2":1,"unresolved":1}`)
		eq("e2e row0", core.Map["spans"].Arr[0].Map["verdict"].Str, "HELD")
	}
	// receipt round trip
	if core != nil {
		rt, _ := receiptOf(core)
		ans, rf := checkReceipt([]byte(rt), []byte(`Latency fell to <sworn r="r1#/p95" k="numeric">240</sworn> ms. <sworn>`), true, []byte(man), true)
		got := ans
		if rf != nil {
			got = rf.Code
		}
		eq("11.3 round trip", got, "REDERIVES")
	}
	fmt.Printf("selftest: %d failures\n", fails)
	if fails > 0 {
		return 1
	}
	return 0
}

func b64enc(b []byte) string {
	const al = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
	var sb strings.Builder
	for i := 0; i < len(b); i += 3 {
		var n uint32
		k := len(b) - i
		if k > 3 {
			k = 3
		}
		for j := 0; j < 3; j++ {
			n <<= 8
			if j < k {
				n |= uint32(b[i+j])
			}
		}
		for j := 0; j < 4; j++ {
			if j <= k {
				sb.WriteByte(al[(n>>(18-6*uint(j)))&63])
			} else {
				sb.WriteByte('=')
			}
		}
	}
	return sb.String()
}
