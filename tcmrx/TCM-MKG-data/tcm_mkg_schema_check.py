#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCM-MKG TSV Schema Checker
--------------------------
- Scans a directory of TSV files (the 25 files from Zenodo).
- Prints the actual headers for each file.
- Compares against an expected schema derived from the import manual and data notes.
- Flags mismatches and suggests fixes (e.g., D3 CPM_ID vs CHP_ID, Predicted_binding_affinity casing).
- Also checks for filename canonicalization differences (case/underscore variants).

Usage:
  python tcm_mkg_schema_check.py --dir /path/to/TCM-MKG-data  [--save report.json]

Author: ChatGPT
License: MIT
"""
import argparse, os, csv, sys, json, re
from collections import OrderedDict

# Canonical filenames (what import Cypher expects)
CANON = [
 "D1_TCM_terminology.tsv",
 "D2_Chinese_patent_medicine.tsv",
 "D3_CPM_TCMT.tsv",
 "D4_CPM_CHP.tsv",
 "D5_CPM_ICD11.tsv",
 "D6_Chinese_herbal_pieces.tsv",
 "D7_CHP_Medicinal_properties.tsv",
 "D8_CHP_PO.tsv",
 "D9_CHP_InChIKey.tsv",
 "D10_Pharmacognostic_origin.tsv",
 "D11_PO_InChIKey.tsv",
 "D12_InChIKey.tsv",
 "D13_InChIKey_EntrezID.tsv",
 "D14_InChIKey_SourceID.tsv",
 "D15_InChIKey_distance.tsv",
 "D16_Protein_Protein_Interactions.tsv",
 "D17_Target_Symbol_Mapping.tsv",
 "D18_ICD11.tsv",
 "D19_ICD11_CUI.tsv",
 "D20_ICD11_MeSH.tsv",
 "D21_ICD11_DOID.tsv",
 "D22_CUI_targets.tsv",
 "D23_MeSH_targets.tsv",
 "D24_DOID_targets.tsv",
 "SD1_predicted_InChIKey_EntrezID.tsv",
]

# Minimal expected headers for each file (subset check)
EXPECTED = {
 "D1_TCM_terminology.tsv": ["TCMT_ID","Chinese_group","English_group","Chinese_term","English_term"],
 "D2_Chinese_patent_medicine.tsv": ["CPM_ID","Chinese_patent_medicine","Pinyin_term","Routes_of_administration"],
 "D3_CPM_TCMT.tsv": ["CPM_ID","TCMT_ID"],
 "D4_CPM_CHP.tsv": ["CPM_ID","CHP_ID","Dosage_ratio"],
 "D5_CPM_ICD11.tsv": ["CPM_ID","ICD11_code"],
 "D6_Chinese_herbal_pieces.tsv": ["CHP_ID","Chinese_herbal_pieces","Pinyin_term","English_term","Sources"],
 "D7_CHP_Medicinal_properties.tsv": ["CHP_ID","Medicinal_properties","Class","x_rank","y_rank"],
 "D8_CHP_PO.tsv": ["CHP_ID","species_ID","species_name","Sources"],
 "D9_CHP_InChIKey.tsv": ["CHP_ID","InChIKey","Source"],
 "D10_Pharmacognostic_origin.tsv": ["species_ID","species_name","kingdom_Name","phylum_Name","class_Name","order_Name","family_Name","genus_Name"],
 "D11_PO_InChIKey.tsv": ["SpeciesID","InChIKey","Source"],
 "D12_InChIKey.tsv": ["InChIKey","SMILES","InChI","Molecular_formula","QED","MolWt","TPSA","MolLogP","NumHAcceptors","NumHDonors"],
 "D13_InChIKey_EntrezID.tsv": ["InChIKey","EntrezID"],
 "D14_InChIKey_SourceID.tsv": ["InChIKey","Source","SourceID"],
 "D15_InChIKey_distance.tsv": ["InChIKey1","InChIKey2","Distance"],
 # Accept either capitalization for "Protein_Protein_Interactions"
 "D16_Protein_Protein_Interactions.tsv": ["EntrezID1","EntrezID2"],
 "D17_Target_Symbol_Mapping.tsv": ["EntrezID","UniProtID","GeneSymbol","ENSGID"],
 "D18_ICD11.tsv": ["ICD11_code","English_term","Chinese_term","ClassKind"],
 "D19_ICD11_CUI.tsv": ["ICD11_code","CUI"],
 "D20_ICD11_MeSH.tsv": ["ICD11_code","MeSH"],
 "D21_ICD11_DOID.tsv": ["ICD11_code","DOID"],
 "D22_CUI_targets.tsv": ["CUI","EntrezID"],
 "D23_MeSH_targets.tsv": ["MeSH","EntrezID"],
 "D24_DOID_targets.tsv": ["DOID","EntrezID"],
 "SD1_predicted_InChIKey_EntrezID.tsv": ["InChIKey","EntrezID","Predicted_binding_affinity"],
}

LIKELY_RENAMES = {
 # file-level common variants to canonical
 "D16_Protein_protein_interactions.tsv": "D16_Protein_Protein_Interactions.tsv",
}

COLUMN_RENAMES = {
 # Known column-case mismatches or mistakes seen in docs/scripts
 "predicted_binding_affinity": "Predicted_binding_affinity",
 # D3 typo case
 "CHP_ID": "CPM_ID",  # for D3 only
}

def sniff_headers(path):
    with open(path, "r", encoding="utf-8") as f:
        # read first line; handle potential BOM
        first = f.readline().lstrip("\ufeff").rstrip("\n\r")
    # split on tab
    hdrs = first.split("\t")
    return [h.strip() for h in hdrs]

def subset_ok(expected, actual):
    s1 = [h for h in expected if h in actual]
    return len(s1) == len(expected), [h for h in expected if h not in actual]

def best_effort_match(missing, actual):
    # fuzzy suggest by lowercase compare and removing underscores
    sugg = {}
    canon = { re.sub(r"[_\- ]","",h).lower(): h for h in actual }
    for m in missing:
        key = re.sub(r"[_\- ]","",m).lower()
        if key in canon:
            sugg[m] = canon[key]
        else:
            # case-only suggestion
            for a in actual:
                if a.lower() == m.lower():
                    sugg[m] = a
                    break
    return sugg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Directory containing TSV files")
    ap.add_argument("--save", default="", help="Optional path to save JSON report")
    args = ap.parse_args()

    files = [f for f in os.listdir(args.dir) if f.endswith(".tsv")]
    report = OrderedDict()
    print(f"Scanning directory: {args.dir}")
    print(f"Found {len(files)} TSV files.\n")

    # filename canonicalization check
    name_check = []
    for f in files:
        if f in LIKELY_RENAMES:
            name_check.append({"file": f, "suggest_rename_to": LIKELY_RENAMES[f]})
    if name_check:
        print("⚠ Filename canonicalization suggestions:")
        for item in name_check:
            print(f"  - {item['file']} → {item['suggest_rename_to']}")
        print("")

    for fname in sorted(files):
        path = os.path.join(args.dir, fname)
        try:
            hdrs = sniff_headers(path)
        except Exception as e:
            print(f"ERROR reading {fname}: {e}")
            continue

        exp = EXPECTED.get(fname, [])
        ok, missing = subset_ok(exp, hdrs) if exp else (True, [])
        sugg = best_effort_match(missing, hdrs) if missing else {}

        # Column-level rename hints for known pitfalls
        extra_hints = []
        if fname == "SD1_predicted_InChIKey_EntrezID.tsv":
            if "Predicted_binding_affinity" not in hdrs and "predicted_binding_affinity" in hdrs:
                extra_hints.append("Use 'Predicted_binding_affinity' (Cypher should read this exact case).")
        if fname == "D3_CPM_TCMT.tsv":
            if "CPM_ID" not in hdrs and "CHP_ID" in hdrs:
                extra_hints.append("Column should be CPM_ID (not CHP_ID). Fix the source or alias in LOAD CSV.")

        report[fname] = {
            "headers": hdrs,
            "expected_minimal": exp,
            "missing_expected": missing,
            "fuzzy_suggestions": sugg,
            "extra_hints": extra_hints,
        }

        print(f"▶ {fname}")
        print(f"  - Detected headers ({len(hdrs)}): {hdrs}")
        if exp:
            if ok:
                print("  - ✅ Minimal expected headers present.")
            else:
                print(f"  - ❌ Missing expected columns: {missing}")
                if sugg:
                    print(f"    ↪ Fuzzy matches in file: {sugg}")
        if extra_hints:
            for h in extra_hints:
                print(f"  - Hint: {h}")
        print("")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON report to: {args.save}")

if __name__ == "__main__":
    main()
