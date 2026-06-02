import json
from collections import Counter
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

d = json.load(open("results/arrow_parser_verify.json"))
fails = d["all_failures"]
print("total diverged:", len(fails))

sigs = Counter()
bucket_examples: dict[str, list] = {}

def classify(pred: str, true: str) -> str:
    pm = Chem.MolFromSmiles(pred)
    tm = Chem.MolFromSmiles(true)
    if pm is None or tm is None:
        return "parse_fail"
    da = pm.GetNumAtoms() - tm.GetNumAtoms()
    db = pm.GetNumBonds() - tm.GetNumBonds()
    pf = len(Chem.GetMolFrags(pm)); tf = len(Chem.GetMolFrags(tm))
    p_charged = sum(1 for a in pm.GetAtoms() if a.GetFormalCharge() != 0)
    t_charged = sum(1 for a in tm.GetAtoms() if a.GetFormalCharge() != 0)
    p_arom = sum(1 for a in pm.GetAtoms() if a.GetIsAromatic())
    t_arom = sum(1 for a in tm.GetAtoms() if a.GetIsAromatic())
    p_rings = pm.GetRingInfo().NumRings()
    t_rings = tm.GetRingInfo().NumRings()
    if da == 0 and db == 0:
        if p_charged != t_charged:
            return "charge_only"
        if p_arom != t_arom:
            return "aromaticity_only"
        return "same_formula_other"
    if da == 0 and db == -1:
        return "missing_bond(ring-not-closed)"
    if da == 0 and p_rings < t_rings:
        return "ring_open_vs_closed"
    if da > 0 and pf > tf:
        return "extra_fragment"
    return f"other(da{da:+d} db{db:+d} fr{pf-tf:+d})"

for r in fails:
    b = classify(r["pred"], r["true"])
    sigs[b] += 1
    bucket_examples.setdefault(b, []).append(r)

for b, c in sigs.most_common():
    print(f"  {c:3d}  {b}")
print()
for b, exs in bucket_examples.items():
    print(f"--- {b} ({len(exs)}) ---")
    for r in exs[:3]:
        print(f"  {r['rxn_id']}")
        print(f"    pred: {r['pred'][:110]}")
        print(f"    true: {r['true'][:110]}")
