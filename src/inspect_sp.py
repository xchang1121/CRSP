import json, os, torch
p = torch.load("/content/cspd/exp2_results/sp_A_heliospay_L128_ctxkl.pt")
print("sp_A shape:", p.shape, "dtype:", p.dtype, "rms:", p.pow(2).mean().sqrt().item(), "max:", p.abs().max().item(), "nan:", torch.isnan(p).any().item())
res = json.load(open("/content/cspd/exp2_results/results.json"))
print("Saved run keys:", list(res["runs"].keys()))
for k in ["stage1_PA", "stage2_PA_PB", "stage3_PA_PB_PC", "single_PB", "single_PC"]:
    if k in res["runs"]:
        print(f"\n=== {k} ===")
        for dom in ["A_heliospay", "B_nvx418", "C_lithox9"]:
            d = res["runs"][k][dom]
            print(f"  {dom}: R={d['rules']}, F={d['facts']}")
            print(f"    rule_pred: {d['sample_rule_pred']!r}")
            print(f"    fact_pred: {d['sample_fact_pred']!r}")
