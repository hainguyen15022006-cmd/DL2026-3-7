import pandas as pd

pr = pd.read_csv("results/prompts.csv")
print("prompts.csv columns:", list(pr.columns))

m = pd.read_csv("results/per_run_with_failure_flags.csv")
sam = m[m["model"] == "sam_vit_b"]
sel = pd.read_csv("results/selected_examples.csv")

cols = [c for c in pr.columns if "inside" in c.lower() or "box_iou" in c.lower()]
print("\n=== 4 selected examples: prompt quality ===")
print(sel[["group", "annotation_id", "prompt_id", "iou"]]
      .merge(pr[["prompt_id"] + cols], on="prompt_id", how="left")
      .round(3).to_string(index=False))

j = sam.merge(pr[["prompt_id"] + cols], on="prompt_id", how="left")
for c in cols:
    print(f"\n=== mean IoU of SAM ViT-B by {c} ===")
    if "inside" in c.lower():
        print(j[j["prompt_type"] == "point"].groupby(["noise_level", c])["iou"]
              .agg(["size", "mean"]).round(3))
    else:
        j["_b"] = pd.cut(j[j["prompt_type"] == "box"][c], [-0.01, 0.0, 0.5, 0.8, 1.0])
        print(j[j["prompt_type"] == "box"].groupby("_b", observed=True)["iou"]
              .agg(["size", "mean"]).round(3))