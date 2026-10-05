from pathlib import Path
import pandas as pd

RAW = "results/raw_predictions.csv"
OUT = Path("results")
KEYS = ["model", "annotation_id", "prompt_type"]

df = pd.read_csv(RAW)
print("rows:", len(df), "| status counts:", df["status"].value_counts().to_dict())

ok = df[df["status"] == "ok"].copy()
ok["iou"] = ok["iou"].astype(float)
ok["noise_level"] = ok["noise_level"].astype(float)

clean = (ok[(ok["noise_level"] == 0) & (ok["trial"] == 0)]
         [KEYS + ["iou"]].rename(columns={"iou": "iou_clean"}))
m = ok.merge(clean, on=KEYS, how="left")
m["iou_drop"] = m["iou_clean"] - m["iou"]
m["F1"] = m["iou"] < 0.5
m["F2"] = (m["noise_level"] > 0) & (m["iou_drop"] >= 0.2)
m["failure"] = m["F1"] | m["F2"]
m.to_csv(OUT / "per_run_with_failure_flags.csv", index=False)

g = ["model", "prompt_type", "noise_level"]
s = (m.groupby(g)
       .agg(n=("iou", "size"), mean_iou=("iou", "mean"),
            median_iou=("iou", "median"), n_F1=("F1", "sum"),
            n_F2=("F2", "sum"), n_failure=("failure", "sum"))
       .reset_index())
s["failure_rate"] = s["n_failure"] / s["n"]
s.to_csv(OUT / "failure_summary.csv", index=False)
print("\n=== FAILURE SUMMARY ===")
print(s.round(3).to_string(index=False))

# Setup 2: point vs box, SAM ViT-B, clean, paired
c = m[(m["model"] == "sam_vit_b") & (m["noise_level"] == 0)]
p = c.pivot(index="annotation_id", columns="prompt_type", values="iou")
p["box_minus_point"] = p["box"] - p["point"]
print("\n=== SETUP 2 (SAM ViT-B clean, paired) ===")
print("n pairs:", len(p), "| mean box-point:", round(p["box_minus_point"].mean(), 4),
      "| box>point in", int((p["box_minus_point"] > 0).sum()), "instances")

# Setup 1: model comparison, clean, paired
c1 = m[m["noise_level"] == 0]
for pt in ["point", "box"]:
    q = c1[c1["prompt_type"] == pt].pivot(index="annotation_id", columns="model", values="iou")
    q["sam_minus_mobile"] = q["sam_vit_b"] - q["mobile_sam"]
    print(f"\n=== SETUP 1 ({pt}) === n:", len(q),
          "| mean SAM-Mobile:", round(q["sam_minus_mobile"].mean(), 4))

# 4 illustrative examples by the public rule
sam = m[m["model"] == "sam_vit_b"]
good = (sam[(sam["prompt_type"] == "box") & (sam["noise_level"] == 0)]
        .sort_values(["iou", "annotation_id"], ascending=[False, True]).head(2)
        .assign(group="good"))
bad = (sam[sam["noise_level"] == 0.2]
       .sort_values(["iou_drop", "annotation_id"], ascending=[False, True]).head(2)
       .assign(group="degraded"))
sel = pd.concat([good, bad])
sel.to_csv(OUT / "selected_examples.csv", index=False)
print("\n=== 4 SELECTED EXAMPLES ===")
print(sel[["group", "annotation_id", "image_id", "prompt_id", "prompt_type",
           "noise_level", "trial", "iou", "iou_clean", "iou_drop"]]
      .round(3).to_string(index=False))