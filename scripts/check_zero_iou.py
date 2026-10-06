import pandas as pd

m = pd.read_csv("results/per_run_with_failure_flags.csv")
z = m[m["iou"] == 0]
print("runs with IoU == 0:", len(z))
print(z.groupby(["model", "prompt_type", "noise_level"]).size())

sel = ["ann36588_box_n20_t3", "ann339546_box_n20_t2"]
cols = ["model", "prompt_id", "iou", "predicted_score", "mask_path"]
print("\n", m[m["prompt_id"].isin(sel) & (m["model"] == "sam_vit_b")][cols].to_string(index=False))

import os
for p in m[m["prompt_id"].isin(sel) & (m["model"] == "sam_vit_b")]["mask_path"]:
    print(p, "exists locally:", os.path.exists(p))
    