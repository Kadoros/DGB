"""Build data/splits.csv: one row per image with split in {train, val, test_int, test_ext}.

- test_ext : PolypGen C3 + C6 (both centres are fully identified from file names; C3 has
             no sequence data, C6 = 88 single frames + positive seq16..seq23). Never used for
             training, model selection or temperature fitting.
- test_int : whole held-out PolypGen sequences (positive and negative) from the training
             centres. Sequence = one video, the closest thing to a patient ID we have.
- val      : other held-out sequences + 10 % of single frames + 10 % of Kvasir-SEG.
- train    : the rest. Kvasir-SEG has no patient IDs, so it is only used in train/val.
HyperKvasir's segmentation set is not used (likely duplicates Kvasir-SEG).
"""
import argparse

import numpy as np
import pandas as pd

EXT_CENTRES = ("C3", "C6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq_test_frac", type=float, default=0.2)
    ap.add_argument("--seq_val_frac", type=float, default=0.15)
    ap.add_argument("--single_val_frac", type=float, default=0.1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    kv = pd.read_csv("data/kvasir/meta.csv")
    pg = pd.read_csv("data/polypgen/meta.csv")
    kv["split"] = np.where(rng.random(len(kv)) < a.single_val_frac, "val", "train")

    pg["sequence"] = pg["sequence"].fillna("")
    pg["split"] = ""
    pg.loc[pg.center.isin(EXT_CENTRES), "split"] = "test_ext"
    rest = pg.split == ""

    for kind in ("pos_", "neg_"):
        seqs = sorted(s for s in pg.loc[rest, "sequence"].unique() if s.startswith(kind))
        seqs = list(rng.permutation(seqs))
        n_test = max(1, round(len(seqs) * a.seq_test_frac))
        n_val = max(1, round(len(seqs) * a.seq_val_frac))
        for s in seqs[:n_test]:
            pg.loc[pg.sequence == s, "split"] = "test_int"
        for s in seqs[n_test:n_test + n_val]:
            pg.loc[pg.sequence == s, "split"] = "val"

    single = rest & (pg.sequence == "")
    pg.loc[single, "split"] = np.where(rng.random(single.sum()) < a.single_val_frac, "val", "train")
    pg.loc[pg.split == "", "split"] = "train"

    df = pd.concat([kv, pg], ignore_index=True)
    df["sequence"] = df["sequence"].fillna("")
    df.to_csv("data/splits.csv", index=False)
    print(df.groupby(["split", "dataset", "positive"]).size().unstack(fill_value=0))
    for sp in ("val", "test_int"):
        print(sp, sorted(df.loc[(df.split == sp) & (df.sequence != ""), "sequence"].unique()))


if __name__ == "__main__":
    main()
