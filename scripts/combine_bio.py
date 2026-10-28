"""Do the posterior and the ensemble carry complementary uncertainty?

Over evaluate_bio.py outputs (one per replication), per task a held-out
logistic regression on the MAP softmax (entropy + 1-conf) plus the log MI of
each of --sets (aliases from --mi joined by "+"; default: posterior,
ensemble, both); 2 folds split by person, repeated
over --n-splits random person splits (mean AUROC). The person of each probe
comes from regenerating the corpus with the seed and counts it was made with
(make_bio_corpus.py is deterministic). Also an unfitted combination: the
mean of the MIs' ranks per set, AUROC within MAP-entropy quintiles.

  uv run python scripts/combine_bio.py figures/bio_eval_err{1,2,3}.json --seeds 1 2 3 \\
      --counts 40 300 300 100 40 40

E28c, 16 members vs 8 + the posterior (evaluate_bio --subsets 8):
      --mi post="token/projected_alternating25_measured MI" ens16="baseline/lora_ensemble MI" \\
          ens8="baseline/lora_ensemble[8:0] MI" ens8b="baseline/lora_ensemble[8:1] MI" \\
      --sets post ens8 ens16 post+ens8 ens8+ens8b
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from analyze_posterior import fit_logistic, stratified_auroc
from bootstrap_bio import task_masks
from evaluate_posterior import auroc
from llmproj.synthbio import make_people, make_probes


def ranks(x):
    return torch.from_numpy(np.argsort(np.argsort(x.numpy())).astype(np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evals", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", required=True, help="corpus seed per eval")
    ap.add_argument("--freqs", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    ap.add_argument("--counts", type=int, nargs="+", required=True)
    ap.add_argument("--mi", nargs="+", default=["post=token/projected_alternating25_measured MI",
                                                "ens=baseline/lora_ensemble MI"],
                    help="alias=score key")
    ap.add_argument("--sets", nargs="+", default=["post", "ens", "post+ens"])
    ap.add_argument("--tasks", nargs="+", default=["unseen", "rare", "wrong"])
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mi = dict(a.split("=", 1) for a in args.mi)
    sets = {"softmax": [], **{f"+{s}": [mi[a] for a in s.split("+")] for s in args.sets}}
    out = {"mi": mi, "sets": args.sets, "tasks": {t: [] for t in args.tasks}}
    for path, seed in zip(args.evals, args.seeds):
        r = json.load(open(path))["report"]
        probes = make_probes(make_people(tuple(args.freqs), dict(zip(args.freqs, args.counts)), seed))
        freq = np.array(r["probe_freq"])
        if len(probes) != len(freq) or any(p["freq"] != f for p, f in zip(probes, freq)):
            raise SystemExit(f"{path}: regenerated probes do not match (seed / counts?)")
        names = list(dict.fromkeys(p["name"] for p in probes))
        pid = np.array([names.index(p["name"]) for p in probes])
        S = {k: torch.tensor(v, dtype=torch.float64) for k, v in r["probe_scores"].items()}
        soft = torch.stack([S["MAP entropy"], S["MAP 1-conf"]], 1)
        logmi = {k: S[k].clamp_min(1e-12).log()[:, None] for k in mi.values()}
        for t, (pos, neg) in task_masks(freq, np.array(r["probe_correct"], bool)).items():
            if t not in args.tasks:
                continue
            keep = pos | neg
            y = torch.from_numpy(pos[keep])
            res = {"eval": path}
            rng = np.random.default_rng(0)
            for name, ks in sets.items():
                X = torch.cat([soft, *(logmi[k] for k in ks)], 1)[torch.from_numpy(keep)]
                aucs = []
                for _ in range(args.n_splits):
                    fold = torch.from_numpy(rng.permutation(len(names))[pid[keep]] % 2 == 1)
                    for te in (fold, ~fold):
                        logit = fit_logistic(X[~te], y[~te].double())(X[te])
                        aucs.append(auroc(logit[~y[te]], logit[y[te]]))
                res[name] = float(np.mean(aucs))
            k = torch.from_numpy(keep)
            strata = S["MAP entropy"][k]
            for name, ks in sets.items():
                if ks:
                    score = sum(ranks(S[key][k]) for key in ks) / len(ks)
                    res[f"within: {name[1:]}"] = stratified_auroc(y, score, strata, 5)["weighted"]
            out["tasks"][t].append(res)

    for t, rows in out["tasks"].items():
        print(f"\n{t}: held-out logistic AUROC (mean over {args.n_splits} person splits) per replication")
        for k in rows[0]:
            if k == "eval":
                continue
            v = [row[k] for row in rows]
            print(f"  {k:20s} " + "  ".join(f"{x:.3f}" for x in v) + f"   mean {np.mean(v):.3f}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
