# PiP input matrices (giant-75 thresholded)

These are the **binary, symmetric, undirected adjacency matrices that feed directly into PiP** (`scripts/2_pip_converge.m`). They are produced from the raw broadband PSI matrices described in [`../README.md`](../README.md) by the **giant-component 75 % thresholding rule** implemented in `scripts/1_make_giant75_per_subject.py` (and the MATLAB equivalent `scripts/1_threshold_giant75.m`):

> For each subject, sweep 1000 candidate density thresholds and keep the most permissive cut whose largest connected component covers at least 75 % of the 66 nodes.

For the group-average matrix (`group_average/AVG_*.mat`), `scripts/1_make_giant75_avg.py` averages the per-subject **weighted** PSI matrices (non-excluded subjects only — AD15 / AD16 dropped per `results/pip_cluster/attack_outliers.csv`) and then re-applies the same giant-75 rule.

## Layout

| Folder | Contents |
|---|---|
| `group_average/` | One `AVG_broadband_psi_adj_giant75_nonexcluded.mat` (cohort mean, n = 19) |
| `individual/`    | 19 per-subject `AD??_broadband_psi_adj_giant75.mat` |

Each `.mat` contains a single variable `psi_adj` (66 × 66, dtype double, values 0 / 1).

## Density and largest-connected-component fraction

All matrices are 66-node graphs. Density = #edges / (66 × 65 / 2); LCC % = size of the largest connected component / 66.

| Matrix | edges | density | LCC | LCC % | # components |
|---|---:|---:|---:|---:|---:|
| `group_average/AVG_…_giant75_nonexcluded.mat` | 113 | 5.27 % | 51 | 77.27 % | 8 |
| `individual/AD01_…_giant75.mat` |  76 | 3.54 % | 51 | 77.27 % | 13 |
| `individual/AD02_…_giant75.mat` |  71 | 3.31 % | 51 | 77.27 % | 15 |
| `individual/AD03_…_giant75.mat` |  79 | 3.68 % | 50 | 75.76 % | 16 |
| `individual/AD04_…_giant75.mat` |  63 | 2.94 % | 52 | 78.79 % | 13 |
| `individual/AD05_…_giant75.mat` |  74 | 3.45 % | 50 | 75.76 % | 16 |
| `individual/AD06_…_giant75.mat` |  81 | 3.78 % | 51 | 77.27 % | 14 |
| `individual/AD07_…_giant75.mat` |  70 | 3.26 % | 50 | 75.76 % | 14 |
| `individual/AD08_…_giant75.mat` |  82 | 3.82 % | 52 | 78.79 % | 13 |
| `individual/AD09_…_giant75.mat` | 120 | 5.59 % | 51 | 77.27 % | 13 |
| `individual/AD10_…_giant75.mat` |  71 | 3.31 % | 51 | 77.27 % | 15 |
| `individual/AD11_…_giant75.mat` |  71 | 3.31 % | 50 | 75.76 % | 16 |
| `individual/AD12_…_giant75.mat` |  68 | 3.17 % | 50 | 75.76 % | 14 |
| `individual/AD13_…_giant75.mat` |  74 | 3.45 % | 52 | 78.79 % | 15 |
| `individual/AD14_…_giant75.mat` |  75 | 3.50 % | 51 | 77.27 % | 15 |
| `individual/AD17_…_giant75.mat` |  75 | 3.50 % | 51 | 77.27 % | 14 |
| `individual/AD18_…_giant75.mat` |  65 | 3.03 % | 55 | 83.33 % | 12 |
| `individual/AD19_…_giant75.mat` |  99 | 4.62 % | 50 | 75.76 % | 15 |
| `individual/AD20_…_giant75.mat` |  98 | 4.57 % | 50 | 75.76 % | 15 |
| `individual/AD21_…_giant75.mat` |  68 | 3.17 % | 50 | 75.76 % | 15 |

### Per-subject summary (n = 19, non-excluded)

| Statistic | Density | LCC % |
|---|---|---|
| Mean   | 3.63 % | 77.19 % |
| Median | 3.45 % | 77.27 % |
| Range  | 2.94 % – 5.59 % | 75.76 % – 83.33 % |

Numbers above were recomputed from the bundled `.mat` files via `scipy.sparse.csgraph.connected_components`; reproduce with:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np, scipy.io as sio
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
for p in sorted(Path("data/PSI_broadband_MEG_mats").rglob("*.mat")):
    A = (sio.loadmat(p)["psi_adj"] > 0).astype(float)
    A = np.triu(A, 1); A = A + A.T; np.fill_diagonal(A, 0)
    n = A.shape[0]; e = int(A.sum() / 2)
    n_comp, lab = connected_components(csr_matrix(A), directed=False)
    lcc = int(np.bincount(lab).max())
    print(f"{p.relative_to('data/PSI_broadband_MEG_mats')}: n={n} edges={e} dens={e/(n*(n-1)/2)*100:.2f}% LCC={lcc} ({lcc/n*100:.2f}%) comps={n_comp}")
PY
```
