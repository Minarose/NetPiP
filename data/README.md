# Data

## Networks used in the paper (`networks/`)

Each folder holds `edges.csv` (undirected edges, 1-based node indices) and
`nodes.csv` (node labels). Load one with

```python
import csv, netpip
n = sum(1 for _ in open("data/networks/hospital/nodes.csv")) - 1
A = netpip.load_edge_list("data/networks/hospital/edges.csv", n_nodes=n)
```

All six are binary and undirected; every component, including isolated nodes,
is kept, so N is the full node count.

| Folder | Nodes | Edges | Density | Construction | Source and licence |
|---|---:|---:|---:|---|---|
| `hospital` | 75 | 135 | 4.87% | Face-to-face contacts among 29 patients and 46 staff in a hospital ward (20-s windows); an edge joins two people whose cumulative contact lasted ≥ 20 min. `nodes.csv`: participant ID and role (PAT, NUR, MED, ADM). | SocioPatterns, [hospital ward dynamic contact network](https://www.sociopatterns.org/datasets/hospital-ward-dynamic-contact-network/); Vanhems et al., *PLoS ONE* 8, e73970 (2013). Reuse under the terms on the SocioPatterns page; please cite the original article. |
| `airport` | 3,214 | 18,858 | 0.37% | OpenFlights direct routes (zero stops) between airports with known IDs; either direction is an undirected edge, duplicate airline routes collapse, self-loops dropped. Routes are a June 2014 snapshot. `nodes.csv`: OpenFlights ID, IATA code, name, city, country. | [OpenFlights](https://openflights.org/data) airports and routes, Open Database License ([ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/)). This derived network is shared under the ODbL. |
| `yeast` | 254 | 1,865 | 5.80% | *S. cerevisiae* ORFs annotated to DNA repair (SGD GO-slim, GO:0006281); an edge joins two proteins with any BioGRID 5.0.261 physical interaction; largest connected component. `nodes.csv`: systematic name and gene symbol. | [BioGRID](https://thebiogrid.org) 5.0.261 (MIT licence; Oughtred et al., *Nucleic Acids Res.* 49, D529 (2021)) and [SGD](https://www.yeastgenome.org) GO-slim mapping (CC BY 4.0). |
| `language` | 66 | 113 | 5.27% | MEG expressive language network (covert verb generation, 19 adolescents). Group-averaged phase-slope-index coupling, thresholded at the highest value that keeps ≥ 75% of regions in the LCC. `nodes.csv`: AAL label, hemisphere and MNI coordinates. | This study; raw group and per-participant matrices in `raw/language_meg/`. |
| `watts_strogatz` | 200 | 400 | 2.01% | `networkx.watts_strogatz_graph(200, 4, 0.1, seed=13353250991570536326)` | Synthetic |
| `erdos_renyi` | 60 | 60 | 3.39% | `networkx.gnm_random_graph(60, 60, seed=202609220)` | Synthetic |

## Rebuilding the networks from their sources

```bash
python data/build_networks.py                 # downloads sources into data/raw/downloads/ (not in git)
python data/build_networks.py --source-dir /path/to/existing/files
```

The script checks every rebuilt adjacency matrix against the SHA-256
fingerprint of the network analysed in the paper (`results/paper_1B/*/run.json`)
and warns if an upstream file has changed since. The hospital contact list must
be downloaded by hand from the SocioPatterns page and saved as
`data/raw/downloads/hospital_contacts.dat.gz`. The BioGRID archive is about
700 MB. All six networks were confirmed to rebuild identically.

## Other files

* `raw/language_meg/` – MEG phase-slope-index adjacency matrices (19 participants
  and the group average), node MNI coordinates and AAL labels. See its README.
* `raw/fsaverage5/` – FreeSurfer fsaverage5 pial surface and sulcal depth, as
  distributed with nilearn (BSD-3-Clause), used only to draw Fig. 5.
