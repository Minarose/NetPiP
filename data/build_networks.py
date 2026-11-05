"""Rebuild the six paper networks from their original sources and check them.

The finished networks are already in data/networks/ (edges.csv, nodes.csv), so
this script is only needed to audit how they were made. Third-party raw files
are downloaded into data/raw/downloads/ (not tracked by git) and verified by
SHA-256 against the files used for the paper; if an upstream file has changed
since, the script says so and the rebuilt network may differ.

Run:  python data/build_networks.py [hospital airport yeast language synthetic]
      python data/build_networks.py yeast --source-dir /path/with/existing/files
"""
import argparse
import csv
import gzip
import hashlib
import io
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NETS = ROOT / "data" / "networks"
CACHE = ROOT / "data" / "raw" / "downloads"
sys.path.insert(0, str(ROOT / "src"))
import netpip  # noqa: E402

SOURCES = {
    "hospital_contacts.dat.gz": dict(
        url="https://www.sociopatterns.org/datasets/hospital-ward-dynamic-contact-network/",
        sha256="12124ddc438300bcaea0022d4937e0e566679e1fd8f11f16a62214fbebc2b012", manual=True),
    "airports.dat": dict(url="https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"),
    "routes.dat": dict(url="https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"),
    "go_slim_mapping.tab": dict(
        url="https://downloads.yeastgenome.org/curation/literature/go_slim_mapping.tab",
        sha256="625fdef266cb7410a7c08225009544df293260d57574c73fb83894b6dc428f75"),
    "biogrid_5.0.261.zip": dict(
        url="https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-5.0.261/"
            "BIOGRID-ORGANISM-5.0.261.tab3.zip",
        sha256="3c62cf57657b4f4d8c583717354518254d1788735613eaa530e6a4f4906a9338"),
}
# Adjacency fingerprints of the networks analysed in the paper (results/*/run.json).
EXPECTED = {k: __import__("json").loads((ROOT / "results" / k / "run.json").read_text())["adjacency_sha256"]
            for k in ["hospital", "airport", "yeast", "language", "watts_strogatz", "erdos_renyi"]}


def sha(path_or_array):
    if isinstance(path_or_array, Path):
        return hashlib.sha256(path_or_array.read_bytes()).hexdigest()
    return hashlib.sha256(np.ascontiguousarray(path_or_array).tobytes()).hexdigest()


def fetch(name, source_dir=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / name
    spec = SOURCES[name]
    if not dest.exists() and source_dir is not None:
        for cand in Path(source_dir).rglob("*"):
            if cand.is_file() and (cand.name == name or ("sha256" in spec and cand.stat().st_size < 2e9
                                                          and cand.name.split(".")[-1] == name.split(".")[-1]
                                                          and sha(cand) == spec["sha256"])):
                shutil.copy2(cand, dest)
                break
    if not dest.exists():
        if spec.get("manual"):
            raise SystemExit(f"Download the hospital contact list from {spec['url']} "
                             f"and save it as {dest} (or pass --source-dir).")
        print(f"  downloading {spec['url']}")
        urllib.request.urlretrieve(spec["url"], dest)
    if "sha256" in spec and sha(dest) != spec["sha256"]:
        print(f"  WARNING: {name} differs from the file used in the paper (upstream changed?)")
    return dest


def check(key, A):
    ok = sha(A.astype(np.uint8)) == EXPECTED[key]
    print(f"  {key}: {len(A)} nodes, {int(A.sum() // 2)} edges -> "
          f"{'identical to the paper network' if ok else 'DIFFERS from the paper network'}")
    return ok


def hospital(source_dir):
    """Face-to-face contacts (20-s windows); edge if a pair's total contact >= 20 min. All 75 people kept."""
    path = fetch("hospital_contacts.dat.gz", source_dir)
    rows = [line.split() for line in gzip.open(path, "rt") if line.strip()]
    people = {}
    duration = Counter()
    for t, i, j, ri, rj in rows:
        people[int(i)], people[int(j)] = ri, rj
        duration[tuple(sorted((int(i), int(j))))] += 20
    ids = sorted(people)
    index = {p: k for k, p in enumerate(ids)}
    A = np.zeros((len(ids), len(ids)), np.uint8)
    for (a, b), seconds in duration.items():
        if seconds >= 1200:
            A[index[a], index[b]] = A[index[b], index[a]] = 1
    return check("hospital", A)


def airport(source_dir):
    """OpenFlights direct routes (stops = 0) between known airports; undirected, duplicates collapsed."""
    airports = {int(r[0]): r for r in csv.reader(fetch("airports.dat", source_dir).open(encoding="utf-8"))}
    edges = set()
    for r in csv.reader(fetch("routes.dat", source_dir).open(encoding="utf-8")):
        if r[7] != "0" or not r[3].isdigit() or not r[5].isdigit():
            continue
        u, v = int(r[3]), int(r[5])
        if u in airports and v in airports and u != v:
            edges.add((min(u, v), max(u, v)))
    ids = sorted({x for e in edges for x in e})
    index = {a: k for k, a in enumerate(ids)}
    A = np.zeros((len(ids), len(ids)), np.uint8)
    for u, v in edges:
        A[index[u], index[v]] = A[index[v], index[u]] = 1
    return check("airport", A)


def yeast(source_dir):
    """SGD GO-slim DNA repair (GO:0006281) ORFs; BioGRID 5.0.261 physical interactions; largest component."""
    import networkx as nx

    genes = {r[0]: r for r in csv.reader(fetch("go_slim_mapping.tab", source_dir).open(), delimiter="\t")
             if r[5] == "GO:0006281" and r[6].startswith("ORF|")}
    G = nx.Graph()
    G.add_nodes_from(sorted(genes))
    with zipfile.ZipFile(fetch("biogrid_5.0.261.zip", source_dir)) as z:
        name = next(n for n in z.namelist() if "Saccharomyces_cerevisiae_S288c" in n)
        for r in csv.DictReader(io.TextIOWrapper(z.open(name)), delimiter="\t"):
            a, b = r["Systematic Name Interactor A"], r["Systematic Name Interactor B"]
            if a in genes and b in genes and a != b and r["Organism ID Interactor A"] == "559292" \
                    and r["Organism ID Interactor B"] == "559292" and r["Experimental System Type"] == "physical":
                G.add_edge(a, b)
    largest = sorted(max(nx.connected_components(G), key=len))
    A = nx.to_numpy_array(G.subgraph(largest), nodelist=largest, dtype=np.uint8, weight=None)
    return check("yeast", A)


def language(_):
    """Group-average MEG PSI matrix; highest threshold (of 1,000) whose LCC keeps >= 75% of regions."""
    from scipy.io import loadmat
    from scipy.sparse.csgraph import connected_components

    z = loadmat(ROOT / "data/raw/language_meg/group_average/AVG_broadband_psi_adj_giant75_nonexcluded.mat")
    W = z["avg_psi_adj"]
    n = len(W)
    best = None
    for t in np.linspace(W.min(), W.max(), 1000):
        A = (W > t).astype(np.uint8)
        np.fill_diagonal(A, 0)
        _, lab = connected_components(A, directed=False)
        if np.bincount(lab).max() >= 0.75 * n:
            best = A
    best = np.maximum(best, best.T)
    print(f"  saved threshold in the .mat file: {float(np.ravel(z['threshold_value'])[0]):.6f}")
    return check("language", best)


def synthetic(_):
    """NetworkX graphs with the paper's seeds (networkx 3.x)."""
    import networkx as nx

    er = nx.gnm_random_graph(60, 60, seed=202609220)
    ws = nx.watts_strogatz_graph(200, 4, 0.1, seed=13353250991570536326)
    a = check("erdos_renyi", nx.to_numpy_array(er, nodelist=range(60), dtype=np.uint8, weight=None))
    b = check("watts_strogatz", nx.to_numpy_array(ws, nodelist=range(200), dtype=np.uint8, weight=None))
    return a and b


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("networks", nargs="*", default=["hospital", "airport", "yeast", "language", "synthetic"])
    p.add_argument("--source-dir", help="folder that already contains the raw source files")
    args = p.parse_args()
    ok = True
    for name in args.networks:
        print(f"== {name}")
        ok &= globals()[name](args.source_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
