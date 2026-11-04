"""Minimal cortical-surface renderer for Figure 5 (fsaverage5 pial surface, matplotlib only).

The surfaces in data/raw/fsaverage5 are the FreeSurfer fsaverage5 meshes as
distributed with nilearn (BSD-3-Clause). fsaverage is aligned to MNI305 space,
close to but not identical with MNI152, so the surface is used for display only.
"""
import base64
import gzip
import xml.etree.ElementTree as ET
import zlib
from functools import lru_cache

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb

from _paper import ROOT

SURF = ROOT / "data" / "raw" / "fsaverage5"
VIEWS = {  # direction towards the viewer, screen-x axis, screen-y axis
    "left": (np.array([-1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, 1.0])),
    "right": (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])),
    "top": (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])),
}


def _gifti_arrays(path):
    root = ET.fromstring(gzip.decompress(path.read_bytes()))
    arrays = []
    for da in root.iter("DataArray"):
        dtype = {"NIFTI_TYPE_FLOAT32": "<f4", "NIFTI_TYPE_INT32": "<i4"}[da.get("DataType")]
        raw = base64.b64decode(da.find("Data").text.strip())
        if da.get("Encoding") == "GZipBase64Binary":
            raw = zlib.decompress(raw)
        dims = [int(da.get(f"Dim{i}")) for i in range(int(da.get("Dimensionality")))]
        arrays.append(np.frombuffer(raw, dtype=dtype).reshape(dims))
    return arrays


@lru_cache(None)
def fsaverage(hemi):
    verts, faces = _gifti_arrays(SURF / f"pial_{hemi}.gii.gz")
    sulc = _gifti_arrays(SURF / f"sulc_{hemi}.gii.gz")[0]
    return verts.astype(float), faces.astype(int), sulc.astype(float)


def project_points(xyz, view):
    to_viewer, sx, sy = VIEWS[view]
    return np.c_[xyz @ sx, xyz @ sy], xyz @ to_viewer


def draw_cortex(ax, view, hemis=("left", "right"), gyri="#E4E5E8", sulci="#B4B8C0", light=(0.35, 0.55)):
    """Shaded pial surface seen from `view`; returns the (min, max) of the projected extent."""
    to_viewer, sx, sy = VIEWS[view]
    polys, cols, depth = [], [], []
    g, s = np.array(to_rgb(gyri)), np.array(to_rgb(sulci))
    L = to_viewer * 0.8 + sy * 0.45 - sx * 0.25
    L /= np.linalg.norm(L)
    for hemi in hemis:
        v, f, sulc = fsaverage(hemi)
        tri = v[f]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        front = nrm @ to_viewer > -0.05
        tri, nrm = tri[front], nrm[front]
        w = np.clip((sulc[f[front]].mean(1) + 1.0) / 2.0, 0, 1)[:, None]
        shade = light[0] + light[1] * np.clip(nrm @ L, 0, 1)
        cols.append(np.clip((g * (1 - w) + s * w) * (0.55 + 0.45 * shade[:, None]) + 0.12, 0, 1))
        polys.append(np.stack([tri @ sx, tri @ sy], -1))
        depth.append(tri.mean(1) @ to_viewer)
    polys, cols, depth = np.concatenate(polys), np.concatenate(cols), np.concatenate(depth)
    order = np.argsort(depth)
    ax.add_collection(PolyCollection(polys[order], facecolors=np.c_[cols[order], np.ones(len(order))],
                                     edgecolors="face", linewidths=0.15, zorder=0, rasterized=True))
    allp = polys.reshape(-1, 2)
    return allp.min(0), allp.max(0)
