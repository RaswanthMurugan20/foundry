#!/usr/bin/env python3
"""
Analyze structural diversity of AlphaFold3 models by computing TM-scores,
clustering folds, and visualizing fold space.
"""

import argparse
import json
import colorsys
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Limit OpenMP/BLAS threading to avoid resource errors on constrained systems.
for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
    os.environ.setdefault(_env, "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS

AA_CATEGORIES = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
NAN_CATEGORY = "NaN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute all-vs-all TM-scores for AlphaFold3 structures, cluster folds, "
            "and visualize fold space."
        )
    )
    parser.add_argument(
        "--csv_path",
        default="enzyme_output_dserine_basic_lysine_fixed/rank_0/best_per_backbone_filtered.csv",
        help="CSV file with at least two columns: backbone ID and sequence ID.",
    )
    parser.add_argument(
        "--jaccard_csv_path",
        default="enzyme_output_dserine_basic_lysine_fixed/rank_0/best_per_backbone.csv",
        help=(
            "CSV file for Jaccard plots/metrics. Defaults to --csv_path, or if it ends "
            "with '_filtered.csv' uses the unfiltered file when present."
        ),
    )
    parser.add_argument(
        "--root",
        default="enzyme_output_dserine_basic_lysine_fixed",
        help="Root directory containing rank_*/backbone_* folders.",
    )
    parser.add_argument(
        "--output-prefix",
        default="tm_scores",
        help="Prefix for TM-score matrix outputs (CSV and NPY).",
    )
    parser.add_argument(
        "--tm-threshold",
        type=float,
        default=0.5,
        help="TM-score threshold for assigning same fold (default: 0.5).",
    )
    parser.add_argument(
        "--tm_cache_path",
        default="tm_score_matrix.npy",
        help="Path to cached TM-score matrix (.npy) to load or overwrite.",
    )
    parser.add_argument(
        "--tm_cache_csv_path",
        default="tm_score_matrix.csv",
        help="Path to write cached TM-score matrix as CSV when recomputing.",
    )
    parser.add_argument(
        "--theozyme_pdb",
        default="theozymes/Theozyme_DFT_resid_rfd3.pdb",
        help="Path to theozyme PDB used to label intended residue identities.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output root directory. Defaults to ./[theozyme_stem]_analysis/",
    )
    return parser.parse_args()


def construct_cif_path(backbone: str, seq: str, root_dir: Path) -> Path:
    """Construct CIF path from tokens like backbone_0001_r0 and seq_0001."""
    backbone_match = re.match(r"backbone_(\d+)_r(\d+)", backbone)
    seq_match = re.match(r"seq_(\d+)", seq)
    if backbone_match is None:
        raise ValueError(f"Cannot parse backbone token: {backbone}")
    if seq_match is None:
        raise ValueError(f"Cannot parse sequence token: {seq}")

    backbone_idx, rank = backbone_match.groups()
    seq_idx = int(seq_match.group(1))

    folder = root_dir / f"rank_{rank}" / f"backbone_{backbone_idx}_r{rank}"
    cif_name = f"unnamed_b0_d{seq_idx}.cif"
    return folder / cif_name


def read_structure_table(csv_path: Path, root_dir: Path) -> List[Dict]:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError("CSV must contain at least two columns.")

    entries: List[Dict] = []
    for idx, row in df.iterrows():
        backbone = str(row.iloc[0])
        seq = str(row.iloc[1])
        cif_path = construct_cif_path(backbone, seq, root_dir)
        if cif_path.exists():
            entries.append(
                {
                    "row_index": idx,
                    "backbone": backbone,
                    "sequence": seq,
                    "id": f"{backbone}|{seq}",
                    "cif_path": cif_path,
                }
            )
        else:
            print(f"Warning: missing CIF file, skipping: {cif_path}", file=sys.stderr)

    if not entries:
        raise FileNotFoundError("No CIF files found for any CSV rows.")
    return entries


def parse_tm_score(output: str) -> float:
    matches: List[float] = []
    for line in output.splitlines():
        line = line.strip()
        if "TM-score" not in line:
            continue
        match = re.search(r"TM-score\s*=\s*([0-9]*\.?[0-9]+)", line)
        if match:
            matches.append(float(match.group(1)))

    if matches:
        return max(matches)
    raise ValueError("TM-score not found in aligner output.")


def run_tm_alignment(bin_path: str, model_a: Path, model_b: Path) -> float:
    """Run TM-align/US-align and return TM-score; NaN on failure."""
    cmd = [bin_path, str(model_a), str(model_b)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    combined_output = (result.stdout or "") + (result.stderr or "")

    if result.returncode != 0:
        print(
            f"Warning: TM-align failed for {model_a} vs {model_b}: "
            f"return code {result.returncode}",
            file=sys.stderr,
        )
        return np.nan
    try:
        return parse_tm_score(combined_output)
    except ValueError:
        print(
            f"Warning: could not parse TM-score for {model_a} vs {model_b}",
            file=sys.stderr,
        )
        return np.nan


def build_tm_matrix(models: List[Path], tm_bin: str) -> np.ndarray:
    """Compute symmetric TM-score matrix."""
    n = len(models)
    tm = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            score = run_tm_alignment(tm_bin, models[i], models[j])
            tm[i, j] = tm[j, i] = score
    return tm


def cluster_structures(tm_matrix: np.ndarray, tm_threshold: float) -> np.ndarray:
    """Cluster using average-linkage with distance threshold derived from TM."""
    dist = 1.0 - tm_matrix
    dist = np.nan_to_num(dist, nan=1.0)
    np.fill_diagonal(dist, 0.0)

    distance_threshold = 1.0 - tm_threshold
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    return model.fit_predict(dist)


def save_tm_outputs(
    tm_matrix: np.ndarray, ids: List[str], output_prefix: str
) -> pd.DataFrame:
    np.save(f"{output_prefix}.npy", tm_matrix)
    df = pd.DataFrame(tm_matrix, index=ids, columns=ids)
    df.to_csv(f"{output_prefix}.csv")
    return df


def save_cluster_mapping(
    entries: List[Dict], labels: np.ndarray, output_path: Path
) -> None:
    records = []
    for entry, label in zip(entries, labels):
        records.append(
            {
                "row_index": entry["row_index"],
                "structure_id": entry["id"],
                "backbone": entry["backbone"],
                "sequence": entry["sequence"],
                "cif_path": str(entry["cif_path"]),
                "cluster_id": int(label),
            }
        )
    pd.DataFrame(records).to_csv(output_path, index=False)


def plot_mds(dist_matrix: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    dist = np.nan_to_num(dist_matrix, nan=1.0)
    np.fill_diagonal(dist, 0.0)
    mds = MDS(
        n_components=2,
        metric="precomputed",
        n_init=4,
        init="random",
        random_state=0,
    )
    coords = mds.fit_transform(dist)

    unique_labels = sorted(set(labels))
    num_clusters = len(unique_labels)
    if num_clusters == 0:
        return
    hues = np.linspace(0, 1, num_clusters, endpoint=False)
    hues = np.concatenate([hues[::2], hues[1::2]])  # maximize separation between neighbors
    sat_levels = [0.95, 0.8, 0.65]
    val_levels = [0.95, 0.8]
    colors = []
    for idx, h in enumerate(hues):
        s = sat_levels[idx % len(sat_levels)]
        v = val_levels[(idx // len(sat_levels)) % len(val_levels)]
        colors.append(colorsys.hsv_to_rgb(h, s, v))
    color_map = {cid: colors[idx] for idx, cid in enumerate(unique_labels)}

    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "H", "p", "8"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for idx, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=50,
            alpha=0.8,
            color=color_map[cluster_id],
            marker=markers[idx % len(markers)],
            label=f"Cluster {cluster_id}",
        )
    ax.set_xlabel("Dim 1 (MDS on TM-distance)")
    ax.set_ylabel("Dim 2 (MDS on TM-distance)")
    ncol = max(1, min(num_clusters, 8))
    nrows = int(np.ceil(num_clusters / ncol))
    ax.legend(
        title="Fold clusters",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=ncol,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.12 + 0.04 * nrows)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def setup_output_dirs(out_dir: Optional[str], theozyme_pdb: Path) -> Dict[str, Path]:
    if out_dir:
        root = Path(out_dir)
    else:
        theozyme_stem = theozyme_pdb.stem or "theozyme"
        root = Path(f"{theozyme_stem}_analysis")

    subdirs = {
        "root": root,
        "tables": root / "tables",
        "plots": root / "plots",
        "clusters": root / "clusters",
        "metrics": root / "metrics",
        "logs": root / "logs",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def load_tm_cache(cache_path: Path, num_structures: int) -> Optional[np.ndarray]:
    msg = "Cache invalid, recomputing TM matrix"
    try:
        matrix = np.load(cache_path)
    except Exception as exc:
        print(f"{msg}: {exc}", file=sys.stderr)
        return None

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != num_structures:
        print(msg, file=sys.stderr)
        return None
    if not np.isfinite(matrix).any():
        print(msg, file=sys.stderr)
        return None
    return matrix


def extract_active_site_residues(cif_path: Path) -> Optional[List[Dict[str, str]]]:
    try:
        import gemmi  # type: ignore
    except ImportError:
        print("Warning: gemmi not installed; skipping active-site analysis.", file=sys.stderr)
        return None

    try:
        doc = gemmi.cif.read_file(str(cif_path))
        structure = gemmi.make_structure_from_block(doc.sole_block())
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to parse CIF {cif_path}: {exc}", file=sys.stderr)
        return None

    model = structure[0]
    ligand_atoms = []
    for chain in model:
        for residue in chain:
            if residue.name != "L:G":
                continue
            for atom in residue:
                if atom.element.is_hydrogen:
                    continue
                ligand_atoms.append(atom.pos)

    if not ligand_atoms:
        print(f"Warning: ligand L:G not found in {cif_path}; skipping.", file=sys.stderr)
        return None

    cutoff_sq = 4.0 * 4.0
    active_site = []
    for chain in model:
        for residue in chain:
            res_info = gemmi.find_tabulated_residue(residue.name)
            if res_info.kind != gemmi.ResidueKind.AA:
                continue
            hit = False
            for atom in residue:
                if atom.element.is_hydrogen:
                    continue
                for lig_pos in ligand_atoms:
                    dx = atom.pos.x - lig_pos.x
                    dy = atom.pos.y - lig_pos.y
                    dz = atom.pos.z - lig_pos.z
                    if (dx * dx + dy * dy + dz * dz) <= cutoff_sq:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                active_site.append(
                    {
                        "chain_id": chain.name,
                        "residue_number": str(residue.seqid.num),
                        "resname": residue.name,
                    }
                )

    return active_site


def compute_active_site_data(
    entries: List[Dict],
) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, set]]]:
    rows: List[Dict[str, str]] = []
    site_info: Dict[str, Dict[str, set]] = {}
    for entry in entries:
        residues = extract_active_site_residues(entry["cif_path"])
        positions: set = set()
        resnames: set = set()
        pos_to_resname: Dict[str, str] = {}
        if residues:
            for res in residues:
                pos = f"{res['chain_id']}{res['residue_number']}"
                positions.add(pos)
                resnames.add(res["resname"])
                if pos not in pos_to_resname:
                    pos_to_resname[pos] = res["resname"]
                rows.append(
                    {
                        "structure_id": entry["id"],
                        "chain_id": res["chain_id"],
                        "residue_number": res["residue_number"],
                        "resname": res["resname"],
                    }
                )
        site_info[entry["id"]] = {
            "positions": positions,
            "resnames": resnames,
            "pos_to_resname": pos_to_resname,
        }
    return rows, site_info


def save_active_site_outputs(rows: List[Dict[str, str]], out_dirs: Dict[str, Path]) -> None:
    if not rows:
        print("Warning: no active-site residues found; skipping outputs.", file=sys.stderr)
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dirs["tables"] / "active_site_residues.csv", index=False)

    counts = df["resname"].value_counts().rename_axis("resname").reset_index(name="count")
    counts.to_csv(out_dirs["tables"] / "active_site_residue_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts["resname"], counts["count"])
    ax.set_xlabel("Residue")
    ax.set_ylabel("Count")
    ax.set_title("Active-site residue frequency")
    fig.savefig(out_dirs["plots"] / "active_site_residue_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def normalize_res_id(res_id: str) -> Optional[str]:
    if not isinstance(res_id, str):
        return None
    match = re.match(r"([A-Za-z])\s*(-?\d+)", res_id.strip())
    if not match:
        return None
    return f"{match.group(1)}{int(match.group(2))}"


def infer_jaccard_csv_path(csv_path: Path, override: Optional[str]) -> Path:
    if override:
        return Path(override)
    if "_filtered" in csv_path.name:
        candidate = csv_path.with_name(csv_path.name.replace("_filtered", ""))
        if candidate.exists():
            return candidate
    return csv_path


def extract_backbone_number(backbone: str) -> Optional[str]:
    match = re.match(r"backbone_(\d+)", backbone)
    if not match:
        return None
    return str(int(match.group(1)))


def load_diffuse_index_map(folder: Path) -> Optional[Dict[str, str]]:
    map_path = folder / "diffused_index_map.json"
    if not map_path.exists():
        print(f"Warning: diffused_index_map.json missing in {folder}", file=sys.stderr)
        return None
    try:
        data = json.loads(map_path.read_text())
        if not isinstance(data, dict):
            print(f"Warning: invalid diffused_index_map in {map_path}", file=sys.stderr)
            return None
        return data
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to read {map_path}: {exc}", file=sys.stderr)
        return None


def load_theozyme_residue_map(pdb_path: Path) -> Dict[str, str]:
    residue_map: Dict[str, str] = {}
    if not pdb_path.exists():
        print(f"Warning: theozyme PDB not found at {pdb_path}", file=sys.stderr)
        return residue_map
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM  "):
            continue
        chain = line[21].strip() or "?"
        resnum = line[22:26].strip()
        resname = line[17:20].strip()
        if not resnum:
            continue
        key = f"{chain}{int(resnum)}"
        if key not in residue_map:
            residue_map[key] = resname
    return residue_map


def load_theozyme_residue_list(pdb_path: Path) -> List[Dict[str, str]]:
    residues: List[Dict[str, str]] = []
    if not pdb_path.exists():
        print(f"Warning: theozyme PDB not found at {pdb_path}", file=sys.stderr)
        return residues
    seen = set()
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM  "):
            continue
        chain = line[21].strip() or "?"
        resnum = line[22:26].strip()
        resname = line[17:20].strip()
        if not resnum:
            continue
        key = f"{chain}{int(resnum)}"
        if key in seen:
            continue
        seen.add(key)
        residues.append(
            {
                "key": key,
                "chain": chain,
                "resnum": str(int(resnum)),
                "resname": resname,
            }
        )
    residues.sort(key=lambda r: (r["chain"], int(r["resnum"])))
    return residues


def normalize_diffuse_map(diff_map: Dict[str, str]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for theo_pos, mapped_pos in diff_map.items():
        theo_norm = normalize_res_id(theo_pos)
        mapped_norm = normalize_res_id(mapped_pos)
        if theo_norm and mapped_norm:
            normalized[theo_norm] = mapped_norm
    return normalized


def build_design_maps(entries: List[Dict]) -> Dict[str, Dict[str, str]]:
    map_cache: Dict[Path, Dict[str, str]] = {}
    design_maps: Dict[str, Dict[str, str]] = {}
    for entry in entries:
        folder = entry["cif_path"].parent
        if folder not in map_cache:
            raw_map = load_diffuse_index_map(folder) or {}
            map_cache[folder] = normalize_diffuse_map(raw_map)
        design_maps[entry["id"]] = map_cache[folder]
    return design_maps


def compute_shannon_entropy(counts: List[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = np.array([count / total for count in counts if count > 0], dtype=float)
    return float(-(probs * np.log2(probs)).sum())


def compute_theozyme_residue_coverage(
    theozyme_residues: List[Dict[str, str]],
    entries: List[Dict],
    site_info: Dict[str, Dict[str, set]],
    design_maps: Dict[str, Dict[str, str]],
) -> List[float]:
    if not theozyme_residues:
        return []
    if not entries:
        print("Warning: no designs found for coverage calculation.", file=sys.stderr)
        return [0.0 for _ in theozyme_residues]
    coverage_rates: List[float] = []
    for residue in theozyme_residues:
        key = residue["key"]
        hits = 0
        for entry in entries:
            design_id = entry["id"]
            mapped_pos = design_maps.get(design_id, {}).get(key)
            pocket_positions = site_info.get(design_id, {}).get("positions", set())
            if mapped_pos and mapped_pos in pocket_positions:
                hits += 1
        coverage_rates.append(hits / len(entries))
    return coverage_rates


def plot_theozyme_residue_coverage(
    theozyme_residues: List[Dict[str, str]],
    unfiltered_rates: List[float],
    filtered_rates: List[float],
    out_dir: Path,
) -> None:
    if not theozyme_residues:
        print("Warning: no theozyme coverage data to plot.", file=sys.stderr)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{res['resnum']} {res['resname']}" for res in theozyme_residues]
    x = np.arange(len(theozyme_residues))
    width = 0.38

    fig_width = max(8.0, 0.45 * len(theozyme_residues))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    ax.bar(x - width / 2, unfiltered_rates, width, label="Unfiltered designs", color="tab:blue")
    ax.bar(x + width / 2, filtered_rates, width, label="Filtered designs", color="red")

    ax.set_ylabel("Fraction of designs")
    ax.set_xlabel("Theozyme residue")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "theozyme_residue_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def categorize_mapped_identity(
    theozyme_key: str,
    design_id: str,
    site_info: Dict[str, Dict[str, set]],
    design_maps: Dict[str, Dict[str, str]],
) -> str:
    mapped_pos = design_maps.get(design_id, {}).get(theozyme_key)
    if not mapped_pos:
        return NAN_CATEGORY
    pocket_positions = site_info.get(design_id, {}).get("positions", set())
    if mapped_pos not in pocket_positions:
        return NAN_CATEGORY
    pos_to_resname = site_info.get(design_id, {}).get("pos_to_resname", {})
    resname = pos_to_resname.get(mapped_pos)
    if not resname:
        return NAN_CATEGORY
    resname = resname.upper()
    return resname if resname in AA_CATEGORIES else NAN_CATEGORY


def plot_theozyme_residue_variability(
    theozyme_residues: List[Dict[str, str]],
    entries: List[Dict],
    site_info: Dict[str, Dict[str, set]],
    design_maps: Dict[str, Dict[str, str]],
    out_dir: Path,
) -> None:
    if not theozyme_residues or not entries:
        print("Warning: no theozyme variability data to plot.", file=sys.stderr)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    categories = AA_CATEGORIES + [NAN_CATEGORY]
    design_ids = [entry["id"] for entry in entries]

    for residue in theozyme_residues:
        counts = {cat: 0 for cat in categories}
        for design_id in design_ids:
            cat = categorize_mapped_identity(residue["key"], design_id, site_info, design_maps)
            counts[cat] += 1
        freq = [counts[cat] / len(entries) for cat in categories]
        entropy = compute_shannon_entropy([counts[cat] for cat in categories])

        colors = ["tab:blue"] * len(categories)
        if residue["resname"] in categories:
            colors[categories.index(residue["resname"])] = "red"

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(categories, freq, color=colors)
        ax.set_ylabel("Frequency")
        ax.set_ylim(0, 1.0)
        ax.set_title(
            f"Theozyme residue {residue['resnum']} {residue['resname']} — entropy = {entropy:.3f}"
        )
        ax.set_xlabel("Residue identity")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        out_path = out_dir / f"theozyme_{residue['resnum']}_{residue['resname']}_entropy.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def compute_jaccard_metrics(
    entries: List[Dict],
    site_info: Dict[str, Dict[str, set]],
    label_map: Dict[str, int],
    coord_map: Dict[str, Tuple[float, float]],
    highlight_ids: set,
    out_dirs: Dict[str, Path],
    theozyme_pdb: Path,
) -> None:
    theozyme_map = load_theozyme_residue_map(theozyme_pdb)
    map_cache: Dict[Path, Optional[Dict[str, str]]] = {}

    rows = []
    for entry in entries:
        structure_id = entry["id"]
        site = site_info.get(structure_id, {"positions": set(), "resnames": set()})
        A_pos = set(site["positions"])
        C_res = set(site["resnames"])

        folder = entry["cif_path"].parent
        if folder not in map_cache:
            map_cache[folder] = load_diffuse_index_map(folder)
        diff_map = map_cache[folder] or {}

        B_pos = set()
        D_res = set()
        for theo_pos, mapped_pos in diff_map.items():
            mapped_norm = normalize_res_id(mapped_pos)
            theo_norm = normalize_res_id(theo_pos)
            if mapped_norm:
                B_pos.add(mapped_norm)
            if theo_norm and theo_norm in theozyme_map:
                D_res.add(theozyme_map[theo_norm])

        inter_pos = A_pos & B_pos
        union_pos = A_pos | B_pos
        j_pos = len(inter_pos) / len(union_pos) if union_pos else 0.0

        inter_chem = C_res & D_res
        union_chem = C_res | D_res
        j_chem = len(inter_chem) / len(union_chem) if union_chem else 0.0

        row = {
            "design_id": structure_id,
            "J_pos": j_pos,
            "J_chem": j_chem,
            "|A|": len(A_pos),
            "|B|": len(B_pos),
            "|A∩B|": len(inter_pos),
            "|A∪B|": len(union_pos),
            "cluster_id": label_map.get(structure_id),
        }
        if structure_id in coord_map:
            row["mds_x"] = coord_map[structure_id][0]
            row["mds_y"] = coord_map[structure_id][1]
        rows.append(row)

    if not rows:
        print("Warning: no Jaccard metrics computed.", file=sys.stderr)
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dirs["tables"] / "jaccard_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    highlight_mask = df["design_id"].isin(highlight_ids)
    ax.scatter(
        df.loc[~highlight_mask, "J_pos"],
        df.loc[~highlight_mask, "J_chem"],
        s=30,
        alpha=0.6,
        color="tab:blue",
        zorder=1,
    )
    ax.scatter(
        df.loc[highlight_mask, "J_pos"],
        df.loc[highlight_mask, "J_chem"],
        s=60,
        alpha=0.9,
        color="red",
        zorder=2,
    )
    for _, row in df.loc[highlight_mask].iterrows():
        backbone_label = extract_backbone_number(row["design_id"].split("|", 1)[0] or "")
        if backbone_label:
            ax.text(row["J_pos"], row["J_chem"], backbone_label, fontsize=7, alpha=0.9)
    ax.set_xlabel("Positional Jaccard (A vs B)")
    ax.set_ylabel("Chemical Jaccard (C vs D)")
    ax.set_title("Jaccard metrics per design")
    fig.savefig(out_dirs["plots"] / "jaccard_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    start = time.monotonic()
    args = parse_args()
    csv_path = Path(args.csv_path)
    root_dir = Path(args.root)
    out_dirs = setup_output_dirs(args.out_dir, Path(args.theozyme_pdb))

    entries = read_structure_table(csv_path, root_dir)
    ids = [entry["id"] for entry in entries]
    cif_paths = [entry["cif_path"] for entry in entries]

    # TM-align installed via Conda; use it directly from PATH.
    tm_bin = "TMalign"

    cache_path = out_dirs["clusters"] / Path(args.tm_cache_path).name
    tm_matrix: Optional[np.ndarray] = None
    if cache_path.exists():
        tm_matrix = load_tm_cache(cache_path, len(cif_paths))

    computed_matrix = tm_matrix is None
    if computed_matrix:
        print(f"Computing TM-scores for {len(cif_paths)} structures...", file=sys.stderr)
        tm_matrix = build_tm_matrix(cif_paths, tm_bin)

    output_prefix = out_dirs["clusters"] / Path(args.output_prefix).name
    df = save_tm_outputs(tm_matrix, ids, str(output_prefix))
    if computed_matrix:
        np.save(str(cache_path), tm_matrix)
        df.to_csv(out_dirs["clusters"] / Path(args.tm_cache_csv_path).name)

    labels = cluster_structures(tm_matrix, args.tm_threshold)
    cluster_ids, counts = np.unique(labels, return_counts=True)
    print(
        f"Identified {len(cluster_ids)} clusters; sizes: "
        + ", ".join(f"{cid}:{cnt}" for cid, cnt in zip(cluster_ids, counts)),
        file=sys.stderr,
    )

    save_cluster_mapping(entries, labels, out_dirs["clusters"] / "cluster_assignments.csv")

    dist_matrix = 1.0 - tm_matrix
    dist_matrix = np.nan_to_num(dist_matrix, nan=1.0)
    np.fill_diagonal(dist_matrix, 0.0)
    plot_mds(dist_matrix, labels, out_dirs["plots"] / "fold_space_pca.png")

    label_map = {entry["id"]: int(label) for entry, label in zip(entries, labels)}
    coord_map: Dict[str, Tuple[float, float]] = {}
    try:
        mds = MDS(
            n_components=2,
            metric="precomputed",
            n_init=4,
            init="random",
            random_state=0,
        )
        coords = mds.fit_transform(dist_matrix)
        for entry, coord in zip(entries, coords):
            coord_map[entry["id"]] = (float(coord[0]), float(coord[1]))
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to compute MDS coords for Jaccard CSV: {exc}", file=sys.stderr)

    active_rows, site_info = compute_active_site_data(entries)
    save_active_site_outputs(active_rows, out_dirs)

    jaccard_csv_path = infer_jaccard_csv_path(csv_path, args.jaccard_csv_path)
    if jaccard_csv_path == csv_path:
        jaccard_entries = entries
        jaccard_site_info = site_info
    else:
        jaccard_entries = read_structure_table(jaccard_csv_path, root_dir)
        _, jaccard_site_info = compute_active_site_data(jaccard_entries)
    compute_jaccard_metrics(
        jaccard_entries,
        jaccard_site_info,
        label_map,
        coord_map,
        set(label_map.keys()),
        out_dirs,
        Path(args.theozyme_pdb),
    )

    theozyme_residues = load_theozyme_residue_list(Path(args.theozyme_pdb))
    filtered_ids = {entry["id"] for entry in entries}
    unfiltered_entries = jaccard_entries
    unfiltered_only_entries = [entry for entry in unfiltered_entries if entry["id"] not in filtered_ids]

    filtered_design_maps = build_design_maps(entries)
    unfiltered_design_maps = build_design_maps(unfiltered_only_entries)

    filtered_rates = compute_theozyme_residue_coverage(
        theozyme_residues,
        entries,
        site_info,
        filtered_design_maps,
    )
    unfiltered_rates = compute_theozyme_residue_coverage(
        theozyme_residues,
        unfiltered_only_entries,
        jaccard_site_info,
        unfiltered_design_maps,
    )
    plot_theozyme_residue_coverage(
        theozyme_residues,
        unfiltered_rates,
        filtered_rates,
        out_dirs["plots"],
    )

    analysis_entries = unfiltered_entries
    analysis_site_info = jaccard_site_info
    analysis_design_maps = build_design_maps(analysis_entries)
    plot_theozyme_residue_variability(
        theozyme_residues,
        analysis_entries,
        analysis_site_info,
        analysis_design_maps,
        out_dirs["plots"] / "variability",
    )

    elapsed = time.monotonic() - start
    print(f"Script completed in {elapsed:.1f} seconds.", file=sys.stderr)

if __name__ == "__main__":
    main()
