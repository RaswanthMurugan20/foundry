import sys;
sys.path.insert(0, '/home/jbutch/Projects/HT25/af3/rfd3-release/lib/atomworks/src')
sys.path.append('/home/raswanth/foundry/models/rfd3/src')
sys.path.append('/home/jbutch/Projects/HT25/af3/rfd3-release/src')

import argparse
import csv
import gc
import math
import random
import json
from collections import Counter
import torch
from pathlib import Path
import sys

import numpy as np
from atomworks.io.utils.visualize import view
from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
from rfd3.inference.input_parsing import DesignInputSpecification
from rf3.inference_engines.rf3 import RF3InferenceEngine
from rf3.utils.inference import InferenceInput
from mpnn.inference_engines.mpnn import MPNNInferenceEngine
from biotite.structure import get_residue_starts, rmsd, superimpose
from biotite.sequence import ProteinSequence
from atomworks.constants import PROTEIN_BACKBONE_ATOM_NAMES
from atomworks.io.utils.io_utils import load_any, to_cif_file
from foundry.utils.alignment import weighted_rigid_align


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def sequence_from_atom_array(atom_array) -> str:
    res_starts = get_residue_starts(atom_array)
    return ''.join(
        ProteinSequence.convert_letter_3to1(res_name)
        for res_name in atom_array.res_name[res_starts] if res_name != 'L:G'
    )


def compute_backbone_rmsd(aa_generated, aa_refolded) -> float:
    bb_generated = aa_generated[np.isin(aa_generated.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)]
    bb_refolded = aa_refolded[np.isin(aa_refolded.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)]
    bb_refolded_fitted, _ = superimpose(bb_generated, bb_refolded)
    return float(rmsd(bb_generated, bb_refolded_fitted))


def compute_theozyme_rmsd(
    theozyme,
    predicted,
    diffused_index_map: dict | None,
    min_atoms: int = 3,
) -> float:
    """
    Align theozyme atoms to their mapped locations in the predicted structure using
    weighted_rigid_align, then compute RMSD on the matched atoms.

    Mapping is taken from diffused_index_map: keys like 'A81' -> values like 'B123'.
    """
    if not diffused_index_map:
        return float("nan")

    def parse_loc(loc: str):
        loc = str(loc)
        chain = "".join([c for c in loc if c.isalpha()]) or None
        resid_str = "".join([c for c in loc if (c.isdigit() or c == "-")])
        resid = int(resid_str) if resid_str else None
        return chain, resid

    theozyme = theozyme[0] if hasattr(theozyme, "stack_depth") and theozyme.stack_depth() else theozyme
    predicted = predicted[0] if hasattr(predicted, "stack_depth") and predicted.stack_depth() else predicted

    align_src = []
    align_dst = []
    score_src = []
    score_dst = []
    for src_token, dst_token in diffused_index_map.items():
        src_chain, src_res = parse_loc(src_token)
        dst_chain, dst_res = parse_loc(dst_token)
        if None in (src_chain, src_res, dst_chain, dst_res):
            continue

        src_atoms = theozyme[(theozyme.chain_id == src_chain) & (theozyme.res_id == src_res)]
        dst_atoms = predicted[(predicted.chain_id == dst_chain) & (predicted.res_id == dst_res)]

        if len(src_atoms) == 0 or len(dst_atoms) == 0:
            continue

        # Normalize atom names and drop hydrogens
        src_names = np.char.strip(src_atoms.atom_name.astype(str))
        dst_names = np.char.strip(dst_atoms.atom_name.astype(str))
        src_is_h = np.char.startswith(np.char.upper(src_names), "H")
        dst_is_h = np.char.startswith(np.char.upper(dst_names), "H")
        src_names = src_names[~src_is_h]
        dst_names = dst_names[~dst_is_h]
        src_coords_raw = src_atoms.coord[~src_is_h]
        dst_coords_raw = dst_atoms.coord[~dst_is_h]

        shared_atoms = np.intersect1d(src_names, dst_names)
        for name in shared_atoms:
            src_coord = src_coords_raw[src_names == name][0]
            dst_coord = dst_coords_raw[dst_names == name][0]
            if np.any(np.isnan(src_coord)) or np.any(np.isnan(dst_coord)):
                continue
            # Backbone atoms for alignment
            if name in ("N", "CA", "C"):
                align_src.append(src_coord)
                align_dst.append(dst_coord)
            # All heavy atoms for scoring
            score_src.append(src_coord)
            score_dst.append(dst_coord)

    if len(align_src) < 3 or len(score_src) == 0:
        return float("nan")

    align_src_arr = np.array(align_src, dtype=np.float32)
    # Check non-collinearity of alignment atoms
    centered = align_src_arr - align_src_arr.mean(axis=0, keepdims=True)
    if np.linalg.matrix_rank(centered) < 2:
        return float("nan")

    # Combine alignment + scoring coords; weights ensure alignment uses backbone only
    src_combined = np.vstack([align_src, score_src]).astype(np.float32)
    dst_combined = np.vstack([align_dst, score_dst]).astype(np.float32)

    src_t = torch.tensor(src_combined, dtype=torch.float32).unsqueeze(0)  # [1, L, 3]
    dst_t = torch.tensor(dst_combined, dtype=torch.float32).unsqueeze(0)  # [1, L, 3]

    exists_mask = torch.ones(dst_t.shape[-2], dtype=torch.bool)
    weights = torch.zeros_like(dst_t[..., 0])
    weights[..., : len(align_src)] = 1.0  # only backbone atoms drive alignment

    aligned_dst = weighted_rigid_align(src_t, dst_t, X_exists_L=exists_mask, w_L=weights)
    # Score RMSD over heavy atoms (the scoring part)
    aligned_dst_score = aligned_dst[..., len(align_src) :, :]
    src_score = src_t[..., len(align_src) :, :]
    rmsd_val = torch.sqrt(torch.mean((aligned_dst_score - src_score) ** 2)).item()
    return float(rmsd_val)


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clear_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def save_backbone_summary_and_plots(global_rows: list[dict], output_root: Path) -> None:
    if not global_rows:
        return

    best_by_backbone: dict[str, dict] = {}
    for row in global_rows:
        backbone_id = row["backbone_id"]
        rmsd_value = row.get("RMSD")
        if rmsd_value is None:
            continue
        if backbone_id not in best_by_backbone or rmsd_value < best_by_backbone[backbone_id]["RMSD"]:
            best_by_backbone[backbone_id] = row

    best_rows = [best_by_backbone[k] for k in sorted(best_by_backbone.keys())]
    if not best_rows:
        return

    summary_path = output_root / "best_per_backbone.csv"
    write_metrics_csv(summary_path, best_rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Plots] Skipping plot generation (matplotlib unavailable): {exc}")
        return

    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("overall_pLDDT", "overall_pLDDT"),
        ("PAE", "PAE"),
        ("overall_PAE", "overall_PAE"),
        ("mean_PAE", "mean_PAE"),
        ("overall_PDE", "overall_PDE"),
        ("pTM", "pTM"),
        ("ipTM", "ipTM"),
        ("ranking_score", "ranking_score"),
        ("RMSD", "RMSD"),
        ("RMSE", "RMSE"),
        ("theozyme_RMSD", "theozyme_RMSD"),
    ]

    x_vals = list(range(1, len(best_rows) + 1))

    for key, label in metrics:
        y_vals = []
        for row in best_rows:
            val = row.get(key)
            y_vals.append(float(val) if val is not None else np.nan)

        finite = [v for v in y_vals if not np.isnan(v)]
        mean_val = np.nanmean(y_vals) if y_vals else np.nan
        std_val = np.nanstd(y_vals) if y_vals else np.nan
        if finite:
            mode_val = Counter(finite).most_common(1)[0][0]
        else:
            mode_val = np.nan

        def _fmt(x):
            return f"{x:.3f}" if not np.isnan(x) else "nan"

        plt.figure(figsize=(10, 5))
        plt.scatter(x_vals, y_vals, marker="o")
        plt.title(f"Best-per-backbone {label}")
        plt.xlabel("Backbone #")
        plt.ylabel(label)
        plt.xticks(x_vals)
        stats_label = f"mean={_fmt(mean_val)}, std={_fmt(std_val)}, mode={_fmt(mode_val)}"
        plt.legend([stats_label], loc="upper right", frameon=True)
        plt.tight_layout()
        out_path = plots_dir / f"backbone_best_{key}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="RFD3 -> LigandMPNN -> RF3 pipeline")
    parser.add_argument("--num_backbones", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seqs_per_backbone", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="/home/raswanth/foundry/enzyme_output")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--diffusion_batch_size", type=int, default=2)
    parser.add_argument("--n_batches", type=int, default=argparse.SUPPRESS)
    args = parser.parse_args()

    num_backbones = getattr(args, "num_backbones", None)
    n_batches = getattr(args, "n_batches", None)
    if num_backbones is None and n_batches is None:
        num_backbones = 1
    if n_batches is None:
        n_batches = math.ceil(num_backbones / args.diffusion_batch_size)
        max_backbones = num_backbones
    else:
        max_backbones = num_backbones
    expected_backbones = n_batches * args.diffusion_batch_size
    total_backbones = expected_backbones if max_backbones is None else min(max_backbones, expected_backbones)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    spec = DesignInputSpecification(
        input='/home/raswanth/foundry/examples/Theozyme_DFT_resid_rfd3.pdb',
        length='380-420',
        ligand='L:G',
        unindex='A81-82,A105-109,A185,A228-231,A371',
    )
    theozyme_atom_array = load_any(spec.input)

    conf = RFD3InferenceConfig(
        ckpt_path='/home/raswanth/.foundry/checkpoints/rfd3_latest.ckpt',
        diffusion_batch_size=args.diffusion_batch_size,
        dump_trajectories=True,
        # devices_per_node=4,
        # low_memory_mode=True,
    )
    rfd3_model = RFD3InferenceEngine(**conf)

    rf3_engine = RF3InferenceEngine(
        ckpt_path='/home/raswanth/.foundry/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt',
        verbose=False,
        # devices_per_node=4
    )

    global_rows = []
    backbone_index = 0
    print(
        f"[RFD3] Generating {expected_backbones} backbones "
        f"(n_batches={n_batches}, diffusion_batch_size={args.diffusion_batch_size})"
    )
    set_seed(args.seed)
    outputs = rfd3_model.run(inputs=spec, n_batches=n_batches)
    for _, data in outputs.items():
        for item in data:
            if max_backbones is not None and backbone_index >= max_backbones:
                break

            b = backbone_index
            backbone_index += 1
            print(f"[Backbone {b + 1}/{total_backbones}] Processing backbone")

            backbone_atom_array = item.atom_array
            backbone_metadata = getattr(item, "metadata", {}) or {}
            backbone_id = f"backbone_{b:04d}"
            backbone_dir = output_root / backbone_id
            backbone_dir.mkdir(parents=True, exist_ok=True)
            to_cif_file(backbone_atom_array, str(backbone_dir / "generated.cif"))
            if backbone_metadata.get("diffused_index_map") is not None:
                with open(backbone_dir / "diffused_index_map.json", "w") as f:
                    json.dump(backbone_metadata["diffused_index_map"], f, indent=2)

            set_seed(None if args.seed is None else args.seed + 1000 * b)
            mpnn_engine_config = {
                "model_type": "ligand_mpnn",
                "is_legacy_weights": True,
                "out_directory": str(backbone_dir),
                "write_structures": True,
                "write_fasta": True,
            }
            input_configs = [
                {
                    "batch_size": args.seqs_per_backbone,
                    "remove_waters": True,
                }
            ]

            mpnn_model = MPNNInferenceEngine(**mpnn_engine_config)
            mpnn_outputs = mpnn_model.run(
                input_dicts=input_configs,
                atom_arrays=[backbone_atom_array],
            )

            backbone_rows = []
            best_row = None

            for s, mpnn_item in enumerate(mpnn_outputs):
                print(
                    f"[Backbone {b + 1}/{total_backbones}] "
                    f"Folding seq {s + 1}/{len(mpnn_outputs)}"
                )
                set_seed(None if args.seed is None else args.seed + 1000 * b + s)

                seq_id = f"seq_{s:04d}"
                seq_dir = backbone_dir / seq_id
                seq_dir.mkdir(parents=True, exist_ok=True)

                mpnn_atom_array = mpnn_item.atom_array
                sequence = sequence_from_atom_array(mpnn_atom_array)
                to_cif_file(mpnn_atom_array, str(seq_dir / "mpnn_design.cif"))

                example_id = f"{backbone_id}_{seq_id}"
                input_structure = InferenceInput.from_atom_array(
                    mpnn_atom_array,
                    example_id=example_id,
                )
                rf3_outputs = rf3_engine.run(inputs=input_structure)
                rf3_output = rf3_outputs[example_id][0]

                model_path = seq_dir / "refolded.cif"
                to_cif_file(rf3_output.atom_array, str(model_path))

                summary = rf3_output.summary_confidences
                conf = rf3_output.confidences
                mean_pae = None
                if conf and conf.get("pae") is not None:
                    try:
                        mean_pae = float(np.mean(np.array(conf["pae"], dtype=float)))
                    except Exception:
                        mean_pae = None
                if mean_pae is None:
                    mean_pae = summary.get("overall_pae")
                rmsd_value = compute_backbone_rmsd(
                    backbone_atom_array,
                    rf3_output.atom_array,
                )
                theozyme_rmsd = compute_theozyme_rmsd(
                    theozyme_atom_array,
                    rf3_output.atom_array,
                    backbone_metadata.get("diffused_index_map"),
                )
                row = {
                    "backbone_id": backbone_id,
                    "seq_id": seq_id,
                    "sequence": sequence,
                    "model_path": str(model_path),
                    "overall_pLDDT": summary.get("overall_plddt"),
                    "PAE": mean_pae,
                    "overall_PAE": summary.get("overall_pae"),
                    "mean_PAE": mean_pae,
                    "overall_PDE": summary.get("overall_pde"),
                    "pTM": summary.get("ptm"),
                    "ipTM": summary.get("iptm"),
                    "ranking_score": summary.get("ranking_score"),
                    "RMSD": rmsd_value,
                    "RMSE": rmsd_value,
                    "theozyme_RMSD": theozyme_rmsd,
                    "diffused_index_map": json.dumps(backbone_metadata.get("diffused_index_map"))
                    if backbone_metadata.get("diffused_index_map") is not None
                    else None,
                }

                backbone_rows.append(row)
                global_rows.append(row)
                if best_row is None or rmsd_value < best_row["RMSD"]:
                    best_row = row

                if args.visualize and b == 0 and s == 0:
                    view(rf3_output.atom_array)

                del rf3_outputs, rf3_output, input_structure
                clear_memory()

            write_metrics_csv(backbone_dir / "metrics.csv", backbone_rows)

            if best_row:
                print(
                    f"[Backbone {b:04d}] best RMSD {best_row['RMSD']:.2f} A | "
                    f"{best_row['seq_id']} | pLDDT {best_row['overall_pLDDT']:.3f} | "
                    f"rank {best_row['ranking_score']:.3f}"
                )

            del mpnn_outputs, mpnn_model, backbone_atom_array
            clear_memory()

        if max_backbones is not None and backbone_index >= max_backbones:
            break

    del outputs
    clear_memory()

    write_metrics_csv(output_root / "run_summary.csv", global_rows)
    save_backbone_summary_and_plots(global_rows, output_root)


if __name__ == "__main__":
    main()
