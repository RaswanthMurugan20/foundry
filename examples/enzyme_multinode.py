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
import socket
import os
import torch
import torch.distributed as dist
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

PROTEIN_RES_NAMES = {
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
}


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

def parse_loc(loc: str):
    loc = str(loc)
    chain = "".join([c for c in loc if c.isalpha()]) or None
    resid_str = "".join([c for c in loc if (c.isdigit() or c == "-")])
    resid = int(resid_str) if resid_str else None
    return chain, resid


def parse_residue_tokens(value: str | None) -> list[str]:
    if value is None:
        return []
    tokens = str(value).replace(",", " ").split()
    return [t for t in tokens if t]


def build_ligand_selection_queries(
    atom_array,
    ligand_res_name: str | None = "L:G",
) -> list[str]:
    """
    Build AtomSelection queries for ligand residues in the input atom array.

    The selections are suitable for RF3 `template_selection` and
    `ground_truth_conformer_selection`, using `CHAIN/RES_NAME/RES_ID` syntax.
    """
    if hasattr(atom_array, "stack_depth") and atom_array.stack_depth():
        atom_array = atom_array[0]

    res_starts = get_residue_starts(atom_array)
    queries: list[str] = []
    seen = set()

    for idx in res_starts:
        res_name = str(atom_array.res_name[idx]).strip()
        chain_id = str(atom_array.chain_id[idx]).strip()
        res_id = str(atom_array.res_id[idx]).strip()

        if not chain_id or not res_id:
            continue

        if ligand_res_name is not None and res_name != ligand_res_name:
            continue

        is_ligand_like = res_name.startswith("L:") or res_name not in PROTEIN_RES_NAMES
        if not is_ligand_like:
            continue

        query = f"{chain_id}/{res_name}/{res_id}"
        if query not in seen:
            seen.add(query)
            queries.append(query)

    return queries


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

    theozyme = theozyme[0] if hasattr(theozyme, "stack_depth") and theozyme.stack_depth() else theozyme
    predicted = predicted[0] if hasattr(predicted, "stack_depth") and predicted.stack_depth() else predicted

    all_src_coords = []
    all_dst_coords = []
    all_is_backbone = []
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
            all_src_coords.append(src_coord)
            all_dst_coords.append(dst_coord)
            all_is_backbone.append(name in ("N", "CA", "C"))

    if len(all_src_coords) == 0 or sum(all_is_backbone) < 3:
        return float("nan")

    src_arr = np.array(all_src_coords, dtype=np.float32)
    dst_arr = np.array(all_dst_coords, dtype=np.float32)

    is_bb = np.array(all_is_backbone, dtype=bool)

    # Check non-collinearity of alignment atoms
    align_src_arr = src_arr[is_bb]
    centered = align_src_arr - align_src_arr.mean(axis=0, keepdims=True)
    if np.linalg.matrix_rank(centered) < 2:
        return float("nan")

    src_t = torch.tensor(src_arr, dtype=torch.float32).unsqueeze(0)  # [1, L, 3]
    dst_t = torch.tensor(dst_arr, dtype=torch.float32).unsqueeze(0)  # [1, L, 3]

    exists_mask = torch.ones(dst_t.shape[-2], dtype=torch.bool)
    weights = torch.zeros_like(dst_t[..., 0])
    weights[..., is_bb] = 1.0  # only backbone atoms drive alignment

    aligned_dst = weighted_rigid_align(src_t, dst_t, X_exists_L=exists_mask, w_L=weights)
    rmsd_val = torch.sqrt(torch.mean((aligned_dst - src_t) ** 2)).item()
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


def get_rank_world(fabric=None) -> tuple[int, int]:
    """
    Resolve (rank, world_size) from torch.distributed/Fabric/env, defaulting to (0,1).
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if fabric is not None:
        rank = getattr(fabric, "global_rank", rank)
        world = getattr(fabric, "world_size", world)
    return rank, world


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
    filtered_rows = []
    for row in best_rows:
        rmsd_value = row.get("RMSD")
        theozyme_rmsd = row.get("theozyme_RMSD")
        if rmsd_value is None or theozyme_rmsd is None:
            continue
        try:
            rmsd_value = float(rmsd_value)
            theozyme_rmsd = float(theozyme_rmsd)
        except Exception:
            continue
        if math.isnan(rmsd_value) or math.isnan(theozyme_rmsd):
            continue
        if rmsd_value <= 2.0 and theozyme_rmsd <= 1.5:
            filtered_rows.append(row)
    filtered_path = output_root / "best_per_backbone_filtered.csv"
    write_metrics_csv(filtered_path, filtered_rows)

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
    parser.add_argument("--output_dir", type=str, default="/home/raswanth/foundry/enzyme_output_multi_node_new")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--theozyme", type=str, default="/home/raswanth/foundry/examples/Theozyme_DFT_resid_rfd3.pdb")
    parser.add_argument("--diffusion_batch_size", type=int, default=2)
    parser.add_argument("--n_batches", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes participating in distributed run.")
    parser.add_argument("--checkpoint_every",type=int,default=400, help="If set, write summary CSV/plots every N backbones on rank 0.",)
    parser.add_argument(
        "--fixed_theozyme_residues",
        type=str,
        default="A82,A301",
        help="Comma or space separated theozyme residue IDs to fix in LigandMPNN (e.g. 'A82,B301').",
    )
    parser.add_argument(
        "--rf3_enforce_ligand_conformer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If enabled, pass ligand residue selections to RF3 via both "
            "template_selection and ground_truth_conformer_selection."
        ),
    )
    parser.add_argument(
        "--rf3_ligand_resname",
        type=str,
        default="L:G",
        help=(
            "Ligand residue name to enforce in RF3 conditioning (e.g. 'L:G'). "
            "If not found, falls back to auto-detecting non-protein residues."
        ),
    )
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

    rank_env = os.environ.get("RANK")
    local_rank_env = os.environ.get("LOCAL_RANK")
    world_env = os.environ.get("WORLD_SIZE")
    local_world_env = os.environ.get("LOCAL_WORLD_SIZE")
    rank_env = int(rank_env) if rank_env is not None else 0
    local_rank_env = int(local_rank_env) if local_rank_env is not None else 0
    world_env = int(world_env) if world_env is not None else 1
    local_world_env = int(local_world_env) if local_world_env is not None else 0
    # Auto-set devices_per_node from launcher env; prefer LOCAL_WORLD_SIZE (per-node procs) over total WORLD_SIZE.
    devices_per_node = local_world_env or torch.cuda.device_count() or 1
    args.num_nodes = max(1, math.ceil(world_env / devices_per_node))
    # Pin this process to its local GPU to avoid all ranks piling onto cuda:0.
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_rank_env % devices_per_node)
        except Exception:
            pass
    # Initialize process group early so downstream engines see the distributed context.
    if world_env > 1 and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank_env, world_size=world_env)
    # Log per-rank device binding so we can verify all GPUs/nodes are used.
    dist_initialized = dist.is_available() and dist.is_initialized()
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else None
    current_device_name = torch.cuda.get_device_name(current_device) if torch.cuda.is_available() else None
    print(
        f"[Startup] host={socket.gethostname()} "
        f"rank={rank_env}/{world_env} "
        f"local_rank={local_rank_env}/{local_world_env} "
        f"devices_per_node={devices_per_node} "
        f"visible_cuda={torch.cuda.device_count()} "
        f"current_device={current_device} ({current_device_name}) "
        f"dist_initialized={dist_initialized}"
    )

    rank, world_size = get_rank_world()
    output_root = Path(args.output_dir)
    if world_size > 1:
        output_root = output_root / f"rank_{rank}"
    output_root.mkdir(parents=True, exist_ok=True)

    total_backbones_per_rank = math.ceil(total_backbones / world_size)
    rank_start = rank * total_backbones_per_rank
    rank_end = min(total_backbones, rank_start + total_backbones_per_rank)
    local_target_backbones = max(rank_end - rank_start, 0)
    local_n_batches = math.ceil(local_target_backbones / args.diffusion_batch_size) if local_target_backbones else 0
    rank_max_backbones = rank_end

    # spec = DesignInputSpecification(
    #     input=args.theozyme,
    #     length='380-420',
    #     ligand='L:G',
    #     unindex='A81-82,A105-109,A185,A228-231,A371',
    # )

    spec = DesignInputSpecification(
        input=args.theozyme,
        length='380-420',
        ligand='L:G',
        unindex='A82,A104-106,A185,A228-231,A298-299,A301,A345,A371',
    )


    theozyme_atom_array = load_any(spec.input)

    conf = RFD3InferenceConfig(
        ckpt_path='/home/raswanth/.foundry/checkpoints/rfd3_latest.ckpt',
        diffusion_batch_size=args.diffusion_batch_size,
        dump_trajectories=True,
        devices_per_node=devices_per_node,
        num_nodes=args.num_nodes,
        # low_memory_mode=True,
    )
    rfd3_model = RFD3InferenceEngine(**conf)

    rf3_engine = RF3InferenceEngine(
        ckpt_path='/home/raswanth/.foundry/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt',
        verbose=False,
        devices_per_node=devices_per_node,
        num_nodes=args.num_nodes,
    )

    global_rows = []
    backbone_index = rank_start
    local_expected_backbones = local_n_batches * args.diffusion_batch_size
    print(
        f"[RFD3] Rank {rank} generating {local_expected_backbones} backbones "
        f"(target {local_target_backbones}, n_batches={local_n_batches}, diffusion_batch_size={args.diffusion_batch_size}) "
        f"of total {total_backbones}"
    )
    set_seed(None if args.seed is None else args.seed + rank)
    outputs = {}
    if local_n_batches > 0:
        outputs = rfd3_model.run(inputs=spec, n_batches=local_n_batches)
    else:
        print(f"[RFD3] Rank {rank} has no assigned backbones (global total {total_backbones}).")
    fabric = getattr(rfd3_model, "trainer", None)
    fabric = getattr(fabric, "fabric", None)
    rank, world_size = get_rank_world(fabric)
    # Manual sharding: each rank only processes its own generated backbones.
    shard_world_size = 1
    for _, data in outputs.items():
        for item in data:
            if rank_max_backbones is not None and backbone_index >= rank_max_backbones:
                break

            b = backbone_index
            backbone_index += 1
            print(f"[Backbone {b + 1}/{total_backbones}] Processing backbone")

            backbone_atom_array = item.atom_array
            backbone_metadata = getattr(item, "metadata", {}) or {}
            backbone_id = f"backbone_{b:04d}_r{rank}"
            backbone_dir = output_root / backbone_id
            backbone_dir.mkdir(parents=True, exist_ok=True)
            to_cif_file(backbone_atom_array, str(backbone_dir / "generated.cif"))
            if backbone_metadata.get("diffused_index_map") is not None:
                with open(backbone_dir / "diffused_index_map.json", "w") as f:
                    json.dump(backbone_metadata["diffused_index_map"], f, indent=2)

            fixed_residues = []
            fixed_residue_set = set()
            diffused_index_map = backbone_metadata.get("diffused_index_map") or {}
            fixed_tokens = parse_residue_tokens(args.fixed_theozyme_residues)
            if fixed_tokens and diffused_index_map:
                theozyme_src_array = (
                    theozyme_atom_array[0]
                    if hasattr(theozyme_atom_array, "stack_depth") and theozyme_atom_array.stack_depth()
                    else theozyme_atom_array
                )
                for token in fixed_tokens:
                    mapped_loc = diffused_index_map.get(token)
                    if not mapped_loc:
                        continue
                    src_chain, src_res = parse_loc(token)
                    dst_chain, dst_res = parse_loc(mapped_loc)
                    if None in (src_chain, src_res, dst_chain, dst_res):
                        continue
                    theozyme_src = theozyme_src_array[
                        (theozyme_src_array.chain_id == src_chain)
                        & (theozyme_src_array.res_id == src_res)
                    ]
                    if len(theozyme_src) == 0:
                        continue
                    src_res_name = str(theozyme_src.res_name[0])
                    dst_mask = (
                        (backbone_atom_array.chain_id == dst_chain)
                        & (backbone_atom_array.res_id == dst_res)
                    )
                    if np.any(dst_mask):
                        backbone_atom_array.set_annotation(
                            "res_name",
                            np.where(dst_mask, src_res_name, backbone_atom_array.res_name),
                        )
                        dst_token = f"{dst_chain}{dst_res}"
                        if dst_token not in fixed_residue_set:
                            fixed_residue_set.add(dst_token)
                            fixed_residues.append(dst_token)

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
                    "fixed_residues": fixed_residues,
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

                prune_threshold = 2.0

                seq_id = f"seq_{s:04d}"
                seq_dir = backbone_dir / seq_id
                seq_dir.mkdir(parents=True, exist_ok=True)

                mpnn_atom_array = mpnn_item.atom_array
                sequence = sequence_from_atom_array(mpnn_atom_array)

                mpnn_cif_path = seq_dir / "mpnn_design.cif"

                example_id = f"{backbone_id}_{seq_id}"
                rf3_template_selection = None
                rf3_conformer_selection = None
                if args.rf3_enforce_ligand_conformer:
                    ligand_queries = build_ligand_selection_queries(
                        mpnn_atom_array,
                        ligand_res_name=args.rf3_ligand_resname,
                    )
                    if not ligand_queries and args.rf3_ligand_resname is not None:
                        ligand_queries = build_ligand_selection_queries(
                            mpnn_atom_array,
                            ligand_res_name=None,
                        )
                    if ligand_queries:
                        rf3_template_selection = ligand_queries
                        rf3_conformer_selection = ligand_queries
                        print(
                            f"[RF3] {backbone_id}/{seq_id} enforcing ligand conditioning with selections: "
                            + ", ".join(ligand_queries)
                        )
                    else:
                        print(
                            f"[RF3] {backbone_id}/{seq_id} no ligand residue found for conditioning; "
                            "running RF3 without ligand conformer/template selection."
                        )

                input_structure = InferenceInput.from_atom_array(
                    mpnn_atom_array,
                    example_id=example_id,
                    template_selection=rf3_template_selection,
                    ground_truth_conformer_selection=rf3_conformer_selection,
                )
                rf3_outputs = rf3_engine.run(inputs=input_structure)
                rf3_output = rf3_outputs[example_id][0]

                model_path = seq_dir / "refolded.cif"

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

                bb_gen_count = np.isin(backbone_atom_array.atom_name, PROTEIN_BACKBONE_ATOM_NAMES).sum()
                bb_ref_count = np.isin(rf3_output.atom_array.atom_name, PROTEIN_BACKBONE_ATOM_NAMES).sum()
                print(
                    f"[Rank {rank}] Backbone {backbone_id}/{seq_id} backbone atoms: "
                    f"RFD3={bb_gen_count}, RF3={bb_ref_count}"
                )

                rmsd_value = compute_backbone_rmsd(
                    backbone_atom_array,
                    rf3_output.atom_array,
                )
                theozyme_rmsd = compute_theozyme_rmsd(
                    theozyme_atom_array,
                    rf3_output.atom_array,
                    backbone_metadata.get("diffused_index_map"),
                )

                should_prune = (
                    rmsd_value is not None
                    and rmsd_value > prune_threshold
                    and theozyme_rmsd is not None
                    and not math.isnan(theozyme_rmsd)
                    and theozyme_rmsd > prune_threshold
                )

                if not should_prune:
                    to_cif_file(mpnn_atom_array, str(mpnn_cif_path))
                    to_cif_file(rf3_output.atom_array, str(model_path))
                    model_path_value = str(model_path)
                else:
                    model_path_value = None

                row = {
                    "backbone_id": backbone_id,
                    "seq_id": seq_id,
                    "sequence": sequence,
                    "model_path": model_path_value,
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
                best_theozyme = None if best_row is None else best_row.get("theozyme_RMSD")
                if best_row is None:
                    best_row = row
                elif theozyme_rmsd is not None and not math.isnan(theozyme_rmsd):
                    if best_theozyme is None or (not math.isnan(best_theozyme) and theozyme_rmsd < best_theozyme) or math.isnan(best_theozyme):
                        best_row = row

                if args.visualize and b == 0 and s == 0:
                    view(rf3_output.atom_array)

                del rf3_outputs, rf3_output, input_structure
                clear_memory()

            write_metrics_csv(backbone_dir / "metrics.csv", backbone_rows)

            if best_row:
                print(
                    f"[Backbone {b:04d}] best RMSD {best_row['RMSD']:.2f} A | "
                    f"[Backbone {b:04d}] best Theozyme_RMSD {best_row['theozyme_RMSD']:.2f} A | "
                    f"{best_row['seq_id']} | pLDDT {best_row['overall_pLDDT']:.3f} | "
                    f"rank {best_row['ranking_score']:.3f}"
                )

            del mpnn_outputs, mpnn_model, backbone_atom_array
            clear_memory()

            # Periodic summary checkpoint
            if args.checkpoint_every and rank == 0 and (backbone_index % args.checkpoint_every == 0):
                write_metrics_csv(output_root / "run_summary.csv", global_rows)
                save_backbone_summary_and_plots(global_rows, output_root)

        if rank_max_backbones is not None and backbone_index >= rank_max_backbones:
            break

    del outputs
    clear_memory()

    combined_rows = global_rows
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        gathered_rows = [None] * world_size
        dist.all_gather_object(gathered_rows, global_rows)
        combined_rows = [row for sub in gathered_rows if sub for row in sub]

    if rank == 0:
        write_metrics_csv(output_root / "run_summary.csv", combined_rows)
        save_backbone_summary_and_plots(combined_rows, output_root)


if __name__ == "__main__":
    main()
