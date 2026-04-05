# Enzyme Multinode Usage Guide

This guide explains how to use `enzyme_multinode.py` to design enzyme backbones and sequences using the RFD3 → LigandMPNN → RF3 pipeline.

## Prerequisites

Before running this script, ensure you have:
1. Set up the foundry environment (see main README)
2. Downloaded required checkpoints (RFD3 and RF3)
3. Your theozyme structure file in PDB format
4. **Important**: Renamed your substrate to `L:G` in the PDB file (e.g., if your ligand is named PLB or PLS, rename it to L:G)

## Quick Start

```bash
python enzyme_multinode.py \
  --num_backbones 10 \
  --seqs_per_backbone 5 \
  --theozyme /path/to/your/theozyme.pdb
```

## Understanding the Arguments

### Core Generation Parameters

#### `--num_backbones` (default: 1)
The total number of backbone structures to generate from the diffusion model. This is your outer loop for diversity.
```bash
--num_backbones 100  # Generate 100 different backbones
```

#### `--seqs_per_backbone` (default: 5)
For each generated backbone, how many different sequences to design. The script runs LigandMPNN on each backbone to generate this many sequences.
```bash
--seqs_per_backbone 10  # Design 10 sequences per backbone
```

#### `--diffusion_batch_size` (default: 2)
How many backbones to generate in a single diffusion model forward pass. Larger values are faster but use more GPU memory.
```bash
--diffusion_batch_size 4  # Generate 4 backbones per diffusion run
```

#### `--n_batches` (default: auto-calculated)
How many times to run the complete diffusion pipeline. Combined with `--diffusion_batch_size` to determine total backbones:
```
Total backbones = --n_batches × --diffusion_batch_size
```

Example:
```bash
--n_batches 50 --diffusion_batch_size 4  # 50 × 4 = 200 total backbones
```

**Workflow**: For each backbone generated, the script automatically runs:
1. RFD3 diffusion → generates backbone
2. LigandMPNN → designs `--seqs_per_backbone` sequences on that backbone
3. RF3 → refolds each sequence and evaluates metrics

### Theozyme Configuration

#### `--theozyme` (default: Theozyme_DFT_resid_rfd3.pdb)
Path to your theozyme structure file. This is used as the template for design.
```bash
--theozyme /home/user/my_enzyme/theozyme.pdb
```

#### `--spec` (default: "default")
Which specification preset to use from `theozymes.json`. Each spec defines:
- `length`: Target sequence length range (e.g., "380-420")
- `ligand`: Ligand residue identifier (e.g., "L:G")
- `unindex`: Residues to keep fixed in SPACE during diffusion

#### Using Different Specs

View available specs in `theozymes.json`:
```json
{
  "base": {
    "length": "380-420",
    "ligand": "L:G",
    "unindex": "A82,A104-106,A185,A228-231,A298-299,A301,A345,A371"
  },
  "y301k": {
    "length": "380-420",
    "ligand": "L:G",
    "unindex": "A81-82,A105-109,A185,A228-231,A371"
  }
}
```

Use an existing spec:
```bash
python enzyme_multinode.py --theozyme myenzyme.pdb --spec base
```

#### Adding a New Spec

To experiment with a new theozyme variant, add it to `theozymes.json` without modifying the script:

```json
{
  "my_new_variant_theozyme": {
    "length": "380-420",
    "ligand": "L:G",
    "unindex": "A82,A104-106,A185,A228-231,A298-299,A301,A345,A371"
  }
}
```

Then use it:
```bash
python enzyme_multinode.py --theozyme myenzyme.pdb --spec my_new_variant
```

### Residue Constraints

#### `--unindex` (defined in JSON spec)
Comma-separated list of theozyme residues to **fix in space** (but allow identity changes). These residues won't move during diffusion. Typically, you include all residues involved in catalysis or substrate binding.

Example in `theozymes.json`:
```json
"unindex": "A82,A104-106,A185,A228-231,A298-299,A301,A345,A371"
```

This means residues A82, A104-106 (range), A185, A228-231 (range), etc. stay in place.

#### `--fixed_theozyme_residues` (default: "A82")
Comma-separated list of theozyme residues whose **identity (amino acid type) to preserve** in the final design. This is passed to LigandMPNN to lock these positions.

```bash
--fixed_theozyme_residues A82,A185,A228-231
```

**Key Difference**:
- `unindex` (in JSON): Fixes position in 3D space during diffusion
- `--fixed_theozyme_residues`: Locks amino acid identity during sequence design (LigandMPNN)

You might have both point to similar residues, but they serve different purposes.

### Output and Checkpointing

#### `--output_dir` (default: "./enzyme_output_multi_node_new")
Directory where all results are saved. For each backbone:
- `generated.cif`: Raw diffusion output
- `mpnn_design.cif`: LigandMPNN designed backbone
- `refolded.cif`: RF3 refolded structure
- `metrics.csv`: Sequence metrics and scores

```bash
--output_dir /scratch/my_run/enzyme_results
```

#### `--checkpoint_every` (default: 400)
Save intermediate results every N backbones. Useful for long runs.
```bash
--checkpoint_every 50  # Save progress every 50 backbones
```

### Distributed Computing

#### `--num_nodes` (default: 1)
Number of compute nodes for multi-node runs. Typically auto-detected from launcher environment (RANK, WORLD_SIZE, etc.).

#### `--diffusion_batch_size` with multiple nodes
The script automatically distributes backbone generation across nodes. Each rank processes its subset independently.

```bash
# On 4 GPUs / 2 nodes:
mpirun -np 4 python enzyme_multinode.py \
  --num_backbones 100 \
  --diffusion_batch_size 4 \
  --num_nodes 2
```

### Other Useful Arguments

#### `--seed` (default: None)
Random seed for reproducibility. If provided, will be offset per rank for diversity.
```bash
--seed 42
```

#### `--visualize` (default: False)
Visualize the first backbone/sequence (requires interactive environment).
```bash
--visualize
```

#### `--rf3_enforce_ligand_conformer` (default: True)
Whether RF3 should enforce the ligand conformer from the design. Keep True for enzyme design.

#### `--rf3_ligand_resname` (default: "L:G")
Ligand residue name for RF3 conditioning. Must match your PDB file.

## Complete Example

```bash
python enzyme_multinode.py \
  --num_backbones 100 \
  --seqs_per_backbone 5 \
  --diffusion_batch_size 4 \
  --theozyme ./Theozyme_DFT_resid_rfd3.pdb \
  --spec base \
  --fixed_theozyme_residues A82,A185,A228-231 \
  --output_dir ./enzyme_output \
  --checkpoint_every 50 \
  --seed 42
```

This will:
1. Generate 100 backbones (25 batches × 4 per batch)
2. Design 5 sequences per backbone (500 total sequences)
3. Save results to `./enzyme_output` with checkpoints every 50 backbones
4. Lock residues A82, A185, A228-231 during sequence design
5. Use the "base" specification for diffusion constraints

## Troubleshooting

### Ligand Not Found
If you see warnings about missing ligand residues, ensure:
1. Your substrate is renamed to `L:G` in the PDB file
2. The `--spec` has the correct ligand name in its JSON definition

### Out of Memory
Reduce `--diffusion_batch_size` or `--seqs_per_backbone`.

### Wrong Residues Fixed
Double-check:
- `unindex` in your JSON spec (space constraints during diffusion)
- `--fixed_theozyme_residues` (identity constraints during LigandMPNN)

## Output Files

For each backbone `backbone_XXXX_rY/`:
```
backbone_0000_r0/
├── generated.cif              # RFD3 output
├── diffused_index_map.json    # Mapping from theozyme → backbone residues
├── seq_0000/
│   ├── mpnn_design.cif        # LigandMPNN design
│   └── refolded.cif           # RF3 refolded result
├── seq_0001/
│   ...
└── metrics.csv                # Sequence metrics (pLDDT, PAE, RMSD, etc.)

run_summary.csv                 # All metrics across all backbones
best_per_backbone.csv           # Best sequence per backbone
best_per_backbone_filtered.csv  # High-quality designs (RMSD ≤2Å, Theozyme_RMSD ≤1.5Å)
```

## Next Steps

1. Analyze results in `run_summary.csv`
2. Filter by `best_per_backbone_filtered.csv` for high-confidence designs
3. Validate top candidates experimentally or with additional docking
