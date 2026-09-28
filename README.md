# SBPT-Net

Code and steps for the superline-based point-token experiments in the paper.

## Environment

```powershell
conda env create -f environment.yml
conda activate sbptnet
```

PointNeXt uses the bundled CUDA extension. With CUDA Toolkit 12.1 and a compatible C++ compiler installed, build it before running that backbone:

```powershell
Push-Location reference_code/STS2R_formal_baselines_100ep_w8/vendor/openpoints/cpp/pointnet2_batch
python setup.py install
Pop-Location
```

## Data preparation

Download `synthetic_data.zip`, `real_test_data.zip`, `generation_assets.zip`, and `canonical_virtual_source_templates.zip` from [STS2R on Zenodo](https://doi.org/10.5281/zenodo.19528228). Clone the [STS2R generation code](https://github.com/MoXiang97/STS2R-code) next to this repository and check out commit `63ff9c2ca4c63df32e2da7cd61a7dc572d4ce199`.

Place the archive contents in the STS2R-code checkout:

| Zenodo contents | STS2R-code destination |
| --- | --- |
| `canonical_virtual_source_templates/*` | `assets/Base_Perfect_scan_60CAD/` |
| `generation_assets/shoe_upper_materials/*` | `assets/Textures/` |
| `generation_assets/decorative_patterns/*` | `assets/Textures_logo/` |
| `generation_assets/sole_region_materials/*` | `assets/Textures_sole/` |
| `synthetic_data/*` | `outputs/Generated_ablation_data/04_STS2R/` |
| `real_test_data/prototype_A/*` | `assets/Real_Data/Real_ShoeA/` |
| `real_test_data/prototype_B/*` | `assets/Real_Data/Real_ShoeB/` |
| `real_test_data/prototype_C/*` | `assets/Real_Data/Real_ShoeC/` |

### Synthetic data used in this study

The 1,000 synthetic point clouds used for model training and validation are generated as `D04_AppGeoPhys1000` and are split into 800 training samples and 200 validation samples.

The `04_STS2R` subset is used to calibrate the local color contrast threshold and is not included in the 800/200 model training and validation split.

From the STS2R-code checkout, generate the D04 synthetic samples and LCC candidate records:

```powershell
python scripts/01_generate_diagnostic_variants.py --overwrite
python scripts/02_generate_lcc_candidates.py --lcc-k 256 --target-synthetic-recall 0.98 --calibration-subset 04_STS2R --subsets D04_AppGeoPhys1000 shoe_a shoe_b shoe_c --overwrite
```

From this repository, convert the LCC records to the model input format:

```powershell
python scripts/convert_lcc_npz_to_candidates.py --lcc-root ../STS2R-code/outputs/LCC_Offline_Data --candidate-root data/candidates
```

## Main model

Run the following stages in order for seeds 42, 43, and 44. The first stage generates the 800/200/67 data split; the following stages train PointMLP, construct the structural input, train the superline-aware token encoder, and save the metrics.

```powershell
$runner = 'scripts/research_candidates/run_sbpt_net.py'
$lcc = '../STS2R-code/outputs/LCC_Offline_Data'
foreach ($seed in 42,43,44) {
  foreach ($stage in 'prepare-protocol','pointmlp','h90-cache','strongest','summarize') {
    python $runner $stage --seed $seed --candidate-root data/candidates --offline-root $lcc
    if ($LASTEXITCODE -ne 0) { throw "Failed: seed $seed, stage $stage" }
  }
}
```

The per-seed results are saved in `outputs/reproduction/seed<seed>/summary.json`.

## Candidate-point backbone baselines

```powershell
$lcc = '../STS2R-code/outputs/LCC_Offline_Data'
foreach ($seed in 42,43,44) {
  python scripts/formal_protocol/run_formal_baseline_with_training_seed.py --training-seed $seed --model all --run-tag "seed$seed" --offline-root $lcc --results-root outputs/reproduction/candidate_backbones --optimizer adamw
  if ($LASTEXITCODE -ne 0) { throw "Failed: candidate-point backbones, seed $seed" }
}
```

## Superline-token backbone comparison

After the seed-42 token cache is ready, run each backbone for seeds 42, 43, and 44:

```powershell
$runner = 'scripts/research_candidates/run_token_backbones.py'
$lcc = '../STS2R-code/outputs/LCC_Offline_Data'
$manifest = 'outputs/reproduction/seed42/split_manifest.csv'
$cache = 'outputs/reproduction/seed42/token_cache'
foreach ($seed in 42,43,44) {
  foreach ($model in 'pointnet','pointnet2','dgcnn','pointtransformer','pointnext','curvenet') {
    python $runner --seed $seed --model $model --candidate-root data/candidates --offline-root $lcc --split-manifest $manifest --token-cache-root $cache
    if ($LASTEXITCODE -ne 0) { throw "Failed: seed $seed, model $model" }
  }
}
```

## Real shoe-style transfer

The six directed Shoe A/B/C transfers use one labeled source style and a different target style. Run the seven candidate-point backbones with the paper settings:

```powershell
python scripts/formal_protocol/run_real_pairwise_shoe_transfer_20260910.py --output-root outputs/reproduction/real_style --candidate-root data/candidates --offline-root ../STS2R-code/outputs/LCC_Offline_Data --token-cache-root outputs/reproduction/seed42/token_cache --representations candidate_points --models pointnet pointnet2 pointmlp dgcnn pointtransformer pointnext curvenet --views-per-style 400 --seed 42
```
