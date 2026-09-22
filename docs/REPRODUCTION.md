# Reproduction protocol

## Scope of this repository

The current code implements the invariant Eq. (2) residual, 6D PointMLP input, center-only 12D structural encoder, fixed logit fusion, and overlap-averaged projection described in the current manuscript.

The repository intentionally does not include an experiment archive or provisional result tables. Paper numbers should be paired with the exact frozen checkpoint, preprocessing cache schema, configuration, run manifest, and validation record that produced them.

## Data split

The default configuration defines 1,000 synthetic clouds, with 800 for training and 200 for style-held-out validation. The command-line trainer obtains its file lists only from the synthetic subset.

Training, checkpoint selection, attribute normalization, and operating-threshold selection performed by the released scripts use synthetic data. During method development, earlier results on the available real evaluation set were observed before the final architecture was fixed. Accordingly, those real scans are evaluation data for the reported experiments but are not a development-independent untouched test set.

## Preprocessing

Use `configs/sbptnet_sts2r.yaml` and regenerate token caches after changing the superline formula or cache schema. Current caches have `schema_version = 2`.

The local color contrast threshold is `0.38169607520103455`. The representative center weight is `lambda_c = 0.35`.

## Two-stage training

Stage 1 trains PointMLP with centered XYZ and normalized mean RGB. Stage 2 reloads the best synthetic-validation PointMLP state, freezes it, fits the six attribute normalization statistics on synthetic training data, and trains only the 12D structural encoder.

Default settings include:

- AdamW;
- PointMLP learning rate `5e-4`, weight decay `1e-4`;
- structural encoder learning rate `5e-4`, weight decay `2e-4`;
- at most 100 epochs per stage and patience 15;
- checkpoint comparison at threshold 0.5 using mean per-cloud foreground IoU;
- final projection threshold selected on synthetic validation.

## Metric space

Evaluation is in the original point-cloud space. Token probabilities are averaged over exact overlapping support memberships and scattered by `candidate_orig_index`. Foreground IoU is computed per cloud. Subset reports use the mean across clouds; an equal-subset summary gives each shoe subset equal weight.

## Verification

```bash
pytest
python tools/verify_release.py
python -m s2rspc.cli.toy_demo --config configs/sbptnet_sts2r.yaml --output toy_output
```

If a checkpoint is distributed separately:

```bash
python tools/verify_release.py --checkpoint weights/sbptnet_best.pt
```
