# Data format

## Raw point clouds

Each raw point cloud is an ASCII file with at least seven whitespace-separated columns:

```text
X Y Z R G B Label
```

RGB may use `[0, 1]` or `[0, 255]`. Values of `Label` above 0.5 are positive. Input files must already contain the shoe-upper region of interest.

## Directory layout

```text
data/raw/synthetic/*.txt
data/raw/shoe_a/*.txt
data/raw/shoe_b/*.txt
data/raw/shoe_c/*.txt
```

Synthetic filenames retain a style prefix before the first hyphen so the configured style-held-out split can be reproduced.

## Candidate cache

Candidate NPZ files store:

- `xyz`, `rgb`, `label`, and local color contrast `lcc`;
- `orig_index`, the exact row index in the raw point cloud;
- `raw_count` and, when labels are available, `raw_label`.

## Token cache schema 2

The current token NPZ stores:

- anchors and labels: `xyz`, `rgb`, `label`, `orig_index`, `fragment_id`;
- trace features: `features`;
- model features: `structural_descriptors` with shape `N x 6`, and `token_attributes` with shape `N x 6`;
- exact support membership: `patch_member_offsets`, `patch_member_indices`;
- projection mapping: `candidate_orig_index`, `raw_count`, and optional `raw_label`;
- `schema_version = 2`.

Support members for token `i` are:

```python
patch_member_indices[patch_member_offsets[i]:patch_member_offsets[i + 1]]
```

These exact members are used for projection. No nearest-neighbor expansion is applied. Caches from the removed 16D/19D/23D contextual implementation are rejected and must be regenerated.

## Excluded artifacts

Raw data, caches, predictions, run directories, and trained checkpoints are ignored by Git. Release checkpoints should be attached to a versioned GitHub Release.
