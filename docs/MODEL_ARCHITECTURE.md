# Current SBPT-Net architecture

The checkpoint architecture identifier is `SBPT-Net-PointMLP-Structural12D-H96`.

## Superline-based point tokens

Local-color-contrast candidates are connected with a spacing-normalized graph. For edge displacement `v` and local unit tangent `t`, the ranking residual is the norm of the orthogonal projection residual `v - (v^T t)t`. Its norm is unchanged by reversing either the edge or tangent direction.

Within a superline, support-region seeds are visited deterministically using their stored original indices and later seeds within the stride radius are suppressed. Each support region stores its exact candidate members. The representative point is an observed member minimizing the center-distance and perpendicular-distance objective in Eq. (4).

## Initial token prediction

Every token contributes six PointMLP channels:

- representative XYZ, centered by the cloud-level token mean;
- mean support-region RGB, normalized to `[0, 1]`.

PointMLP produces two class logits. The initial binary logit is class 1 minus class 0.

## Center-only structural encoder

The structural vector contains 12 values:

- six selected structural descriptors;
- six token attributes standardized with statistics fitted on synthetic training data.

The encoder is:

```text
Linear(12, 96) -> LayerNorm -> GELU -> Dropout -> Linear(96, 1)
```

It operates independently on each center token. It does not aggregate neighboring token embeddings.

## Fusion and training objective

The final logit is:

```text
z_final = (1 - beta) * clip(z_init, -c, c) + beta * z_str
```

The default configuration uses `beta = 0.5`, `c = 3`, and a separate deviation-regularizer bound `c_dev = 10`. PointMLP is frozen while the structural encoder is trained.

## Projection

Each token probability is assigned to the exact candidate points in its stored support region. Overlapping probabilities are averaged. Candidate values are then scattered to their original point indices; uncovered points remain background.
