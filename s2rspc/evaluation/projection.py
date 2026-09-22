"""Project token probabilities to the original raw ROI point space."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def project_tokens_to_candidates(
    token_probability: np.ndarray,
    patch_member_offsets: np.ndarray,
    patch_member_indices: np.ndarray,
    candidate_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    token_probability = np.asarray(token_probability, dtype=np.float32)
    offsets = np.asarray(patch_member_offsets, dtype=np.int64)
    members = np.asarray(patch_member_indices, dtype=np.int64)
    if len(offsets) != len(token_probability) + 1:
        raise ValueError("patch_member_offsets must contain one entry per token plus a final offset")
    probability_sum = np.zeros(int(candidate_count), dtype=np.float64)
    vote_count = np.zeros(int(candidate_count), dtype=np.uint16)
    for token_index, probability in enumerate(token_probability):
        local = members[offsets[token_index] : offsets[token_index + 1]]
        if np.any((local < 0) | (local >= candidate_count)):
            raise ValueError("Patch membership index is outside the candidate array")
        probability_sum[local] += float(probability)
        vote_count[local] += 1
    candidate_probability = np.zeros(int(candidate_count), dtype=np.float32)
    covered = vote_count > 0
    candidate_probability[covered] = (
        probability_sum[covered] / vote_count[covered]
    ).astype(np.float32)
    return candidate_probability, vote_count


def project_tokens_to_original_points(
    tokenized: Dict[str, np.ndarray],
    token_probability: np.ndarray,
    threshold: float,
) -> Dict[str, np.ndarray]:
    candidate_orig_index = np.asarray(tokenized["candidate_orig_index"], dtype=np.int64)
    raw_count = int(np.asarray(tokenized["raw_count"]).item())
    candidate_probability, vote_count = project_tokens_to_candidates(
        token_probability,
        tokenized["patch_member_offsets"],
        tokenized["patch_member_indices"],
        len(candidate_orig_index),
    )
    if len(np.unique(candidate_orig_index)) != len(candidate_orig_index):
        raise ValueError("candidate_orig_index contains duplicates")
    if np.any((candidate_orig_index < 0) | (candidate_orig_index >= raw_count)):
        raise ValueError("candidate_orig_index is outside the raw point cloud")
    full_probability = np.zeros(raw_count, dtype=np.float32)
    full_prediction = np.zeros(raw_count, dtype=bool)
    full_probability[candidate_orig_index] = candidate_probability
    full_prediction[candidate_orig_index] = candidate_probability >= float(threshold)
    return {
        "candidate_probability": candidate_probability,
        "candidate_vote_count": vote_count,
        "full_probability": full_probability,
        "full_prediction": full_prediction,
    }
