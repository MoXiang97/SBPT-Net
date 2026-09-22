import numpy as np

from s2rspc.evaluation.projection import project_tokens_to_original_points


def test_uniform_projection_and_original_index_scatter():
    tokenized = {
        "candidate_orig_index": np.array([1, 3, 4], dtype=np.int64),
        "raw_count": np.asarray(6, dtype=np.int64),
        "patch_member_offsets": np.array([0, 2, 4], dtype=np.int64),
        "patch_member_indices": np.array([0, 1, 1, 2], dtype=np.int64),
    }
    output = project_tokens_to_original_points(
        tokenized, np.array([0.2, 0.8], dtype=np.float32), threshold=0.45
    )
    np.testing.assert_allclose(output["candidate_probability"], [0.2, 0.5, 0.8])
    np.testing.assert_array_equal(
        output["full_prediction"], [False, False, False, True, True, False]
    )
