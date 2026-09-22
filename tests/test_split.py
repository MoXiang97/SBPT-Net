from pathlib import Path

from s2rspc.data.split import source_only_split, synthetic_style_id


def test_style_prefix_and_holdout_split():
    files = [Path(f"A-{i}_cloud.npz") for i in range(3)] + [
        Path(f"TeBu-{i}_cloud.npz") for i in range(2)
    ]
    train, validation = source_only_split(files, 2, 2, 42, ["TeBu"])
    assert all(synthetic_style_id(path) == "A" for path in train)
    assert all(synthetic_style_id(path) == "TeBu" for path in validation)
    assert not set(train) & set(validation)


def test_random_split_is_disjoint():
    files = [Path(f"sample-{index}.npz") for index in range(10)]
    train, validation = source_only_split(files, 7, 3, 42, [])
    assert len(train) == 7 and len(validation) == 3
    assert not set(train) & set(validation)
