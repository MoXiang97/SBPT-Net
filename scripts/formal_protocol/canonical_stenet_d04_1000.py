"""Single approved Compact16D-H96 OrderedPT1 STE-Net checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ROOT = (
    ROOT
    / "outputs"
    / "Formal_STENet_SixWay_Ablation_100ep_h8_20260730"
    / "full_with_context"
)
CANONICAL_MANIFEST = CANONICAL_ROOT / "CANONICAL_MODEL.json"
CANONICAL_CHECKPOINT = CANONICAL_ROOT / "stenet_best.pt"
CANONICAL_TRAINING_CODE = (
    ROOT
    / "scripts"
    / "formal_protocol"
    / "train_stenet_sixway_ablation_d04_1000.py"
)
CANONICAL_ARCHITECTURE_CODE = (
    ROOT / "scripts" / "formal_protocol" / "train_stenet_d04_1000.py"
)
CANONICAL_TRAINING_CORE = (
    ROOT / "scripts" / "formal_protocol" / "stenet_training_core_d04_1000.py"
)
CANONICAL_CHECKPOINT_SHA256 = "164CB5BEF7D5C8CDF55A582F3A065418B69C219D085442E5A8DD2E8876905C89"
CANONICAL_ARCHITECTURE_ID = "STE-Net-OrderedPT1-Compact16D-H96"
CANONICAL_TRAINING_STRATEGY = (
    "independent training from random initialization with a maximum of "
    "100 epochs and synthetic-validation early stopping patience 15"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require_canonical_checkpoint() -> Dict[str, Any]:
    if not CANONICAL_MANIFEST.is_file():
        raise FileNotFoundError(CANONICAL_MANIFEST)
    if not CANONICAL_CHECKPOINT.is_file():
        raise FileNotFoundError(CANONICAL_CHECKPOINT)
    if not CANONICAL_TRAINING_CODE.is_file():
        raise FileNotFoundError(CANONICAL_TRAINING_CODE)
    if not CANONICAL_ARCHITECTURE_CODE.is_file():
        raise FileNotFoundError(CANONICAL_ARCHITECTURE_CODE)
    if not CANONICAL_TRAINING_CORE.is_file():
        raise FileNotFoundError(CANONICAL_TRAINING_CORE)
    metadata = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))
    expected = str(metadata["checkpoint_sha256"]).upper()
    actual = sha256(CANONICAL_CHECKPOINT)
    if expected != CANONICAL_CHECKPOINT_SHA256 or actual != expected:
        raise RuntimeError(
            "Canonical STE-Net checkpoint hash mismatch: "
            f"constant={CANONICAL_CHECKPOINT_SHA256}, manifest={expected}, actual={actual}"
        )
    expected_code = str(metadata["training_code_sha256"]).upper()
    actual_code = sha256(CANONICAL_TRAINING_CODE)
    if actual_code != expected_code:
        raise RuntimeError(
            "Canonical STE-Net training code hash mismatch: "
            f"manifest={expected_code}, actual={actual_code}"
        )
    expected_architecture_code = str(metadata["architecture_code_sha256"]).upper()
    actual_architecture_code = sha256(CANONICAL_ARCHITECTURE_CODE)
    if actual_architecture_code != expected_architecture_code:
        raise RuntimeError(
            "Canonical STE-Net architecture-code hash mismatch: "
            f"manifest={expected_architecture_code}, actual={actual_architecture_code}"
        )
    expected_core = str(metadata["training_core_sha256"]).upper()
    actual_core = sha256(CANONICAL_TRAINING_CORE)
    if actual_core != expected_core:
        raise RuntimeError(
            "Canonical STE-Net training-core hash mismatch: "
            f"manifest={expected_core}, actual={actual_core}"
        )
    if metadata.get("training_strategy") != CANONICAL_TRAINING_STRATEGY:
        raise RuntimeError("The approved STE-Net training strategy does not match the final protocol")
    if int(metadata.get("maximum_epochs", -1)) != 100:
        raise RuntimeError("The canonical STE-Net maximum epoch setting must be 100")
    if int(metadata.get("early_stopping_patience", -1)) != 15:
        raise RuntimeError("The canonical STE-Net early-stopping patience must be 15")
    if metadata.get("architecture_id") != CANONICAL_ARCHITECTURE_ID:
        raise RuntimeError("The canonical STE-Net architecture is not Compact16D-H96")
    if metadata.get("status") != "final_canonical_locked":
        raise RuntimeError("The canonical STE-Net manifest is not locked as final")
    architecture = metadata.get("architecture", {})
    required_dimensions = {
        "descriptor_dim": 16,
        "token_attribute_dim": 19,
        "context_input_dim": 23,
        "hidden_dim": 96,
    }
    for key, expected_value in required_dimensions.items():
        if int(architecture.get(key, -1)) != expected_value:
            raise RuntimeError(
                f"Canonical STE-Net {key} mismatch: "
                f"expected={expected_value}, manifest={architecture.get(key)}"
            )
    thresholds = metadata["source_selected_thresholds"]
    token_threshold = float(
        thresholds.get("refined_token", thresholds.get("refined", -1.0))
    )
    if abs(token_threshold - 0.65) > 1e-12:
        raise RuntimeError(
            f"Canonical refined token threshold must be 0.65, got {token_threshold}"
        )
    projection_threshold = float(thresholds.get("full_point_projection", -1.0))
    if abs(projection_threshold - 0.35) > 1e-12:
        raise RuntimeError(
            "Canonical full-point projection threshold must be 0.35, "
            f"got {projection_threshold}"
        )
    return metadata
