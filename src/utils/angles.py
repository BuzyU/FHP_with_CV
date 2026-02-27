"""
Biomechanical Angle Computations

Computes clinically relevant angles from 3D skeleton data:
- CVA (Craniovertebral Angle) proxy
- Shoulder rounding angle
- Head forward displacement
- Head tilt angle
- Neck flexion angle

These values are used as INPUT FEATURES for the GCN, not as classification rules.
The neural network learns which combinations indicate FHP from data.
"""

import numpy as np
from typing import Dict, Optional, Tuple


def _safe_angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle (degrees) between two vectors, handling edge cases."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _vector_angle_with_axis(vec: np.ndarray, axis: np.ndarray) -> float:
    """Angle between a vector and a reference axis (degrees)."""
    return _safe_angle_between(vec, axis)


# ============================================================
# Individual Angle Computations
# ============================================================

def compute_cva_proxy(
    joints_3d: np.ndarray,
    neck_idx: int = 5,
    head_idx: int = 6,
) -> float:
    """
    Compute Craniovertebral Angle (CVA) proxy.

    The gold standard CVA is measured between:
      - A horizontal line
      - A line from C7 vertebra to ear tragus

    Since pose models don't have C7 or tragus directly, we use:
      - Neck joint (idx 5) as C7 proxy
      - Head joint (idx 6) as tragus proxy

    A normal CVA is typically 49-56°. FHP is indicated when CVA < ~46°.

    Args:
        joints_3d: (N, 3) upper body joints (normalized)
        neck_idx: Index of neck joint
        head_idx: Index of head joint

    Returns:
        CVA proxy angle in degrees
    """
    neck = joints_3d[neck_idx]
    head = joints_3d[head_idx]

    # Vector from neck (C7 proxy) to head (tragus proxy)
    neck_to_head = head - neck

    # Horizontal plane reference (sagittal: x-axis is depth/forward)
    # In normalized skeleton: x = depth, y = width, z = height
    horizontal = np.array([1.0, 0.0, 0.0])

    # CVA is measured in the sagittal plane (x-z)
    sagittal_vec = np.array([neck_to_head[0], 0.0, neck_to_head[2]])

    if np.linalg.norm(sagittal_vec) < 1e-8:
        return 90.0  # perfectly vertical = ideal

    angle = _safe_angle_between(sagittal_vec, horizontal)

    return angle


def compute_shoulder_rounding(
    joints_3d: np.ndarray,
    l_shoulder_idx: int = 7,
    r_shoulder_idx: int = 10,
    spine_idx: int = 3,
) -> float:
    """
    Compute shoulder rounding angle.

    Measures how far forward the shoulders are relative to the spine,
    indicating rounded/protracted shoulders often associated with FHP.

    Args:
        joints_3d: (N, 3) upper body joints
        l_shoulder_idx: Left shoulder index
        r_shoulder_idx: Right shoulder index
        spine_idx: Spine joint index

    Returns:
        Shoulder rounding angle in degrees (0° = perfectly aligned, 
        positive = shoulders forward/rounded)
    """
    l_shoulder = joints_3d[l_shoulder_idx]
    r_shoulder = joints_3d[r_shoulder_idx]
    spine = joints_3d[spine_idx]

    shoulder_mid = (l_shoulder + r_shoulder) / 2.0

    # Forward displacement of shoulders relative to spine
    shoulder_offset = shoulder_mid - spine

    # Angle relative to the vertical (z-axis)
    vertical = np.array([0.0, 0.0, 1.0])
    angle = _safe_angle_between(shoulder_offset, vertical)

    return angle


def compute_head_forward_displacement(
    joints_3d: np.ndarray,
    head_idx: int = 6,
    l_shoulder_idx: int = 7,
    r_shoulder_idx: int = 10,
    pelvis_idx: int = 0,
    neck_idx: int = 5,
) -> float:
    """
    Compute normalized head-to-shoulder forward displacement ratio.

    Measures how far the head is anterior to the shoulder line,
    normalized by torso length for body-size invariance.

    Args:
        joints_3d: (N, 3) upper body joints

    Returns:
        Normalized displacement ratio (positive = head forward)
    """
    head = joints_3d[head_idx]
    shoulder_mid = (joints_3d[l_shoulder_idx] + joints_3d[r_shoulder_idx]) / 2.0
    pelvis = joints_3d[pelvis_idx]
    neck = joints_3d[neck_idx]

    # Torso length for normalization
    torso_length = np.linalg.norm(neck - pelvis)
    if torso_length < 1e-8:
        return 0.0

    # Forward (depth-axis) displacement
    head_forward = head[0] - shoulder_mid[0]
    ratio = head_forward / torso_length

    return float(ratio)


def compute_head_tilt(
    joints_3d: np.ndarray,
    head_idx: int = 6,
    neck_idx: int = 5,
) -> float:
    """
    Compute head tilt angle relative to vertical.

    Measures the angle of the neck-to-head vector from the vertical axis.
    A perfectly upright head has ~0° tilt.

    Returns:
        Head tilt angle in degrees
    """
    neck = joints_3d[neck_idx]
    head = joints_3d[head_idx]

    neck_to_head = head - neck
    vertical = np.array([0.0, 0.0, 1.0])

    return _safe_angle_between(neck_to_head, vertical)


def compute_neck_flexion(
    joints_3d: np.ndarray,
    chest_idx: int = 4,
    neck_idx: int = 5,
    head_idx: int = 6,
) -> float:
    """
    Compute neck flexion angle.

    The angle at the neck joint between the chest-to-neck and neck-to-head vectors.
    This captures the degree of flexion/extension of the cervical spine.

    Returns:
        Neck flexion angle in degrees
    """
    chest = joints_3d[chest_idx]
    neck = joints_3d[neck_idx]
    head = joints_3d[head_idx]

    v1 = chest - neck   # chest → neck (looking down the spine)
    v2 = head - neck    # neck → head (looking up to head)

    return _safe_angle_between(v1, v2)


def compute_shoulder_symmetry(
    joints_3d: np.ndarray,
    l_shoulder_idx: int = 7,
    r_shoulder_idx: int = 10,
) -> float:
    """
    Compute shoulder height asymmetry.

    Measures the vertical (z-axis) difference between shoulders.
    Asymmetry can indicate compensatory posture patterns.

    Returns:
        Absolute height difference (normalized)
    """
    l_height = joints_3d[l_shoulder_idx, 2]
    r_height = joints_3d[r_shoulder_idx, 2]

    return float(abs(l_height - r_height))


# ============================================================
# Master Feature Extraction Function
# ============================================================

def compute_all_biomechanical_features(
    joints_3d: np.ndarray,
) -> Dict[str, float]:
    """
    Extract all biomechanical features from a single frame of 3D upper body joints.

    These features are used as AUXILIARY INPUTS to the GCN classifier.
    They do NOT determine the classification directly — the GCN learns
    which combinations of joint positions + angles indicate FHP.

    Args:
        joints_3d: Array of shape (13, 3) — upper body joints, normalized

    Returns:
        Dictionary of named biomechanical features
    """
    features = {
        "cva_proxy_angle": compute_cva_proxy(joints_3d),
        "shoulder_rounding_angle": compute_shoulder_rounding(joints_3d),
        "head_forward_displacement": compute_head_forward_displacement(joints_3d),
        "head_tilt_angle": compute_head_tilt(joints_3d),
        "neck_flexion_angle": compute_neck_flexion(joints_3d),
        "shoulder_symmetry": compute_shoulder_symmetry(joints_3d),
    }
    return features


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dict to a fixed-order numpy vector."""
    keys = sorted(features.keys())
    return np.array([features[k] for k in keys], dtype=np.float32)


def get_feature_names() -> list:
    """Get sorted list of feature names (matches vector order)."""
    dummy = compute_all_biomechanical_features(np.zeros((13, 3)))
    return sorted(dummy.keys())


# ============================================================
# Clinical Reference (for display/reporting only)
# ============================================================

CVA_REFERENCE = {
    "normal": {"min": 49, "max": 56, "color": (46, 204, 113)},    # green
    "mild_fhp": {"min": 44, "max": 49, "color": (241, 196, 15)},  # yellow
    "moderate_fhp": {"min": 40, "max": 44, "color": (243, 156, 18)},  # orange
    "severe_fhp": {"min": 0, "max": 40, "color": (231, 76, 60)},  # red
}


def classify_cva_severity(cva_angle: float) -> Tuple[str, Tuple[int, int, int]]:
    """
    Classify CVA into severity category (for DISPLAY ONLY, not model input).

    Returns:
        (severity_label, display_color)
    """
    for severity, info in CVA_REFERENCE.items():
        if info["min"] <= cva_angle <= info["max"]:
            return severity, info["color"]

    if cva_angle > 56:
        return "normal", CVA_REFERENCE["normal"]["color"]
    return "severe_fhp", CVA_REFERENCE["severe_fhp"]["color"]


if __name__ == "__main__":
    # Quick self-test with a synthetic pose
    np.random.seed(42)

    # Create a somewhat realistic upper body pose
    joints = np.array([
        [0.0,  0.0,  0.0],    # 0: Pelvis
        [0.1, -0.1,  0.0],    # 1: R_Hip
        [0.1,  0.1,  0.0],    # 2: L_Hip
        [0.0,  0.0,  0.3],    # 3: Spine
        [0.0,  0.0,  0.6],    # 4: Chest
        [0.0,  0.0,  0.8],    # 5: Neck
        [0.1,  0.0,  0.95],   # 6: Head (slightly forward = mild FHP)
        [0.0,  0.2,  0.75],   # 7: L_Shoulder
        [0.0,  0.3,  0.5],    # 8: L_Elbow
        [0.0,  0.35, 0.3],    # 9: L_Wrist
        [0.0, -0.2,  0.75],   # 10: R_Shoulder
        [0.0, -0.3,  0.5],    # 11: R_Elbow
        [0.0, -0.35, 0.3],    # 12: R_Wrist
    ], dtype=np.float32)

    features = compute_all_biomechanical_features(joints)
    print("Biomechanical Features:")
    for name, value in features.items():
        print(f"  {name}: {value:.2f}")

    vec = features_to_vector(features)
    print(f"\nFeature vector: {vec}")
    print(f"Feature names: {get_feature_names()}")

    severity, color = classify_cva_severity(features["cva_proxy_angle"])
    print(f"\nCVA severity: {severity} (color: {color})")
