"""
Skeleton Graph Definitions & Utilities

Defines the anatomical skeleton graph topology used by the GCN.
Supports multiple formats (Human3.6M, COCO) and provides utilities
for adjacency matrix construction, joint mapping, and visualization.
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# Joint Definitions — Human3.6M / VideoPose3D (17 joints)
# ============================================================

H36M_JOINT_NAMES = [
    "Pelvis",       # 0
    "R_Hip",        # 1
    "R_Knee",       # 2
    "R_Ankle",      # 3
    "L_Hip",        # 4
    "L_Knee",       # 5
    "L_Ankle",      # 6
    "Spine",        # 7
    "Chest",        # 8
    "Neck",         # 9  — proxy for C7 / bottom neck
    "Head",         # 10 — proxy for top of head / ear tragus
    "L_Shoulder",   # 11
    "L_Elbow",      # 12
    "L_Wrist",      # 13
    "R_Shoulder",   # 14
    "R_Elbow",      # 15
    "R_Wrist",      # 16
]

H36M_EDGES = [
    (0, 1), (1, 2), (2, 3),       # right leg
    (0, 4), (4, 5), (5, 6),       # left leg
    (0, 7), (7, 8), (8, 9),       # spine → neck
    (9, 10),                       # neck → head
    (9, 11), (11, 12), (12, 13),  # left arm
    (9, 14), (14, 15), (15, 16),  # right arm
]

# ============================================================
# Upper Body Subset (13 joints) — used for FHP detection
# ============================================================

UPPER_BODY_INDICES = [0, 1, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

UPPER_BODY_JOINT_NAMES = [
    "Pelvis",       # 0 → UB 0
    "R_Hip",        # 1 → UB 1
    "L_Hip",        # 4 → UB 2
    "Spine",        # 7 → UB 3
    "Chest",        # 8 → UB 4
    "Neck",         # 9 → UB 5  (C7 proxy / bottom neck)
    "Head",         # 10 → UB 6 (tragus proxy / head top)
    "L_Shoulder",   # 11 → UB 7
    "L_Elbow",      # 12 → UB 8
    "L_Wrist",      # 13 → UB 9
    "R_Shoulder",   # 14 → UB 10
    "R_Elbow",      # 15 → UB 11
    "R_Wrist",      # 16 → UB 12
]

# Edges remapped to upper-body indices
UPPER_BODY_EDGES = [
    (0, 1),    # pelvis → r_hip
    (0, 2),    # pelvis → l_hip
    (0, 3),    # pelvis → spine
    (3, 4),    # spine → chest
    (4, 5),    # chest → neck
    (5, 6),    # neck → head
    (5, 7),    # neck → l_shoulder
    (7, 8),    # l_shoulder → l_elbow
    (8, 9),    # l_elbow → l_wrist
    (5, 10),   # neck → r_shoulder
    (10, 11),  # r_shoulder → r_elbow
    (11, 12),  # r_elbow → r_wrist
]

# Landmark role mapping for clinical interpretation
LANDMARK_ROLES = {
    "head_top":      6,   # Head joint — tracks cranium position
    "ear_tragus":    6,   # Proxied by Head joint (ear ≈ head center)
    "top_neck":      5,   # Neck joint — upper cervical region
    "bottom_neck":   5,   # Neck joint — C7 proxy
    "shoulder_left": 7,
    "shoulder_right": 10,
    "hand_left":     9,
    "hand_right":    12,
}

# Joint colors for visualization (RGB)
JOINT_COLORS = {
    "spine": (52, 152, 219),      # blue
    "left_arm": (46, 204, 113),   # green
    "right_arm": (231, 76, 60),   # red
    "head": (241, 196, 15),       # gold
    "hips": (155, 89, 182),       # purple
}

BONE_GROUPS = {
    "spine": [(0, 3), (3, 4), (4, 5), (5, 6)],
    "left_arm": [(5, 7), (7, 8), (8, 9)],
    "right_arm": [(5, 10), (10, 11), (11, 12)],
    "hips": [(0, 1), (0, 2)],
}


# ============================================================
# Adjacency Matrix Construction
# ============================================================

def build_adjacency_matrix(
    edges: List[Tuple[int, int]],
    num_joints: int,
    self_loops: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Build adjacency matrix from edge list.

    Args:
        edges: List of (src, dst) joint pairs
        num_joints: Total number of joints
        self_loops: Whether to add self-connections
        normalize: Whether to apply symmetric normalization (D^{-1/2} A D^{-1/2})

    Returns:
        Adjacency matrix of shape (num_joints, num_joints)
    """
    A = np.zeros((num_joints, num_joints), dtype=np.float32)

    for src, dst in edges:
        A[src, dst] = 1.0
        A[dst, src] = 1.0  # undirected

    if self_loops:
        A += np.eye(num_joints, dtype=np.float32)

    if normalize:
        D = np.diag(A.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D.diagonal(), 1e-8)))
        A = D_inv_sqrt @ A @ D_inv_sqrt

    return A


def build_edge_index(edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Build edge_index tensor for PyTorch Geometric (COO format).

    Returns:
        Edge index array of shape (2, num_edges * 2) — bidirectional
    """
    src_list, dst_list = [], []
    for src, dst in edges:
        src_list.extend([src, dst])
        dst_list.extend([dst, src])

    return np.array([src_list, dst_list], dtype=np.int64)


def get_upper_body_adjacency(normalize: bool = True) -> np.ndarray:
    """Get pre-built upper body adjacency matrix."""
    return build_adjacency_matrix(UPPER_BODY_EDGES, len(UPPER_BODY_JOINT_NAMES), normalize=normalize)


def get_upper_body_edge_index() -> np.ndarray:
    """Get pre-built upper body edge index (PyG format)."""
    return build_edge_index(UPPER_BODY_EDGES)


def extract_upper_body(joints_3d: np.ndarray) -> np.ndarray:
    """
    Extract upper body joints from full 17-joint skeleton.

    Args:
        joints_3d: Array of shape (..., 17, 3)

    Returns:
        Array of shape (..., 13, 3)
    """
    return joints_3d[..., UPPER_BODY_INDICES, :]


def h36m_to_upper_body_index(h36m_idx: int) -> Optional[int]:
    """Map a Human3.6M joint index to upper body index."""
    try:
        return UPPER_BODY_INDICES.index(h36m_idx)
    except ValueError:
        return None


def get_joint_name(idx: int, upper_body: bool = True) -> str:
    """Get joint name by index."""
    names = UPPER_BODY_JOINT_NAMES if upper_body else H36M_JOINT_NAMES
    if 0 <= idx < len(names):
        return names[idx]
    return f"Joint_{idx}"


def get_bone_color(src: int, dst: int) -> Tuple[int, int, int]:
    """Get visualization color for a bone segment."""
    for group_name, bones in BONE_GROUPS.items():
        if (src, dst) in bones or (dst, src) in bones:
            return JOINT_COLORS.get(group_name, (200, 200, 200))
    return (200, 200, 200)


# ============================================================
# MediaPipe ↔ Human3.6M Mapping
# ============================================================

_MEDIAPIPE_TO_H36M = {
    0:  10,   # nose → head
    11: 11,   # left_shoulder
    12: 14,   # right_shoulder
    13: 12,   # left_elbow
    14: 15,   # right_elbow
    15: 13,   # left_wrist
    16: 16,   # right_wrist
    23: 4,    # left_hip
    24: 1,    # right_hip
    25: 5,    # left_knee
    26: 2,    # right_knee
    27: 6,    # left_ankle
    28: 3,    # right_ankle
}

def mediapipe_to_h36m(mp_landmarks: np.ndarray) -> np.ndarray:
    """
    Convert MediaPipe 33-landmark output to H36M 17-joint format.

    Args:
        mp_landmarks: Array of shape (33, 3) or (33, 4)

    Returns:
        Array of shape (17, 3)
    """
    h36m_joints = np.zeros((17, 3), dtype=np.float32)

    for mp_idx, h36m_idx in _MEDIAPIPE_TO_H36M.items():
        h36m_joints[h36m_idx] = mp_landmarks[mp_idx, :3]

    # Synthesize missing joints
    # Pelvis (0) = midpoint of hips
    h36m_joints[0] = (h36m_joints[1] + h36m_joints[4]) / 2.0

    # Spine (7) = midpoint of pelvis and chest
    chest_proxy = (h36m_joints[11] + h36m_joints[14]) / 2.0  # mid-shoulders
    h36m_joints[7] = (h36m_joints[0] + chest_proxy) / 2.0

    # Chest (8) = mid-shoulders
    h36m_joints[8] = chest_proxy

    # Neck (9) = slightly above chest toward head
    h36m_joints[9] = chest_proxy + 0.3 * (h36m_joints[10] - chest_proxy)

    return h36m_joints


if __name__ == "__main__":
    # Quick self-test
    adj = get_upper_body_adjacency(normalize=False)
    print(f"Upper body adjacency matrix shape: {adj.shape}")
    print(f"Upper body adjacency matrix:\n{adj.astype(int)}")

    edge_index = get_upper_body_edge_index()
    print(f"\nEdge index shape: {edge_index.shape}")
    print(f"Number of directed edges: {edge_index.shape[1]}")

    print(f"\nLandmark roles: {LANDMARK_ROLES}")
    print(f"\nJoint 6 name: {get_joint_name(6)}")
