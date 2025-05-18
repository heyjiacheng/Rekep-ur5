import json
from pathlib import Path
import numpy as np


def load_tblock(json_path: str | Path):
    """读取 JSON 并返回数据及 4×4 NumPy 矩阵 X_WB。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data, np.asarray(data["X_WB"], dtype=np.float32)  # (4,4)


def transform_pts(points_B: np.ndarray, T_BW: np.ndarray) -> np.ndarray:
    """批量将 (N,3) 点从物体坐标系 B → 世界坐标系 W。"""
    homo_B = np.hstack((points_B, np.ones((points_B.shape[0], 1), points_B.dtype)))  # (N,4)
    return (T_BW @ homo_B.T).T[:, :3]  # (N,3)


def replace_keypoints_with_nearest(
    json_path: str | Path,
    init_keypoints,
    X_WB_is_B2W: bool = False,
):
    """
    把 init_keypoints 替换成与之最近的 Gaussian/Particle 中心点（世界坐标系下）。

    Returns
    -------
    new_keypoints : ndarray, shape (N,3)
    # idx_info     : ndarray, shape (N,), int  (可选) <0=Gaussian, ≥0=Particle 索引>
    """
    # ---------- 0. 读取 & 变换 ----------
    data, X_WB = load_tblock(json_path)
    T_BW = X_WB if X_WB_is_B2W else np.linalg.inv(X_WB)  # (4,4)

    g_means_B = np.asarray(data["gaussians"]["means"], dtype=np.float32)  # (Ng,3)
    p_means_B = np.asarray(data["particles"]["means"], dtype=np.float32)  # (Np,3)

    g_means_W = transform_pts(g_means_B, T_BW)  # (Ng,3)
    p_means_W = transform_pts(p_means_B, T_BW)  # (Np,3)

    # 把两类候选点连在一起，方便一次性计算欧氏距离
    candidates = np.vstack((g_means_W, p_means_W))       # (Ng+Np,3)
    split = len(g_means_W)                               # 用于区分高斯和粒子

    # ---------- 1. 对每个关键点找最近邻 ----------
    init_pts = np.asarray(init_keypoints, dtype=np.float32)  # (N,3)
    # (N,1,3) - (1,M,3) => (N,M,3) => 求范数 (N,M)
    diffs = init_pts[:, None, :] - candidates[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)               # (N, Ng+Np)

    nearest_idx = dists.argmin(axis=1)                   # (N,)
    new_keypoints = candidates[nearest_idx]              # (N,3)

    # 如果想知道命中了高斯(返回 -i-1) 还是粒子(返回 j)，可取消下一行注释
    # idx_info = np.where(nearest_idx < split, -(nearest_idx + 1), nearest_idx - split)

    return new_keypoints  # , idx_info


# -------------------- 示例调用 --------------------
if __name__ == "__main__":
    init_keypoint_positions = [
        [-0.1690338161117148, -0.1250190099266853, 0.5740000000000001],
        [-0.15763108811651552, 0.061827029131252834, 0.58],
        [-0.15424598991193214, 0.16958281969765782, 0.585],
        [-0.022477837066350932, 0.15519262959839095, 0.588],
        [0.0013521416870752774, -0.12697923830532667, 0.583],
        [0.04738007038941239, 0.10997132290923246, 0.591],
        [0.15375298830627504, 0.021725037306170116, 0.436],
        [0.2196976661206351, -0.07336994744696035, 0.593],
        [0.16895405225684026, 0.09713877895283203, 0.4],
        [0.42175916765330773, -0.14394309998053592, 0.835],
    ]

    new_kpts = replace_keypoints_with_nearest(
        "objects/tblock.json",
        init_keypoint_positions,
        X_WB_is_B2W=False,   # ← 根据实际含义调整
    )
    print("替换后的关键点坐标：\n", new_kpts)
