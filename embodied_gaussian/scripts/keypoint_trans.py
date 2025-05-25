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


def find_keypoint_bindings(
    json_path: str | Path,
    init_keypoints,
    X_WB_is_B2W: bool = False,
):
    """
    第一次运行：找到每个关键点对应的最近邻点索引，建立绑定关系。

    Returns
    -------
    bindings : list of dict
        每个关键点的绑定信息，格式为：
        [
            {"type": "gaussian", "index": 123},  # 绑定到第123个高斯点
            {"type": "particle", "index": 45},   # 绑定到第45个粒子点
            ...
        ]
    new_keypoints : ndarray, shape (N,3)
        替换后的关键点坐标
    """
    # 处理空输入的情况
    if len(init_keypoints) == 0:
        return [], np.empty((0, 3), dtype=np.float32)
    
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
    
    # 确保 init_pts 是二维数组
    if init_pts.ndim == 1:
        init_pts = init_pts.reshape(1, -1)
    
    # (N,1,3) - (1,M,3) => (N,M,3) => 求范数 (N,M)
    diffs = init_pts[:, None, :] - candidates[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)               # (N, Ng+Np)

    nearest_idx = dists.argmin(axis=1)                   # (N,)
    new_keypoints = candidates[nearest_idx]              # (N,3)

    # ---------- 2. 生成绑定信息 ----------
    bindings = []
    for idx in nearest_idx:
        if idx < split:
            # 绑定到高斯点
            bindings.append({"type": "gaussian", "index": int(idx)})
        else:
            # 绑定到粒子点
            bindings.append({"type": "particle", "index": int(idx - split)})

    return bindings, new_keypoints


def update_keypoints_from_bindings(
    json_path: str | Path,
    bindings: list,
    X_WB_is_B2W: bool = False,
):
    """
    根据已有的绑定关系更新关键点坐标。

    Parameters
    ----------
    json_path : str | Path
        JSON文件路径
    bindings : list of dict
        关键点绑定信息，由 find_keypoint_bindings 返回
    X_WB_is_B2W : bool
        X_WB矩阵的含义

    Returns
    -------
    updated_keypoints : ndarray, shape (N,3)
        更新后的关键点坐标
    """
    if len(bindings) == 0:
        return np.empty((0, 3), dtype=np.float32)
    
    # ---------- 0. 读取 & 变换 ----------
    data, X_WB = load_tblock(json_path)
    T_BW = X_WB if X_WB_is_B2W else np.linalg.inv(X_WB)  # (4,4)

    g_means_B = np.asarray(data["gaussians"]["means"], dtype=np.float32)  # (Ng,3)
    p_means_B = np.asarray(data["particles"]["means"], dtype=np.float32)  # (Np,3)

    g_means_W = transform_pts(g_means_B, T_BW)  # (Ng,3)
    p_means_W = transform_pts(p_means_B, T_BW)  # (Np,3)

    # ---------- 1. 根据绑定信息获取对应点坐标 ----------
    updated_keypoints = []
    for binding in bindings:
        if binding["type"] == "gaussian":
            idx = binding["index"]
            if idx < len(g_means_W):
                updated_keypoints.append(g_means_W[idx])
            else:
                raise IndexError(f"高斯点索引 {idx} 超出范围 (总数: {len(g_means_W)})")
        elif binding["type"] == "particle":
            idx = binding["index"]
            if idx < len(p_means_W):
                updated_keypoints.append(p_means_W[idx])
            else:
                raise IndexError(f"粒子点索引 {idx} 超出范围 (总数: {len(p_means_W)})")
        else:
            raise ValueError(f"未知的绑定类型: {binding['type']}")
    
    return np.array(updated_keypoints, dtype=np.float32)


def replace_keypoints_with_nearest(
    json_path: str | Path,
    init_keypoints,
    X_WB_is_B2W: bool = False,
):
    """
    把 init_keypoints 替换成与之最近的 Gaussian/Particle 中心点（世界坐标系下）。
    
    这是原有的函数，保持向后兼容性。
    如果需要后续更新功能，建议使用 find_keypoint_bindings + update_keypoints_from_bindings。

    Returns
    -------
    new_keypoints : ndarray, shape (N,3)
    """
    bindings, new_keypoints = find_keypoint_bindings(json_path, init_keypoints, X_WB_is_B2W)
    return new_keypoints


def save_bindings(bindings: list, save_path: str | Path):
    """保存绑定信息到JSON文件"""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(bindings, f, indent=2, ensure_ascii=False)


def load_bindings(load_path: str | Path) -> list:
    """从JSON文件加载绑定信息"""
    with open(load_path, "r", encoding="utf-8") as f:
        return json.load(f)


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

    print("=== 第一次运行：建立绑定关系 ===")
    bindings, new_kpts = find_keypoint_bindings(
        "embodied_gaussian/objects/tblock.json",
        init_keypoint_positions,
        X_WB_is_B2W=False,
    )
    
    print("绑定信息：")
    for i, binding in enumerate(bindings):
        print(f"  关键点{i}: {binding}")
    
    print(f"\n替换后的关键点坐标：\n{new_kpts}")
    
    # 保存绑定信息
    save_bindings(bindings, "keypoint_bindings.json")
    print("\n绑定信息已保存到 keypoint_bindings.json")
    
    print("\n=== 后续更新：根据绑定关系更新坐标 ===")
    # 模拟后续更新（实际使用时JSON文件内容会变化）
    updated_kpts = update_keypoints_from_bindings(
        "embodied_gaussian/objects/tblock.json",
        bindings,
        X_WB_is_B2W=False,
    )
    
    print(f"更新后的关键点坐标：\n{updated_kpts}")
    
    # 验证两次结果应该相同（因为JSON文件没变）
    print(f"\n坐标是否一致: {np.allclose(new_kpts, updated_kpts)}")
