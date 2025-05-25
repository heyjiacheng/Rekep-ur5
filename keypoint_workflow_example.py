#!/usr/bin/env python3
"""
关键点绑定工作流程示例

这个脚本演示了如何使用新的关键点绑定机制：
1. 第一次运行：建立关键点与Gaussian/Particle的绑定关系
2. 后续运行：根据绑定关系快速更新关键点坐标
"""

import numpy as np
from pathlib import Path
from embodied_gaussian.scripts.keypoint_trans import (
    find_keypoint_bindings,
    update_keypoints_from_bindings,
    save_bindings,
    load_bindings
)


def workflow_example():
    """完整的工作流程示例"""
    
    # 你的初始关键点位置
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
    
    json_file = "embodied_gaussian/objects/tblock.json"
    bindings_file = "my_keypoint_bindings.json"
    
    print("🔧 关键点绑定工作流程示例")
    print("=" * 50)
    
    # ========== 第一次运行：建立绑定关系 ==========
    print("\n📍 步骤1: 建立关键点绑定关系")
    
    if not Path(bindings_file).exists():
        print("  正在分析初始关键点，寻找最近邻...")
        
        bindings, keypoints = find_keypoint_bindings(
            json_file,
            init_keypoint_positions,
            X_WB_is_B2W=False
        )
        
        # 保存绑定信息
        save_bindings(bindings, bindings_file)
        
        print(f"  ✅ 绑定关系已建立并保存到 {bindings_file}")
        print(f"  📊 绑定统计:")
        
        gaussian_count = sum(1 for b in bindings if b["type"] == "gaussian")
        particle_count = sum(1 for b in bindings if b["type"] == "particle")
        
        print(f"    - 绑定到Gaussian点: {gaussian_count}个")
        print(f"    - 绑定到Particle点: {particle_count}个")
        
        print(f"  📋 详细绑定信息:")
        for i, binding in enumerate(bindings):
            print(f"    关键点{i}: {binding['type']} #{binding['index']}")
            
    else:
        print(f"  📁 发现已存在的绑定文件: {bindings_file}")
        bindings = load_bindings(bindings_file)
        print(f"  ✅ 已加载 {len(bindings)} 个关键点的绑定信息")
    
    # ========== 后续运行：更新关键点坐标 ==========
    print(f"\n🔄 步骤2: 根据绑定关系更新关键点坐标")
    print("  正在从JSON文件读取最新的点位置...")
    
    updated_keypoints = update_keypoints_from_bindings(
        json_file,
        bindings,
        X_WB_is_B2W=False
    )
    
    print(f"  ✅ 已更新 {len(updated_keypoints)} 个关键点的坐标")
    
    # ========== 显示结果 ==========
    print(f"\n📊 更新后的关键点坐标:")
    print("-" * 60)
    for i, (binding, coord) in enumerate(zip(bindings, updated_keypoints)):
        orig_coord = init_keypoint_positions[i]
        distance = np.linalg.norm(np.array(orig_coord) - coord)
        print(f"关键点{i:2d}: {binding['type']:8s} #{binding['index']:3d} -> "
              f"[{coord[0]:8.4f}, {coord[1]:8.4f}, {coord[2]:8.4f}] "
              f"(移动距离: {distance:.4f})")
    
    print(f"\n💡 使用说明:")
    print(f"  1. 第一次运行会建立绑定关系并保存到 {bindings_file}")
    print(f"  2. 后续运行只需调用 update_keypoints_from_bindings() 即可快速更新")
    print(f"  3. 即使JSON文件中的点位置发生变化，关键点也会自动跟随对应的点更新")
    
    return bindings, updated_keypoints


def quick_update_example(bindings_file: str = "my_keypoint_bindings.json"):
    """快速更新示例（适用于后续调用）"""
    
    json_file = "embodied_gaussian/objects/tblock.json"
    
    if not Path(bindings_file).exists():
        print(f"❌ 绑定文件 {bindings_file} 不存在，请先运行 workflow_example()")
        return None
    
    print(f"⚡ 快速更新关键点坐标...")
    
    # 加载绑定信息
    bindings = load_bindings(bindings_file)
    
    # 更新坐标
    updated_keypoints = update_keypoints_from_bindings(
        json_file,
        bindings,
        X_WB_is_B2W=False
    )
    
    print(f"✅ 已更新 {len(updated_keypoints)} 个关键点")
    
    return updated_keypoints


if __name__ == "__main__":
    # 运行完整工作流程
    bindings, keypoints = workflow_example()
    
    print(f"\n" + "="*50)
    print(f"🚀 后续使用时，只需调用:")
    print(f"   updated_keypoints = quick_update_example()")
    print(f"   # 或者直接使用:")
    print(f"   # bindings = load_bindings('my_keypoint_bindings.json')")
    print(f"   # keypoints = update_keypoints_from_bindings(json_file, bindings)") 