import numpy as np
from scipy.spatial.transform import Rotation as R

# 你的四元数
qw = 0.8139158987083844
qx = 0.16906151376184375
qy = 0.2439698001163062
qz = -0.49943753465822605

# 平移
x = -0.27250514982855756
y = 0.28462689756862836
z = 0.3575306266789925

# 四元数注意顺序: (x, y, z, w)
quat = [qx, qy, qz, qw]

# 转换成旋转矩阵
r = R.from_quat(quat)

# 提取欧拉角 (默认是XYZ顺序)
rx, ry, rz = r.as_euler('xyz', degrees=False)  # degrees=True 可以直接得到角度

print(f"x: {x}")
print(f"y: {y}")
print(f"z: {z}")
print(f"rx: {rx} rad")
print(f"ry: {ry} rad")
print(f"rz: {rz} rad")
