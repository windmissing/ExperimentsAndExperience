"""
实验2：3DGS物理驱动绑定方式对比实验
对比4种绑定方式：LBS / PartMM / SparseCP / Direct
"""

import taichi as ti
import numpy as np
import json
import os
import time

# 初始化Taichi（使用GPU）
ti.init(arch=ti.cuda, device_memory_GB=4.0)

# ============ 实验配置 ============

CONFIG = {
    "binding_methods": ["LBS", "PartMM", "SparseCP", "Direct"],
    "force_levels": [
        {"name": "light", "magnitude": 0.5},
        {"name": "medium", "magnitude": 1.0},
        {"name": "heavy", "magnitude": 2.0},
    ],
    "cylinder": {
        "radius": 0.1,
        "height": 0.3,
        "num_gaussians": 5000,
    },
    "simulation": {
        "fps": 60,
        "duration": 3.0,
        "gravity": -9.81,
        "dt": 1.0 / 60.0,
    },
}

OUTPUT_DIR = "D:/Experiment2_Binding/results"

# ============ 高斯点系统 ============

MAX_GAUSSIANS = 10000

# 高斯点数据
gaussian_pos = ti.Vector.field(3, ti.f32, shape=(MAX_GAUSSIANS,))
gaussian_rest_pos = ti.Vector.field(3, ti.f32, shape=(MAX_GAUSSIANS,))  # 静止位置
gaussian_vel = ti.Vector.field(3, ti.f32, shape=(MAX_GAUSSIANS,))
gaussian_pred = ti.Vector.field(3, ti.f32, shape=(MAX_GAUSSIANS,))
gaussian_inv_mass = ti.field(ti.f32, shape=(MAX_GAUSSIANS,))
gaussian_fixed = ti.field(ti.i32, shape=(MAX_GAUSSIANS,))

# 骨骼数据（用于LBS）
num_bones = ti.field(ti.i32, shape=())
bone_pos = ti.Vector.field(3, ti.f32, shape=(20,))
bone_rest_pos = ti.Vector.field(3, ti.f32, shape=(20,))
bone_vel = ti.Vector.field(3, ti.f32, shape=(20,))

# 权重数据（用于LBS）
bone_weights = ti.field(ti.f32, shape=(MAX_GAUSSIANS, 20))

# 部件数据（用于PartMM）
num_parts = ti.field(ti.i32, shape=())
part_id = ti.field(ti.i32, shape=(MAX_GAUSSIANS,))  # 每个高斯点属于哪个部件
part_pos = ti.Vector.field(3, ti.f32, shape=(10,))  # 部件中心
part_rest_pos = ti.Vector.field(3, ti.f32, shape=(10,))

# 控制点数据（用于SparseCP）
num_control_points = ti.field(ti.i32, shape=())
control_point_pos = ti.Vector.field(3, ti.f32, shape=(512,))
control_point_rest_pos = ti.Vector.field(3, ti.f32, shape=(512,))
control_point_vel = ti.Vector.field(3, ti.f32, shape=(512,))
cp_weights = ti.field(ti.f32, shape=(MAX_GAUSSIANS, 512))

# 实验参数
current_method = ti.field(ti.i32, shape=())  # 0: LBS, 1: PartMM, 2: SparseCP, 3: Direct
current_force = ti.field(ti.f32, shape=())
num_gaussians = ti.field(ti.i32, shape=())

# 统计信息
volume_ratio = ti.field(ti.f32, shape=())
laplacian_energy = ti.field(ti.f32, shape=())
overlap_count = ti.field(ti.i32, shape=())


# ============ 初始化 ============

@ti.kernel
def init_cylinder_gaussians(radius: ti.f32, height: ti.f32, num_pts: ti.i32):
    """初始化圆柱体高斯点"""
    num_gaussians[None] = num_pts
    
    for i in range(num_pts):
        # 均匀分布在圆柱体内
        theta = ti.random() * 2 * ti.pi
        r = radius * ti.sqrt(ti.random())
        y = ti.random() * height
        
        x = r * ti.cos(theta)
        z = r * ti.sin(theta)
        
        gaussian_rest_pos[i] = ti.Vector([x, y, z])
        gaussian_pos[i] = gaussian_rest_pos[i]
        gaussian_vel[i] = ti.Vector([0.0, 0.0, 0.0])
        gaussian_pred[i] = gaussian_pos[i]
        gaussian_inv_mass[i] = 1.0 / (0.5 / num_pts)
        gaussian_fixed[i] = 0
    
    # 固定底部（y < 0.01）
    for i in range(num_pts):
        if gaussian_pos[i][1] < 0.01:
            gaussian_fixed[i] = 1


@ti.kernel
def init_bones():
    """初始化骨骼（用于LBS）"""
    num_bones[None] = 5
    
    # 沿圆柱体高度分布5个骨骼
    for i in range(5):
        y = i * 0.06  # 0, 0.06, 0.12, 0.18, 0.24
        bone_rest_pos[i] = ti.Vector([0.0, y, 0.0])
        bone_pos[i] = bone_rest_pos[i]
        bone_vel[i] = ti.Vector([0.0, 0.0, 0.0])
    
    # 计算权重（基于距离）
    for i in range(num_gaussians[None]):
        pos = gaussian_rest_pos[i]
        total_weight = 0.0
        
        for j in range(num_bones[None]):
            dist = (pos - bone_rest_pos[j]).norm()
            weight = 1.0 / (dist + 0.01)
            bone_weights[i, j] = weight
            total_weight += weight
        
        # 归一化
        for j in range(num_bones[None]):
            bone_weights[i, j] /= total_weight


@ti.kernel
def init_parts():
    """初始化部件（用于PartMM）"""
    num_parts[None] = 5
    
    # 沿圆柱体高度分成5个部件
    for i in range(num_gaussians[None]):
        y = gaussian_rest_pos[i][1]
        part_idx = int(y / 0.06)
        if part_idx >= 5:
            part_idx = 4
        part_id[i] = part_idx
    
    # 计算部件中心
    for p in range(num_parts[None]):
        center = ti.Vector([0.0, 0.0, 0.0])
        count = 0
        for i in range(num_gaussians[None]):
            if part_id[i] == p:
                center += gaussian_rest_pos[i]
                count += 1
        if count > 0:
            center /= count
        part_rest_pos[p] = center
        part_pos[p] = center


@ti.kernel
def init_control_points():
    """初始化控制点（用于SparseCP）"""
    num_control_points[None] = 64  # 简化为64个控制点
    
    # 均匀分布在圆柱体表面和内部
    for i in range(num_control_points[None]):
        theta = ti.random() * 2 * ti.pi
        r = 0.1 * ti.sqrt(ti.random())
        y = ti.random() * 0.3
        
        x = r * ti.cos(theta)
        z = r * ti.sin(theta)
        
        control_point_rest_pos[i] = ti.Vector([x, y, z])
        control_point_pos[i] = control_point_rest_pos[i]
        control_point_vel[i] = ti.Vector([0.0, 0.0, 0.0])
    
    # 计算权重（基于距离）
    for i in range(num_gaussians[None]):
        pos = gaussian_rest_pos[i]
        total_weight = 0.0
        
        for j in range(num_control_points[None]):
            dist = (pos - control_point_rest_pos[j]).norm()
            weight = 1.0 / (dist + 0.01)
            cp_weights[i, j] = weight
            total_weight += weight
        
        # 归一化
        for j in range(num_control_points[None]):
            cp_weights[i, j] /= total_weight


# ============ LBS绑定 ============

@ti.kernel
def lbs_update():
    """LBS：根据骨骼更新高斯点位置"""
    for i in range(num_gaussians[None]):
        if gaussian_fixed[i] == 1:
            continue
        
        # 线性混合蒙皮
        new_pos = ti.Vector([0.0, 0.0, 0.0])
        for j in range(num_bones[None]):
            weight = bone_weights[i, j]
            # 骨骼的位移
            bone_disp = bone_pos[j] - bone_rest_pos[j]
            new_pos += weight * (gaussian_rest_pos[i] + bone_disp)
        
        gaussian_pos[i] = new_pos


# ============ PartMM绑定 ============

@ti.kernel
def partmm_update():
    """PartMM：根据部件更新高斯点位置"""
    for i in range(num_gaussians[None]):
        if gaussian_fixed[i] == 1:
            continue
        
        p = part_id[i]
        # 部件的位移
        part_disp = part_pos[p] - part_rest_pos[p]
        
        # 局部变形（相对于部件中心）
        local_pos = gaussian_rest_pos[i] - part_rest_pos[p]
        
        # 简单的缩放变形（模拟软体效果）
        scale = 1.0 + part_disp[0] * 0.5  # X方向拉伸
        local_pos[0] *= scale
        local_pos[1] *= (2.0 - scale)  # Y方向压缩（体积保持）
        
        gaussian_pos[i] = part_pos[p] + local_pos


# ============ SparseCP绑定 ============

@ti.kernel
def sparsecp_update():
    """SparseCP：根据控制点更新高斯点位置"""
    for i in range(num_gaussians[None]):
        if gaussian_fixed[i] == 1:
            continue
        
        # 加权平均控制点的位移
        disp = ti.Vector([0.0, 0.0, 0.0])
        for j in range(num_control_points[None]):
            weight = cp_weights[i, j]
            cp_disp = control_point_pos[j] - control_point_rest_pos[j]
            disp += weight * cp_disp
        
        gaussian_pos[i] = gaussian_rest_pos[i] + disp


# ============ Direct绑定 ============

@ti.kernel
def direct_predict(dt: ti.f32, gravity: ti.f32):
    """Direct：直接物理仿真（预测）"""
    for i in range(num_gaussians[None]):
        if gaussian_fixed[i] == 0:
            gaussian_vel[i][1] += gravity * dt
            gaussian_pred[i] = gaussian_pos[i] + gaussian_vel[i] * dt


@ti.kernel
def direct_solve_constraints(dt: ti.f32):
    """Direct：求解约束"""
    # 地面碰撞
    for i in range(num_gaussians[None]):
        if gaussian_pred[i][1] < 0.0:
            gaussian_pred[i][1] = 0.0
    
    # 形状约束（保持圆柱体形状）
    # 计算质心
    center = ti.Vector([0.0, 0.0, 0.0])
    for i in range(num_gaussians[None]):
        center += gaussian_pred[i]
    center /= num_gaussians[None]
    
    # 约束到圆柱体表面
    radius = CONFIG["cylinder"]["radius"]
    for i in range(num_gaussians[None]):
        if gaussian_fixed[i] == 1:
            continue
        
        pos = gaussian_pred[i]
        # XZ平面约束
        xz_dist = ti.sqrt(pos[0] * pos[0] + pos[2] * pos[2])
        if xz_dist > radius:
            scale = radius / xz_dist
            gaussian_pred[i][0] *= scale
            gaussian_pred[i][2] *= scale


@ti.kernel
def direct_update_velocities(dt: ti.f32):
    """Direct：更新速度"""
    for i in range(num_gaussians[None]):
        if gaussian_fixed[i] == 0:
            gaussian_vel[i] = (gaussian_pred[i] - gaussian_pos[i]) / dt
            gaussian_pos[i] = gaussian_pred[i]


# ============ 物理仿真（所有方法共用） ============

@ti.kernel
def apply_external_force(force_magnitude: ti.f32, dt: ti.f32):
    """施加外力（顶部侧向力）"""
    for i in range(num_gaussians[None]):
        if gaussian_fixed[i] == 1:
            continue
        
        # 只对顶部施加力
        y = gaussian_pos[i][1]
        if y > 0.25:
            force = ti.Vector([force_magnitude, 0.0, 0.0])
            gaussian_vel[i] += force * dt


@ti.kernel
def update_bones(force_magnitude: ti.f32, dt: ti.f32):
    """更新骨骼位置（LBS用）"""
    for i in range(num_bones[None]):
        if i == 0:
            continue  # 底部骨骼固定
        
        # 顶部骨骼受力最大
        factor = i / num_bones[None]
        force = ti.Vector([force_magnitude * factor, 0.0, 0.0])
        bone_vel[i] += force * dt
        bone_pos[i] += bone_vel[i] * dt


@ti.kernel
def update_parts(force_magnitude: ti.f32, dt: ti.f32):
    """更新部件位置（PartMM用）"""
    for i in range(num_parts[None]):
        if i == 0:
            continue  # 底部部件固定
        
        factor = i / num_parts[None]
        force = ti.Vector([force_magnitude * factor, 0.0, 0.0])
        part_pos[i] += force * dt


@ti.kernel
def update_control_points(force_magnitude: ti.f32, dt: ti.f32):
    """更新控制点位置（SparseCP用）"""
    for i in range(num_control_points[None]):
        y = control_point_pos[i][1]
        if y < 0.01:
            continue  # 底部控制点固定
        
        factor = y / 0.3
        force = ti.Vector([force_magnitude * factor, 0.0, 0.0])
        control_point_vel[i] += force * dt
        control_point_pos[i] += control_point_vel[i] * dt


# ============ 数据提取 ============

@ti.kernel
def compute_volume_ratio() -> ti.f32:
    """计算体积保持率"""
    # 简化的体积计算：计算包围盒
    min_x = 1000.0
    max_x = -1000.0
    min_y = 1000.0
    max_y = -1000.0
    min_z = 1000.0
    max_z = -1000.0
    
    for i in range(num_gaussians[None]):
        pos = gaussian_pos[i]
        if pos[0] < min_x:
            min_x = pos[0]
        if pos[0] > max_x:
            max_x = pos[0]
        if pos[1] < min_y:
            min_y = pos[1]
        if pos[1] > max_y:
            max_y = pos[1]
        if pos[2] < min_z:
            min_z = pos[2]
        if pos[2] > max_z:
            max_z = pos[2]
    
    current_volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
    
    # 静止时的体积
    min_x = 1000.0
    max_x = -1000.0
    min_y = 1000.0
    max_y = -1000.0
    min_z = 1000.0
    max_z = -1000.0
    
    for i in range(num_gaussians[None]):
        pos = gaussian_rest_pos[i]
        if pos[0] < min_x:
            min_x = pos[0]
        if pos[0] > max_x:
            max_x = pos[0]
        if pos[1] < min_y:
            min_y = pos[1]
        if pos[1] > max_y:
            max_y = pos[1]
        if pos[2] < min_z:
            min_z = pos[2]
        if pos[2] > max_z:
            max_z = pos[2]
    
    rest_volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
    
    if rest_volume > 0.0001:
        return current_volume / rest_volume
    else:
        return 1.0


@ti.kernel
def compute_laplacian_energy() -> ti.f32:
    """计算拉普拉斯能量（形变平滑度）"""
    energy = 0.0
    
    # 简化的拉普拉斯：相邻点的差异
    for i in range(num_gaussians[None] - 1):
        diff = gaussian_pos[i] - gaussian_pos[i + 1]
        rest_diff = gaussian_rest_pos[i] - gaussian_rest_pos[i + 1]
        deformation = diff - rest_diff
        energy += deformation.norm_sqr()
    
    return energy / num_gaussians[None]


@ti.kernel
def compute_overlap_count() -> ti.i32:
    """计算重叠数量（伪影程度）"""
    count = 0
    threshold = 0.005  # 5mm
    
    # 简化的重叠检测：检查相邻点距离
    for i in range(num_gaussians[None] - 1):
        dist = (gaussian_pos[i] - gaussian_pos[i + 1]).norm()
        if dist < threshold:
            count += 1
    
    return count


@ti.kernel
def extract_frame_data() -> ti.types.ndarray():
    """提取当前帧数据"""
    # 计算质心
    center = ti.Vector([0.0, 0.0, 0.0])
    for i in range(num_gaussians[None]):
        center += gaussian_pos[i]
    center /= num_gaussians[None]
    
    # 计算平均速度
    avg_vel = ti.Vector([0.0, 0.0, 0.0])
    for i in range(num_gaussians[None]):
        avg_vel += gaussian_vel[i]
    avg_vel /= num_gaussians[None]
    
    # 计算最大位移
    max_disp = 0.0
    for i in range(num_gaussians[None]):
        disp = (gaussian_pos[i] - gaussian_rest_pos[i]).norm()
        if disp > max_disp:
            max_disp = disp
    
    # 计算动能
    kinetic = 0.0
    for i in range(num_gaussians[None]):
        mass = 1.0 / gaussian_inv_mass[i]
        kinetic += 0.5 * mass * gaussian_vel[i].norm_sqr()
    
    vol_ratio = compute_volume_ratio()
    lap_energy = compute_laplacian_energy()
    overlap = compute_overlap_count()
    
    data = np.zeros(10, dtype=np.float32)
    data[0] = center[0]
    data[1] = center[1]
    data[2] = center[2]
    data[3] = avg_vel[0]
    data[4] = avg_vel[1]
    data[5] = avg_vel[2]
    data[6] = max_disp
    data[7] = kinetic
    data[8] = vol_ratio
    data[9] = lap_energy
    
    return data, overlap


# ============ 仿真循环 ============

def run_simulation(method, force_config):
    """运行单个仿真"""
    method_map = {"LBS": 0, "PartMM": 1, "SparseCP": 2, "Direct": 3}
    current_method[None] = method_map[method]
    current_force[None] = force_config["magnitude"]
    
    dt = CONFIG["simulation"]["dt"]
    gravity = CONFIG["simulation"]["gravity"]
    fps = CONFIG["simulation"]["fps"]
    duration = CONFIG["simulation"]["duration"]
    num_frames = int(fps * duration)
    
    # 初始化
    init_cylinder_gaussians(
        CONFIG["cylinder"]["radius"],
        CONFIG["cylinder"]["height"],
        CONFIG["cylinder"]["num_gaussians"]
    )
    
    if method == "LBS":
        init_bones()
    elif method == "PartMM":
        init_parts()
    elif method == "SparseCP":
        init_control_points()
    
    # 运行仿真
    frame_data = []
    start_time = time.time()
    
    for frame in range(num_frames):
        # 施加外力
        apply_external_force(force_config["magnitude"], dt)
        
        # 根据方法更新
        if method == "LBS":
            update_bones(force_config["magnitude"], dt)
            lbs_update()
        elif method == "PartMM":
            update_parts(force_config["magnitude"], dt)
            partmm_update()
        elif method == "SparseCP":
            update_control_points(force_config["magnitude"], dt)
            sparsecp_update()
        elif method == "Direct":
            direct_predict(dt, gravity)
            direct_solve_constraints(dt)
            direct_update_velocities(dt)
        
        # 每10帧提取一次数据
        if frame % 10 == 0:
            data, overlap = extract_frame_data()
            frame_data.append({
                "frame": frame,
                "time": frame / fps,
                "center_x": float(data[0]),
                "center_y": float(data[1]),
                "center_z": float(data[2]),
                "velocity_x": float(data[3]),
                "velocity_y": float(data[4]),
                "velocity_z": float(data[5]),
                "max_displacement": float(data[6]),
                "kinetic_energy": float(data[7]),
                "volume_ratio": float(data[8]),
                "laplacian_energy": float(data[9]),
                "overlap_count": int(overlap),
            })
    
    elapsed = time.time() - start_time
    
    return frame_data, elapsed


# ============ 批量运行 ============

def run_all_experiments():
    """运行所有实验"""
    os.makedirs(f"{OUTPUT_DIR}/data", exist_ok=True)
    
    experiments = []
    results = []
    
    for method in CONFIG["binding_methods"]:
        for force in CONFIG["force_levels"]:
            exp_name = f"{method}_{force['name']}"
            print(f"运行实验: {exp_name}")
            
            experiments.append({
                "name": exp_name,
                "method": method,
                "force_level": force["name"],
                "force_magnitude": force["magnitude"],
            })
            
            # 运行仿真
            frame_data, elapsed = run_simulation(method, force)
            
            # 保存数据
            import csv
            csv_path = f"{OUTPUT_DIR}/data/{exp_name}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=frame_data[0].keys())
                writer.writeheader()
                writer.writerows(frame_data)
            
            # 计算统计
            avg_volume_ratio = sum(d["volume_ratio"] for d in frame_data) / len(frame_data)
            avg_laplacian = sum(d["laplacian_energy"] for d in frame_data) / len(frame_data)
            avg_overlap = sum(d["overlap_count"] for d in frame_data) / len(frame_data)
            
            results.append({
                "experiment": exp_name,
                "method": method,
                "force_level": force["name"],
                "simulation_time": elapsed,
                "avg_volume_ratio": avg_volume_ratio,
                "avg_laplacian_energy": avg_laplacian,
                "avg_overlap_count": avg_overlap,
            })
            
            print(f"  完成，耗时: {elapsed:.2f}秒")
    
    # 保存实验配置
    with open(f"{OUTPUT_DIR}/experiments.json", "w") as f:
        json.dump(experiments, f, indent=2)
    
    # 保存运行统计
    with open(f"{OUTPUT_DIR}/data/simulation_stats.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n所有实验完成！")
    print(f"数据保存在: {OUTPUT_DIR}")
    
    return experiments, results


if __name__ == "__main__":
    print("=" * 60)
    print("3DGS绑定方式对比实验 - Taichi实现")
    print("=" * 60)
    
    experiments, results = run_all_experiments()
    
    print("\n下一步:")
    print("1. 将 results/ 目录复制到分析环境")
    print("2. 运行 python analyze_binding.py 分析数据")
