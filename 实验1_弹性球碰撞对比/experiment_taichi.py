"""
弹性球碰撞对比实验 - Taichi实现
开源方案，支持GPU加速

求解器：PBD / XPBD / FEM
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
    "solvers": ["PBD", "XPBD", "FEM"],
    "stiffness": [
        {"name": "soft", "youngs_modulus": 1e4, "desc": "果冻/橡胶"},
        {"name": "medium_soft", "youngs_modulus": 1e5, "desc": "网球"},
        {"name": "medium", "youngs_modulus": 1e6, "desc": "篮球"},
        {"name": "hard", "youngs_modulus": 1e7, "desc": "高尔夫球"},
    ],
    "drop_heights": [
        {"name": "low", "height": 0.5},
        {"name": "medium", "height": 1.0},
        {"name": "high", "height": 2.0},
    ],
    "ball": {
        "radius": 0.1,
        "mass": 0.5,
        "num_particles": 1000,  # 粒子数量
    },
    "simulation": {
        "fps": 240,
        "duration": 3.0,
        "gravity": -9.81,
        "dt": 1.0 / 240.0,
    },
}

OUTPUT_DIR = "D:/Experiment1_Taichi/results"


# ============ 粒子系统 ============

@ti.dataclass
class Particle:
    pos: ti.types.vector(3, ti.f32)      # 位置
    vel: ti.types.vector(3, ti.f32)      # 速度
    pred_pos: ti.types.vector(3, ti.f32) # 预测位置
    mass: ti.f32                          # 质量
    fixed: ti.i32                         # 是否固定

# 粒子数组
particles = ti.Struct.field(Particle, shape=(10000,))

# 弹簧连接（用于FEM）
springs = ti.field(ti.i32, shape=(50000, 2))
spring_rest_length = ti.field(ti.f32, shape=(50000,))
spring_stiffness = ti.field(ti.f32, shape=(50000,))
num_springs = ti.field(ti.i32, shape=())

# 实验参数
current_solver = ti.field(ti.i32, shape=())
current_stiffness = ti.field(ti.f32, shape=())
ground_y = ti.field(ti.f32, shape=())
num_particles = ti.field(ti.i32, shape=())


# ============ 初始化 ============

@ti.kernel
def init_ball_particles(radius: ti.f32, center_y: ti.f32, num_pts: ti.i32):
    """初始化球体粒子"""
    num_particles[None] = num_pts
    
    # 在球体内均匀分布粒子
    for i in range(num_pts):
        # 使用Fibonacci球面分布
        phi = ti.acos(1.0 - 2.0 * (i + 0.5) / num_pts)
        theta = ti.pi * (1.0 + ti.sqrt(5.0)) * (i + 0.5)
        
        # 径向分布（0到radius）
        r = radius * ti.pow(ti.random(), 1.0/3.0)
        
        # 转换为笛卡尔坐标
        x = r * ti.sin(phi) * ti.cos(theta)
        y = r * ti.sin(phi) * ti.sin(theta) + center_y
        z = r * ti.cos(phi)
        
        particles[i].pos = ti.Vector([x, y, z])
        particles[i].vel = ti.Vector([0.0, 0.0, 0.0])
        particles[i].pred_pos = particles[i].pos
        particles[i].mass = CONFIG["ball"]["mass"] / num_pts
        particles[i].fixed = 0


@ti.kernel
def init_springs(radius: ti.f32, stiffness: ti.f32):
    """初始化弹簧连接（用于FEM）"""
    spring_idx = 0
    max_springs = 50000
    
    # 为相邻粒子创建弹簧
    for i in range(num_particles[None]):
        for j in range(i + 1, num_particles[None]):
            if spring_idx >= max_springs:
                break
            
            dist = (particles[i].pos - particles[j].pos).norm()
            # 连接距离小于2*radius/10的粒子
            if dist < radius * 0.3:
                springs[spring_idx, 0] = i
                springs[spring_idx, 1] = j
                spring_rest_length[spring_idx] = dist
                spring_stiffness[spring_idx] = stiffness
                spring_idx += 1
    
    num_springs[None] = spring_idx


# ============ PBD求解器 ============

@ti.kernel
def pbd_predict(dt: ti.f32, gravity: ti.f32):
    """PBD: 预测位置"""
    for i in range(num_particles[None]):
        if particles[i].fixed == 0:
            # 应用重力
            particles[i].vel[1] += gravity * dt
            # 预测位置
            particles[i].pred_pos = particles[i].pos + particles[i].vel * dt


@ti.kernel
def pbd_solve_constraints(dt: ti.f32, stiffness: ti.f32):
    """PBD: 求解约束"""
    # 地面碰撞约束
    for i in range(num_particles[None]):
        if particles[i].pred_pos[1] < 0.0:
            # 地面约束
            particles[i].pred_pos[1] = 0.0
    
    # 形状约束（保持球体形状）
    # 计算质心
    center = ti.Vector([0.0, 0.0, 0.0])
    for i in range(num_particles[None]):
        center += particles[i].pred_pos
    center /= num_particles[None]
    
    # 约束粒子到球体表面
    target_radius = CONFIG["ball"]["radius"]
    for i in range(num_particles[None]):
        diff = particles[i].pred_pos - center
        dist = diff.norm()
        if dist > 0.001:
            # 根据刚度调整约束强度
            correction = (dist - target_radius) * stiffness
            particles[i].pred_pos -= diff.normalized() * correction


@ti.kernel
def pbd_update_velocities(dt: ti.f32):
    """PBD: 更新速度"""
    for i in range(num_particles[None]):
        particles[i].vel = (particles[i].pred_pos - particles[i].pos) / dt
        particles[i].pos = particles[i].pred_pos


# ============ XPBD求解器 ============

@ti.kernel
def xpbd_predict(dt: ti.f32, gravity: ti.f32):
    """XPBD: 预测位置"""
    for i in range(num_particles[None]):
        if particles[i].fixed == 0:
            particles[i].vel[1] += gravity * dt
            particles[i].pred_pos = particles[i].pos + particles[i].vel * dt


@ti.kernel
def xpbd_solve_constraints(dt: ti.f32, compliance: ti.f32):
    """XPBD: 求解约束（使用compliance）"""
    # 地面碰撞
    for i in range(num_particles[None]):
        if particles[i].pred_pos[1] < 0.0:
            particles[i].pred_pos[1] = 0.0
    
    # 形状约束（XPBD使用compliance）
    center = ti.Vector([0.0, 0.0, 0.0])
    for i in range(num_particles[None]):
        center += particles[i].pred_pos
    center /= num_particles[None]
    
    target_radius = CONFIG["ball"]["radius"]
    
    for i in range(num_particles[None]):
        diff = particles[i].pred_pos - center
        dist = diff.norm()
        if dist > 0.001:
            # XPBD约束
            C = dist - target_radius
            # compliance是刚度的倒数
            alpha = compliance
            # 计算拉格朗日乘子
            grad_C = diff.normalized()
            w = 1.0 / particles[i].mass
            delta_lambda = -C / (grad_C.norm_sqr() * w + alpha)
            
            # 应用约束
            correction = delta_lambda * grad_C * w
            particles[i].pred_pos += correction


@ti.kernel
def xpbd_update_velocities(dt: ti.f32):
    """XPBD: 更新速度"""
    for i in range(num_particles[None]):
        particles[i].vel = (particles[i].pred_pos - particles[i].pos) / dt
        particles[i].pos = particles[i].pred_pos


# ============ FEM求解器 ============

@ti.kernel
def fem_apply_forces(dt: ti.f32, gravity: ti.f32, youngs_modulus: ti.f32):
    """FEM: 计算并应用力"""
    # 重置力
    forces = ti.Vector.field(3, ti.f32, shape=(10000,))
    for i in range(10000):
        forces[i] = ti.Vector([0.0, 0.0, 0.0])
    
    # 重力
    for i in range(num_particles[None]):
        forces[i][1] += particles[i].mass * gravity
    
    # 弹簧力（FEM简化为弹簧系统）
    for s in range(num_springs[None]):
        i = springs[s, 0]
        j = springs[s, 1]
        
        diff = particles[i].pos - particles[j].pos
        dist = diff.norm()
        if dist > 0.001:
            # 胡克定律
            direction = diff.normalized()
            delta = dist - spring_rest_length[s]
            force = spring_stiffness[s] * delta * direction
            
            forces[i] -= force
            forces[j] += force
    
    # 更新速度和位置（显式欧拉）
    for i in range(num_particles[None]):
        if particles[i].fixed == 0:
            acc = forces[i] / particles[i].mass
            particles[i].vel += acc * dt
            particles[i].pos += particles[i].vel * dt


@ti.kernel
def fem_solve_collision():
    """FEM: 碰撞处理"""
    for i in range(num_particles[None]):
        if particles[i].pos[1] < 0.0:
            particles[i].pos[1] = 0.0
            # 反弹
            particles[i].vel[1] = -particles[i].vel[1] * 0.5


# ============ 数据提取 ============

@ti.kernel
def extract_frame_data() -> ti.types.ndarray():
    """提取当前帧数据"""
    # 计算质心
    center = ti.Vector([0.0, 0.0, 0.0])
    for i in range(num_particles[None]):
        center += particles[i].pos
    center /= num_particles[None]
    
    # 计算平均速度
    avg_vel = ti.Vector([0.0, 0.0, 0.0])
    for i in range(num_particles[None]):
        avg_vel += particles[i].vel
    avg_vel /= num_particles[None]
    
    # 计算最大形变
    max_deform = 0.0
    for i in range(num_particles[None]):
        dist = (particles[i].pos - center).norm()
        deform = dist - CONFIG["ball"]["radius"]
        if deform > max_deform:
            max_deform = deform
    
    # 计算能量
    mass = CONFIG["ball"]["mass"]
    kinetic = 0.5 * mass * avg_vel.norm_sqr()
    potential = mass * abs(CONFIG["simulation"]["gravity"]) * center[1]
    elastic = 0.5 * current_stiffness[None] * max_deform * max_deform
    
    data = np.zeros(11, dtype=np.float32)
    data[0] = center[0]
    data[1] = center[1]
    data[2] = center[2]
    data[3] = avg_vel[0]
    data[4] = avg_vel[1]
    data[5] = avg_vel[2]
    data[6] = max_deform
    data[7] = kinetic
    data[8] = potential
    data[9] = elastic
    data[10] = kinetic + potential + elastic
    
    return data


# ============ 仿真循环 ============

def run_simulation(solver_type, stiffness_config, height_config):
    """运行单个仿真"""
    solver_map = {"PBD": 0, "XPBD": 1, "FEM": 2}
    current_solver[None] = solver_map[solver_type]
    current_stiffness[None] = stiffness_config["youngs_modulus"]
    ground_y[None] = 0.0
    
    dt = CONFIG["simulation"]["dt"]
    gravity = CONFIG["simulation"]["gravity"]
    fps = CONFIG["simulation"]["fps"]
    duration = CONFIG["simulation"]["duration"]
    num_frames = int(fps * duration)
    
    # 初始化球体
    init_ball_particles(
        CONFIG["ball"]["radius"],
        height_config["height"] + CONFIG["ball"]["radius"],
        CONFIG["ball"]["num_particles"]
    )
    
    # 如果是FEM，初始化弹簧
    if solver_type == "FEM":
        init_springs(CONFIG["ball"]["radius"], stiffness_config["youngs_modulus"])
    
    # 运行仿真
    frame_data = []
    start_time = time.time()
    
    for frame in range(num_frames):
        # 根据求解器类型执行不同的步骤
        if solver_type == "PBD":
            pbd_predict(dt, gravity)
            # 迭代求解约束
            for _ in range(10):
                pbd_solve_constraints(dt, min(1.0, np.log10(stiffness_config["youngs_modulus"]) / 10.0))
            pbd_update_velocities(dt)
        
        elif solver_type == "XPBD":
            xpbd_predict(dt, gravity)
            # 迭代求解约束
            compliance = 1.0 / stiffness_config["youngs_modulus"]
            for _ in range(20):
                xpbd_solve_constraints(dt, compliance)
            xpbd_update_velocities(dt)
        
        elif solver_type == "FEM":
            fem_apply_forces(dt, gravity, stiffness_config["youngs_modulus"])
            fem_solve_collision()
        
        # 每10帧提取一次数据
        if frame % 10 == 0:
            data = extract_frame_data()
            frame_data.append({
                "frame": frame,
                "time": frame / fps,
                "center_x": float(data[0]),
                "center_y": float(data[1]),
                "center_z": float(data[2]),
                "velocity_x": float(data[3]),
                "velocity_y": float(data[4]),
                "velocity_z": float(data[5]),
                "max_deformation": float(data[6]),
                "kinetic_energy": float(data[7]),
                "potential_energy": float(data[8]),
                "elastic_energy": float(data[9]),
                "total_energy": float(data[10]),
            })
    
    elapsed = time.time() - start_time
    
    return frame_data, elapsed


# ============ 批量运行 ============

def run_all_experiments():
    """运行所有实验"""
    os.makedirs(f"{OUTPUT_DIR}/data", exist_ok=True)
    
    experiments = []
    results = []
    
    for solver in CONFIG["solvers"]:
        for stiffness in CONFIG["stiffness"]:
            for height in CONFIG["drop_heights"]:
                exp_name = f"{solver}_{stiffness['name']}_h{height['name']}"
                print(f"运行实验: {exp_name}")
                
                experiments.append({
                    "name": exp_name,
                    "solver": solver,
                    "stiffness": stiffness["name"],
                    "height": height["name"],
                    "youngs_modulus": stiffness["youngs_modulus"],
                    "drop_height": height["height"],
                })
                
                # 运行仿真
                frame_data, elapsed = run_simulation(solver, stiffness, height)
                
                # 保存数据
                import csv
                csv_path = f"{OUTPUT_DIR}/data/{exp_name}.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=frame_data[0].keys())
                    writer.writeheader()
                    writer.writerows(frame_data)
                
                results.append({
                    "experiment": exp_name,
                    "solver": solver,
                    "stiffness": stiffness["name"],
                    "height": height["name"],
                    "simulation_time": elapsed,
                })
                
                print(f"  完成，耗时: {elapsed:.2f}秒")
    
    # 保存实验配置
    with open(f"{OUTPUT_DIR}/experiments.json", "w") as f:
        json.dump(experiments, f, indent=2)
    
    # 保存运行统计
    with open(f"{OUTPUT_DIR}/data/simulation_times.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n所有实验完成！")
    print(f"数据保存在: {OUTPUT_DIR}")
    
    return experiments, results


if __name__ == "__main__":
    print("=" * 60)
    print("弹性球碰撞对比实验 - Taichi实现")
    print("=" * 60)
    
    experiments, results = run_all_experiments()
    
    print("\n下一步:")
    print("1. 将 results/ 目录复制到分析环境")
    print("2. 运行 python analyze_results.py 分析数据")
