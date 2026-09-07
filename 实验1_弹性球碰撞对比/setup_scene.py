"""
Houdini场景搭建脚本 - 弹性球碰撞对比实验
在Houdini Python Shell或hython中运行

用法：
1. 在Houdini中打开Python Shell
2. 执行：exec(open('setup_scene.py').read())
3. 或批量运行：hython batch_run.py
"""

import hou
import json
import os
import time
import math

# ============ 实验参数配置 ============

EXPERIMENT_CONFIG = {
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
        "radius": 0.1,  # 米
        "mass": 0.5,  # kg
        "resolution": 32,  # 网格细分
    },
    "simulation": {
        "fps": 240,
        "duration": 3.0,  # 秒
        "gravity": -9.81,
    },
}

OUTPUT_DIR = "D:/Experiment1/results"


# ============ 场景搭建函数 ============

def clear_scene():
    """清空场景"""
    hou.node("/obj").destroyChildren()
    hou.node("/out").destroyChildren()


def create_ground():
    """创建地面（刚性平面）"""
    geo = hou.node("/obj").createNode("geo", "ground")
    
    # 创建大平面
    grid = geo.createNode("grid", "ground_grid")
    grid.parm("sizex").set(10)
    grid.parm("sizey").set(10)
    grid.parm("rows").set(1)
    grid.parm("cols").set(1)
    
    # 移动到地面位置
    xform = geo.createNode("xform", "position")
    xform.setInput(0, grid)
    xform.parm("ty").set(0)
    
    # 设置为静态刚体
    ground_node = xform
    
    return ground_node


def create_ball(height, resolution=32):
    """创建球体"""
    geo = hou.node("/obj").createNode("geo", "ball")
    
    # 创建球
    sphere = geo.createNode("sphere", "ball_sphere")
    sphere.parm("type").set(0)  # 多边形球
    sphere.parm("radx").set(EXPERIMENT_CONFIG["ball"]["radius"])
    sphere.parm("rady").set(EXPERIMENT_CONFIG["ball"]["radius"])
    sphere.parm("radz").set(EXPERIMENT_CONFIG["ball"]["radius"])
    sphere.parm("freqx").set(resolution)
    sphere.parm("freqy").set(resolution)
    
    # 设置初始高度
    xform = geo.createNode("xform", "position")
    xform.setInput(0, sphere)
    xform.parm("ty").set(height)
    
    return xform


def setup_pbd_solver(ball_geo, ground_geo, stiffness_config):
    """设置PBD求解器（Vellum）"""
    # 创建Vellum Sim网络
    sim = hou.node("/obj").createNode("dopnet", "pbd_sim")
    
    # 进入DOP网络
    sim_path = sim.path()
    
    # 创建Vellum Config
    vellum_config = sim.createNode("vellumconfig", "vellum_config")
    vellum_config.parm("solvertype").set(0)  # PBD
    
    # 创建Vellum Constraints
    constraints = sim.createNode("vellumconstraints", "constraints")
    constraints.parm("constrainttype").set("Soft")  # 软体约束
    
    # 设置刚度参数
    youngs = stiffness_config["youngs_modulus"]
    # PBD使用stiffness参数（0-1范围）
    # 将杨氏模量映射到PBD stiffness
    pbd_stiffness = min(1.0, math.log10(youngs) / 10.0)
    constraints.parm("stiffness").set(pbd_stiffness)
    
    # 创建地面碰撞
    ground_coll = sim.createNode("vellumgroundplane", "ground")
    ground_coll.parm("collisiony").set(0)
    
    # 连接
    vellum_config.setInput(0, constraints)
    
    return sim


def setup_xpbd_solver(ball_geo, ground_geo, stiffness_config):
    """设置XPBD求解器（Vellum with XPBD settings）"""
    # Houdini Vellum可以通过参数调整模拟XPBD行为
    sim = hou.node("/obj").createNode("dopnet", "xpbd_sim")
    
    # 创建Vellum Config
    vellum_config = sim.createNode("vellumconfig", "vellum_config")
    vellum_config.parm("solvertype").set(1)  # XPBD模式（如果支持）
    
    # 如果Houdini版本不支持XPBD，使用PBD但调整参数
    vellum_config.parm("substeps").set(10)  # 增加子步长提高精度
    vellum_config.parm("iterations").set(20)  # 增加迭代次数
    
    # 创建约束
    constraints = sim.createNode("vellumconstraints", "constraints")
    constraints.parm("constrainttype").set("Soft")
    
    youngs = stiffness_config["youngs_modulus"]
    # XPBD使用compliance（柔度），是刚度的倒数
    compliance = 1.0 / youngs
    constraints.parm("compliance").set(compliance)
    
    # 地面
    ground_coll = sim.createNode("vellumgroundplane", "ground")
    ground_coll.parm("collisiony").set(0)
    
    vellum_config.setInput(0, constraints)
    
    return sim


def setup_fem_solver(ball_geo, ground_geo, stiffness_config):
    """设置FEM求解器"""
    sim = hou.node("/obj").createNode("dopnet", "fem_sim")
    
    # 创建FEM Config
    fem_config = sim.createNode("femconfig", "fem_config")
    
    # 设置材料属性
    youngs = stiffness_config["youngs_modulus"]
    fem_config.parm("youngsmodulus").set(youngs)
    fem_config.parm("poissonsratio").set(0.3)  # 泊松比
    fem_config.parm("density").set(1000)  # 密度 kg/m³
    
    # 创建地面碰撞
    ground_coll = sim.createNode("femgroundplane", "ground")
    ground_coll.parm("collisiony").set(0)
    
    return sim


def setup_data_export(sim_node, experiment_name):
    """设置数据导出"""
    # 创建Output节点导出每帧数据
    out = hou.node("/out").createNode("geometry", experiment_name + "_output")
    
    # 设置输出路径
    output_path = f"{OUTPUT_DIR}/raw/{experiment_name}/ball_$F4.bgeo"
    out.parm("sopoutput").set(output_path)
    
    # 设置帧范围
    fps = EXPERIMENT_CONFIG["simulation"]["fps"]
    duration = EXPERIMENT_CONFIG["simulation"]["duration"]
    end_frame = int(fps * duration)
    
    out.parm("f1").set(1)
    out.parm("f2").set(end_frame)
    out.parm("f3").set(1)
    
    return out


def setup_experiment(solver_type, stiffness_config, height_config):
    """搭建单个实验场景"""
    # 清空场景
    clear_scene()
    
    # 创建实验名称
    exp_name = f"{solver_type}_{stiffness_config['name']}_h{height_config['name']}"
    
    print(f"搭建实验: {exp_name}")
    
    # 创建地面
    ground = create_ground()
    
    # 创建球体
    ball = create_ball(height_config["height"], EXPERIMENT_CONFIG["ball"]["resolution"])
    
    # 设置求解器
    if solver_type == "PBD":
        sim = setup_pbd_solver(ball, ground, stiffness_config)
    elif solver_type == "XPBD":
        sim = setup_xpbd_solver(ball, ground, stiffness_config)
    elif solver_type == "FEM":
        sim = setup_fem_solver(ball, ground, stiffness_config)
    
    # 设置数据导出
    output = setup_data_export(sim, exp_name)
    
    # 设置渲染
    setup_render(exp_name)
    
    return exp_name


def setup_render(experiment_name):
    """设置渲染"""
    # 创建摄像机
    cam = hou.node("/obj").createNode("cam", "camera")
    cam.parmTuple("t").set((2, 1, 2))
    cam.parmTuple("r").set((-20, 45, 0))
    
    # 创建灯光
    light = hou.node("/obj").createNode("light", "key_light")
    light.parmTuple("t").set((3, 3, 3))
    
    # 设置ROP输出
    rop = hou.node("/out").createNode("mantra", "render")
    rop.parm("camera").set(cam.path())
    
    # 设置帧范围
    fps = EXPERIMENT_CONFIG["simulation"]["fps"]
    duration = EXPERIMENT_CONFIG["simulation"]["duration"]
    end_frame = int(fps * duration)
    
    rop.parm("f1").set(1)
    rop.parm("f2").set(end_frame)
    rop.parm("f3").set(1)
    
    # 输出路径
    output_path = f"{OUTPUT_DIR}/renders/{experiment_name}/frame_$F4.exr"
    rop.parm("vm_picture").set(output_path)


# ============ 批量运行 ============

def run_all_experiments():
    """运行所有实验"""
    # 创建输出目录
    os.makedirs(f"{OUTPUT_DIR}/raw", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/renders", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/data", exist_ok=True)
    
    experiments = []
    
    for solver in EXPERIMENT_CONFIG["solvers"]:
        for stiffness in EXPERIMENT_CONFIG["stiffness"]:
            for height in EXPERIMENT_CONFIG["drop_heights"]:
                exp_name = setup_experiment(solver, stiffness, height)
                experiments.append({
                    "name": exp_name,
                    "solver": solver,
                    "stiffness": stiffness["name"],
                    "height": height["name"],
                    "youngs_modulus": stiffness["youngs_modulus"],
                    "drop_height": height["height"],
                })
    
    # 保存实验配置
    config_path = f"{OUTPUT_DIR}/experiments.json"
    with open(config_path, "w") as f:
        json.dump(experiments, f, indent=2)
    
    print(f"共搭建 {len(experiments)} 个实验场景")
    print(f"配置已保存到: {config_path}")
    
    return experiments


def run_simulation(experiment_name):
    """运行单个实验的仿真"""
    print(f"开始仿真: {experiment_name}")
    start_time = time.time()
    
    # 设置帧范围
    fps = EXPERIMENT_CONFIG["simulation"]["fps"]
    duration = EXPERIMENT_CONFIG["simulation"]["duration"]
    end_frame = int(fps * duration)
    
    # 运行仿真
    hou.playbar.setFrameRange(1, end_frame)
    hou.playbar.play()
    
    # 等待完成
    while hou.playbar.state() == hou.playState.Playing:
        hou.ui.displayMessage("仿真运行中...", severity=hou.severityType.Message, title="仿真")
    
    elapsed = time.time() - start_time
    print(f"仿真完成: {experiment_name}, 耗时: {elapsed:.2f}秒")
    
    return elapsed


def batch_run_simulations():
    """批量运行所有仿真"""
    # 加载实验配置
    config_path = f"{OUTPUT_DIR}/experiments.json"
    with open(config_path, "r") as f:
        experiments = json.load(f)
    
    results = []
    
    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] 运行实验: {exp['name']}")
        
        # 加载场景（这里需要根据实际情况调整）
        # setup_experiment(exp['solver'], ...)
        
        # 运行仿真
        elapsed = run_simulation(exp['name'])
        
        results.append({
            "experiment": exp['name'],
            "solver": exp['solver'],
            "stiffness": exp['stiffness'],
            "height": exp['height'],
            "simulation_time": elapsed,
        })
    
    # 保存运行结果
    results_path = f"{OUTPUT_DIR}/data/simulation_times.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n所有仿真完成，结果保存到: {results_path}")
    
    return results


# ============ 数据提取 ============

def extract_frame_data(frame_num):
    """提取单帧数据"""
    # 获取球体几何
    ball_geo = hou.node("/obj/ball/position")
    if not ball_geo:
        return None
    
    geo = ball_geo.geometry()
    
    # 计算球心位置
    points = geo.points()
    if len(points) == 0:
        return None
    
    center = hou.Vector3(0, 0, 0)
    for pt in points:
        center += pt.position()
    center /= len(points)
    
    # 计算最大形变（从球心的最大距离）
    max_deform = 0
    for pt in points:
        dist = (pt.position() - center).length()
        max_deform = max(max_deform, dist)
    
    # 理论半径
    theoretical_radius = EXPERIMENT_CONFIG["ball"]["radius"]
    deformation = max_deform - theoretical_radius
    
    # 获取速度（如果有）
    velocity = hou.Vector3(0, 0, 0)
    if geo.findPointAttrib("v"):
        for pt in points:
            velocity += pt.attribValue("v")
        velocity /= len(points)
    
    # 计算能量
    mass = EXPERIMENT_CONFIG["ball"]["mass"]
    gravity = abs(EXPERIMENT_CONFIG["simulation"]["gravity"])
    
    kinetic_energy = 0.5 * mass * velocity.length() ** 2
    potential_energy = mass * gravity * center.y()
    # 弹性势能估算（简化）
    elastic_energy = 0.5 * 1e5 * deformation ** 2  # 使用平均刚度
    
    return {
        "frame": frame_num,
        "time": frame_num / EXPERIMENT_CONFIG["simulation"]["fps"],
        "center_x": center.x(),
        "center_y": center.y(),
        "center_z": center.z(),
        "velocity_x": velocity.x(),
        "velocity_y": velocity.y(),
        "velocity_z": velocity.z(),
        "max_deformation": deformation,
        "kinetic_energy": kinetic_energy,
        "potential_energy": potential_energy,
        "elastic_energy": elastic_energy,
        "total_energy": kinetic_energy + potential_energy + elastic_energy,
    }


def extract_all_data(experiment_name):
    """提取单个实验的所有帧数据"""
    import csv
    
    fps = EXPERIMENT_CONFIG["simulation"]["fps"]
    duration = EXPERIMENT_CONFIG["simulation"]["duration"]
    end_frame = int(fps * duration)
    
    data = []
    
    for frame in range(1, end_frame + 1):
        hou.setFrame(frame)
        frame_data = extract_frame_data(frame)
        if frame_data:
            data.append(frame_data)
    
    # 保存为CSV
    if data:
        output_path = f"{OUTPUT_DIR}/data/{experiment_name}.csv"
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        print(f"数据已保存: {output_path}")
    
    return data


def batch_extract_data():
    """批量提取所有实验数据"""
    config_path = f"{OUTPUT_DIR}/experiments.json"
    with open(config_path, "r") as f:
        experiments = json.load(f)
    
    for exp in experiments:
        print(f"提取数据: {exp['name']}")
        extract_all_data(exp['name'])


# ============ 主函数 ============

if __name__ == "__main__":
    print("=" * 50)
    print("弹性球碰撞对比实验 - Houdini场景搭建")
    print("=" * 50)
    
    # 运行所有实验
    experiments = run_all_experiments()
    
    print("\n下一步:")
    print("1. 运行 batch_run_simulations() 执行仿真")
    print("2. 运行 batch_extract_data() 提取数据")
    print("3. 在另一个环境运行 analyze_results.py 分析数据")
