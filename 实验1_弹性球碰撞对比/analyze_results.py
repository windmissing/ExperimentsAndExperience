"""
数据分析脚本 - 弹性球碰撞对比实验
独立于Houdini，只需要matplotlib和pandas

用法：
python analyze_results.py
"""

import os
import json
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============ 配置 ============

OUTPUT_DIR = "D:/Experiment1/results"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"
PROCESSED_DIR = f"{OUTPUT_DIR}/processed"

# 实验参数
BALL_RADIUS = 0.1  # 米
BALL_MASS = 0.5  # kg
GRAVITY = 9.81  # m/s²
FPS = 240


# ============ 数据加载 ============

def load_experiment_data(experiment_name):
    """加载单个实验的CSV数据"""
    csv_path = f"{OUTPUT_DIR}/data/{experiment_name}.csv"
    if not os.path.exists(csv_path):
        print(f"警告: 文件不存在 {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    return df


def load_all_experiments():
    """加载所有实验数据"""
    config_path = f"{OUTPUT_DIR}/experiments.json"
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在 {config_path}")
        return []
    
    with open(config_path, "r") as f:
        experiments = json.load(f)
    
    all_data = {}
    for exp in experiments:
        df = load_experiment_data(exp['name'])
        if df is not None:
            all_data[exp['name']] = {
                'config': exp,
                'data': df,
            }
    
    return all_data


# ============ 指标计算 ============

def calculate_bounce_metrics(df):
    """计算弹跳相关指标"""
    # 找到球心Y坐标的局部最大值（弹跳峰值）
    y = df['center_y'].values
    t = df['time'].values
    
    # 简单的峰值检测
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i-1] and y[i] > y[i+1] and y[i] > BALL_RADIUS * 1.1:
            peaks.append({
                'time': t[i],
                'height': y[i],
            })
    
    # 计算弹跳衰减率
    if len(peaks) >= 2:
        bounce_ratios = []
        for i in range(1, len(peaks)):
            ratio = peaks[i]['height'] / peaks[i-1]['height']
            bounce_ratios.append(ratio)
        avg_bounce_ratio = np.mean(bounce_ratios)
    else:
        avg_bounce_ratio = 0
    
    # 计算接触时间（球心Y < 1.1 * 半径的时间）
    contact_mask = y < BALL_RADIUS * 1.1
    contact_frames = np.sum(contact_mask)
    contact_time = contact_frames / FPS
    
    return {
        'num_bounces': len(peaks),
        'avg_bounce_ratio': avg_bounce_ratio,
        'contact_time': contact_time,
        'peaks': peaks,
    }


def calculate_energy_metrics(df):
    """计算能量相关指标"""
    total_energy = df['total_energy'].values
    initial_energy = total_energy[0]
    final_energy = total_energy[-1]
    
    # 能量损失率
    energy_loss = (initial_energy - final_energy) / initial_energy * 100
    
    # 能量耗散率（每帧）
    energy_diff = np.diff(total_energy)
    avg_dissipation = np.mean(np.abs(energy_diff))
    
    # 能量稳定性（标准差）
    energy_std = np.std(total_energy)
    
    return {
        'initial_energy': initial_energy,
        'final_energy': final_energy,
        'energy_loss_percent': energy_loss,
        'avg_dissipation_per_frame': avg_dissipation,
        'energy_std': energy_std,
    }


def calculate_deformation_metrics(df):
    """计算形变相关指标"""
    deform = df['max_deformation'].values
    
    max_deform = np.max(deform)
    avg_deform = np.mean(deform)
    
    # 形变恢复率（最大形变后恢复到初始的百分比）
    if max_deform > 0:
        final_deform = deform[-1]
        recovery_ratio = (max_deform - final_deform) / max_deform * 100
    else:
        recovery_ratio = 100
    
    return {
        'max_deformation': max_deform,
        'avg_deformation': avg_deform,
        'recovery_ratio': recovery_ratio,
    }


def calculate_stability_metrics(df):
    """计算稳定性指标"""
    # 检测数值爆炸（位置异常）
    y = df['center_y'].values
    explosion = np.any(np.abs(y) > 10)  # 如果Y超过10米认为爆炸
    
    # 检测穿模（球心Y < 半径）
    penetration = np.any(y < BALL_RADIUS * 0.9)
    max_penetration = BALL_RADIUS - np.min(y) if penetration else 0
    
    # 检测NaN
    has_nan = df.isnull().any().any()
    
    return {
        'explosion': explosion,
        'penetration': penetration,
        'max_penetration_depth': max_penetration,
        'has_nan': has_nan,
        'is_stable': not (explosion or penetration or has_nan),
    }


# ============ 对比分析 ============

def compare_solvers(all_data):
    """对比不同求解器"""
    results = []
    
    for exp_name, exp_info in all_data.items():
        config = exp_info['config']
        df = exp_info['data']
        
        bounce = calculate_bounce_metrics(df)
        energy = calculate_energy_metrics(df)
        deform = calculate_deformation_metrics(df)
        stability = calculate_stability_metrics(df)
        
        result = {
            'experiment': exp_name,
            'solver': config['solver'],
            'stiffness': config['stiffness'],
            'height': config['height'],
            'youngs_modulus': config['youngs_modulus'],
            'drop_height': config['drop_height'],
            **{f'bounce_{k}': v for k, v in bounce.items() if k != 'peaks'},
            **{f'energy_{k}': v for k, v in energy.items()},
            **{f'deform_{k}': v for k, v in deform.items()},
            **{f'stability_{k}': v for k, v in stability.items()},
        }
        results.append(result)
    
    return pd.DataFrame(results)


# ============ 可视化 ============

def plot_bounce_curves(all_data, stiffness='medium', height='medium'):
    """绘制弹跳曲线对比"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for exp_name, exp_info in all_data.items():
        config = exp_info['config']
        if config['stiffness'] == stiffness and config['height'] == height:
            df = exp_info['data']
            solver = config['solver']
            ax.plot(df['time'], df['center_y'], label=solver, linewidth=2)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Ball Center Y (m)', fontsize=12)
    ax.set_title(f'Bounce Curve Comparison (Stiffness: {stiffness}, Height: {height})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=BALL_RADIUS, color='r', linestyle='--', alpha=0.5, label='Ground')
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/bounce_curve_{stiffness}_{height}.png", dpi=150)
    plt.close()


def plot_energy_curves(all_data, stiffness='medium', height='medium'):
    """绘制能量衰减曲线"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for exp_name, exp_info in all_data.items():
        config = exp_info['config']
        if config['stiffness'] == stiffness and config['height'] == height:
            df = exp_info['data']
            solver = config['solver']
            ax.plot(df['time'], df['total_energy'], label=solver, linewidth=2)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Total Energy (J)', fontsize=12)
    ax.set_title(f'Energy Conservation (Stiffness: {stiffness}, Height: {height})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/energy_curve_{stiffness}_{height}.png", dpi=150)
    plt.close()


def plot_deformation_curves(all_data, stiffness='medium', height='medium'):
    """绘制形变曲线"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for exp_name, exp_info in all_data.items():
        config = exp_info['config']
        if config['stiffness'] == stiffness and config['height'] == height:
            df = exp_info['data']
            solver = config['solver']
            ax.plot(df['time'], df['max_deformation'], label=solver, linewidth=2)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Max Deformation (m)', fontsize=12)
    ax.set_title(f'Deformation Comparison (Stiffness: {stiffness}, Height: {height})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/deformation_curve_{stiffness}_{height}.png", dpi=150)
    plt.close()


def plot_solver_comparison(comparison_df):
    """绘制求解器综合对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 能量损失对比
    ax = axes[0, 0]
    solvers = comparison_df['solver'].unique()
    for solver in solvers:
        mask = comparison_df['solver'] == solver
        ax.boxplot(comparison_df[mask]['energy_loss_percent'], positions=[list(solvers).index(solver)])
    ax.set_xticks(range(len(solvers)))
    ax.set_xticklabels(solvers)
    ax.set_ylabel('Energy Loss (%)')
    ax.set_title('Energy Loss by Solver')
    ax.grid(True, alpha=0.3)
    
    # 2. 弹跳次数对比
    ax = axes[0, 1]
    for solver in solvers:
        mask = comparison_df['solver'] == solver
        ax.boxplot(comparison_df[mask]['bounce_num_bounces'], positions=[list(solvers).index(solver)])
    ax.set_xticks(range(len(solvers)))
    ax.set_xticklabels(solvers)
    ax.set_ylabel('Number of Bounces')
    ax.set_title('Bounce Count by Solver')
    ax.grid(True, alpha=0.3)
    
    # 3. 最大形变对比
    ax = axes[1, 0]
    for solver in solvers:
        mask = comparison_df['solver'] == solver
        ax.boxplot(comparison_df[mask]['deform_max_deformation'], positions=[list(solvers).index(solver)])
    ax.set_xticks(range(len(solvers)))
    ax.set_xticklabels(solvers)
    ax.set_ylabel('Max Deformation (m)')
    ax.set_title('Max Deformation by Solver')
    ax.grid(True, alpha=0.3)
    
    # 4. 稳定性对比
    ax = axes[1, 1]
    stability_counts = []
    for solver in solvers:
        mask = comparison_df['solver'] == solver
        stable_count = comparison_df[mask]['stability_is_stable'].sum()
        stability_counts.append(stable_count)
    ax.bar(solvers, stability_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Stable Experiments')
    ax.set_title('Stability by Solver')
    ax.set_ylim(0, len(comparison_df) / len(solvers) + 2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/solver_comparison.png", dpi=150)
    plt.close()


# ============ 报告生成 ============

def generate_report(comparison_df):
    """生成实验报告"""
    report = []
    report.append("=" * 60)
    report.append("弹性球碰撞对比实验 - 分析报告")
    report.append("=" * 60)
    report.append("")
    
    # 总体统计
    report.append("## 实验统计")
    report.append(f"总实验数: {len(comparison_df)}")
    report.append(f"求解器: {comparison_df['solver'].unique()}")
    report.append(f"刚度级别: {comparison_df['stiffness'].unique()}")
    report.append(f"高度级别: {comparison_df['height'].unique()}")
    report.append("")
    
    # 按求解器分组统计
    report.append("## 求解器性能对比")
    report.append("")
    
    for solver in comparison_df['solver'].unique():
        mask = comparison_df['solver'] == solver
        solver_data = comparison_df[mask]
        
        report.append(f"### {solver}")
        report.append(f"- 平均能量损失: {solver_data['energy_loss_percent'].mean():.2f}%")
        report.append(f"- 平均弹跳次数: {solver_data['bounce_num_bounces'].mean():.1f}")
        report.append(f"- 平均最大形变: {solver_data['deform_max_deformation'].mean():.4f}m")
        report.append(f"- 稳定性: {solver_data['stability_is_stable'].sum()}/{len(solver_data)} 实验稳定")
        report.append("")
    
    # 关键发现
    report.append("## 关键发现")
    report.append("")
    
    # 能量守恒最好的求解器
    best_energy = comparison_df.loc[comparison_df['energy_loss_percent'].idxmin()]
    report.append(f"1. 能量守恒最好: {best_energy['solver']} (损失 {best_energy['energy_loss_percent']:.2f}%)")
    
    # 最稳定的求解器
    stability_by_solver = comparison_df.groupby('solver')['stability_is_stable'].sum()
    most_stable = stability_by_solver.idxmax()
    report.append(f"2. 最稳定: {most_stable} ({stability_by_solver[most_stable]}/{len(comparison_df)/len(stability_by_solver)} 实验稳定)")
    
    # 形变最大的求解器
    max_deform = comparison_df.loc[comparison_df['deform_max_deformation'].idxmax()]
    report.append(f"3. 最大形变: {max_deform['solver']} ({max_deform['deform_max_deformation']:.4f}m)")
    
    report.append("")
    report.append("=" * 60)
    
    # 保存报告
    report_text = "\n".join(report)
    with open(f"{PROCESSED_DIR}/report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(report_text)
    return report_text


# ============ 主函数 ============

def main():
    """主函数"""
    print("=" * 60)
    print("弹性球碰撞对比实验 - 数据分析")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 加载所有实验数据
    print("\n加载实验数据...")
    all_data = load_all_experiments()
    print(f"加载了 {len(all_data)} 个实验")
    
    if not all_data:
        print("错误: 没有实验数据")
        return
    
    # 对比分析
    print("\n计算指标...")
    comparison_df = compare_solvers(all_data)
    
    # 保存对比结果
    comparison_df.to_csv(f"{PROCESSED_DIR}/comparison.csv", index=False)
    print(f"对比结果已保存: {PROCESSED_DIR}/comparison.csv")
    
    # 生成可视化
    print("\n生成可视化图表...")
    
    # 对每种刚度和高度组合绘制曲线
    for stiffness in ['soft', 'medium_soft', 'medium', 'hard']:
        for height in ['low', 'medium', 'high']:
            plot_bounce_curves(all_data, stiffness, height)
            plot_energy_curves(all_data, stiffness, height)
            plot_deformation_curves(all_data, stiffness, height)
    
    # 求解器综合对比
    plot_solver_comparison(comparison_df)
    
    print(f"图表已保存: {PLOTS_DIR}")
    
    # 生成报告
    print("\n生成报告...")
    generate_report(comparison_df)
    print(f"报告已保存: {PROCESSED_DIR}/report.txt")
    
    print("\n分析完成!")


if __name__ == "__main__":
    main()
