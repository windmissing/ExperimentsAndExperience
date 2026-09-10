"""
实验2：3DGS绑定方式对比 - 数据分析脚本
"""

import os
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "D:/Experiment2_Binding/results"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"
PROCESSED_DIR = f"{OUTPUT_DIR}/processed"


def load_all_experiments():
    """加载所有实验数据"""
    config_path = f"{OUTPUT_DIR}/experiments.json"
    with open(config_path, "r") as f:
        experiments = json.load(f)
    
    all_data = {}
    for exp in experiments:
        csv_path = f"{OUTPUT_DIR}/data/{exp['name']}.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_data[exp['name']] = {'config': exp, 'data': df}
    
    return all_data


def plot_displacement_curves(all_data, force_level='medium'):
    """绘制最大位移曲线"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = ['LBS', 'PartMM', 'SparseCP', 'Direct']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for method, color in zip(methods, colors):
        exp_name = f"{method}_{force_level}"
        if exp_name in all_data:
            df = all_data[exp_name]['data']
            ax.plot(df['time'], df['max_displacement'], label=method, linewidth=2, color=color)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Max Displacement (m)', fontsize=12)
    ax.set_title(f'Max Displacement Comparison ({force_level} force)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/displacement_curve_{force_level}.png", dpi=150)
    plt.close()


def plot_volume_ratio_curves(all_data, force_level='medium'):
    """绘制体积保持率曲线"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = ['LBS', 'PartMM', 'SparseCP', 'Direct']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for method, color in zip(methods, colors):
        exp_name = f"{method}_{force_level}"
        if exp_name in all_data:
            df = all_data[exp_name]['data']
            ax.plot(df['time'], df['volume_ratio'], label=method, linewidth=2, color=color)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ideal (1.0)')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Volume Ratio', fontsize=12)
    ax.set_title(f'Volume Preservation ({force_level} force)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/volume_ratio_{force_level}.png", dpi=150)
    plt.close()


def plot_laplacian_energy_curves(all_data, force_level='medium'):
    """绘制拉普拉斯能量曲线（形变平滑度）"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = ['LBS', 'PartMM', 'SparseCP', 'Direct']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for method, color in zip(methods, colors):
        exp_name = f"{method}_{force_level}"
        if exp_name in all_data:
            df = all_data[exp_name]['data']
            ax.plot(df['time'], df['laplacian_energy'], label=method, linewidth=2, color=color)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Laplacian Energy', fontsize=12)
    ax.set_title(f'Deformation Smoothness ({force_level} force)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/laplacian_energy_{force_level}.png", dpi=150)
    plt.close()


def plot_overlap_curves(all_data, force_level='medium'):
    """绘制重叠数量曲线（伪影程度）"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = ['LBS', 'PartMM', 'SparseCP', 'Direct']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for method, color in zip(methods, colors):
        exp_name = f"{method}_{force_level}"
        if exp_name in all_data:
            df = all_data[exp_name]['data']
            ax.plot(df['time'], df['overlap_count'], label=method, linewidth=2, color=color)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Overlap Count', fontsize=12)
    ax.set_title(f'Artifact Level ({force_level} force)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/overlap_count_{force_level}.png", dpi=150)
    plt.close()


def plot_method_comparison(all_data):
    """绘制方法综合对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = ['LBS', 'PartMM', 'SparseCP', 'Direct']
    force_levels = ['light', 'medium', 'heavy']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. 体积保持率对比
    ax = axes[0, 0]
    x = np.arange(len(methods))
    width = 0.25
    for i, force in enumerate(force_levels):
        means = []
        for method in methods:
            exp_name = f"{method}_{force}"
            if exp_name in all_data:
                df = all_data[exp_name]['data']
                means.append(df['volume_ratio'].mean())
            else:
                means.append(0)
        axes[0, 0].bar(x + i * width, means, width, label=force)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Avg Volume Ratio')
    ax.set_title('Volume Preservation')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. 形变平滑度对比
    ax = axes[0, 1]
    for i, force in enumerate(force_levels):
        means = []
        for method in methods:
            exp_name = f"{method}_{force}"
            if exp_name in all_data:
                df = all_data[exp_name]['data']
                means.append(df['laplacian_energy'].mean())
            else:
                means.append(0)
        axes[0, 1].bar(x + i * width, means, width, label=force)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Avg Laplacian Energy')
    ax.set_title('Deformation Smoothness (lower = better)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. 伪影程度对比
    ax = axes[1, 0]
    for i, force in enumerate(force_levels):
        means = []
        for method in methods:
            exp_name = f"{method}_{force}"
            if exp_name in all_data:
                df = all_data[exp_name]['data']
                means.append(df['overlap_count'].mean())
            else:
                means.append(0)
        axes[1, 0].bar(x + i * width, means, width, label=force)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Avg Overlap Count')
    ax.set_title('Artifact Level (lower = better)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 最大位移对比
    ax = axes[1, 1]
    for i, force in enumerate(force_levels):
        maxes = []
        for method in methods:
            exp_name = f"{method}_{force}"
            if exp_name in all_data:
                df = all_data[exp_name]['data']
                maxes.append(df['max_displacement'].max())
            else:
                maxes.append(0)
        axes[1, 1].bar(x + i * width, maxes, width, label=force)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Max Displacement (m)')
    ax.set_title('Responsiveness (higher = more responsive)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/method_comparison.png", dpi=150)
    plt.close()


def generate_report(all_data):
    """生成实验报告"""
    report = []
    report.append("=" * 60)
    report.append("3DGS绑定方式对比实验 - 分析报告")
    report.append("=" * 60)
    report.append("")
    
    report.append("## 实验统计")
    report.append(f"总实验数: {len(all_data)}")
    report.append(f"绑定方式: LBS / PartMM / SparseCP / Direct")
    report.append(f"力度水平: light / medium / heavy")
    report.append("")
    
    report.append("## 各方法性能对比")
    report.append("")
    
    methods = ['LBS', 'PartMM', 'SparseCP', 'Direct']
    
    for method in methods:
        report.append(f"### {method}")
        
        # 收集中等力度下的数据
        exp_name = f"{method}_medium"
        if exp_name in all_data:
            df = all_data[exp_name]['data']
            report.append(f"- 平均体积保持率: {df['volume_ratio'].mean():.4f}")
            report.append(f"- 平均拉普拉斯能量: {df['laplacian_energy'].mean():.6f}")
            report.append(f"- 平均重叠数: {df['overlap_count'].mean():.1f}")
            report.append(f"- 最大位移: {df['max_displacement'].max():.4f}m")
        report.append("")
    
    report.append("## 关键发现")
    report.append("")
    
    # 体积保持最好的方法
    best_volume = None
    best_volume_val = 0
    for method in methods:
        exp_name = f"{method}_medium"
        if exp_name in all_data:
            df = all_data[exp_name]['data']
            val = df['volume_ratio'].mean()
            if abs(val - 1.0) < abs(best_volume_val - 1.0):
                best_volume = method
                best_volume_val = val
    
    report.append(f"1. 体积保持最好: {best_volume} (平均体积比 {best_volume_val:.4f})")
    
    # 形变最平滑的方法
    best_smooth = None
    best_smooth_val = float('inf')
    for method in methods:
        exp_name = f"{method}_medium"
        if exp_name in all_data:
            df = all_data[exp_name]['data']
            val = df['laplacian_energy'].mean()
            if val < best_smooth_val:
                best_smooth = method
                best_smooth_val = val
    
    report.append(f"2. 形变最平滑: {best_smooth} (平均拉普拉斯能量 {best_smooth_val:.6f})")
    
    # 伪影最少的方法
    best_artifact = None
    best_artifact_val = float('inf')
    for method in methods:
        exp_name = f"{method}_medium"
        if exp_name in all_data:
            df = all_data[exp_name]['data']
            val = df['overlap_count'].mean()
            if val < best_artifact_val:
                best_artifact = method
                best_artifact_val = val
    
    report.append(f"3. 伪影最少: {best_artifact} (平均重叠数 {best_artifact_val:.1f})")
    
    report.append("")
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    with open(f"{PROCESSED_DIR}/report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(report_text)


def main():
    print("=" * 60)
    print("3DGS绑定方式对比实验 - 数据分析")
    print("=" * 60)
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("\n加载实验数据...")
    all_data = load_all_experiments()
    print(f"加载了 {len(all_data)} 个实验")
    
    if not all_data:
        print("错误: 没有实验数据")
        return
    
    print("\n生成可视化图表...")
    
    for force in ['light', 'medium', 'heavy']:
        plot_displacement_curves(all_data, force)
        plot_volume_ratio_curves(all_data, force)
        plot_laplacian_energy_curves(all_data, force)
        plot_overlap_curves(all_data, force)
    
    plot_method_comparison(all_data)
    
    print(f"图表已保存: {PLOTS_DIR}")
    
    print("\n生成报告...")
    generate_report(all_data)
    print(f"报告已保存: {PROCESSED_DIR}/report.txt")
    
    print("\n分析完成!")


if __name__ == "__main__":
    main()
