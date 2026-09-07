"""
批量运行脚本 - 弹性球碰撞对比实验
在Houdini hython中运行

用法：
hython batch_run.py
"""

import sys
import os

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入场景搭建脚本
import setup_scene


def main():
    """主函数"""
    print("=" * 60)
    print("弹性球碰撞对比实验 - 批量运行")
    print("=" * 60)
    
    # 1. 搭建所有实验场景
    print("\n[1/4] 搭建实验场景...")
    experiments = setup_scene.run_all_experiments()
    
    # 2. 运行所有仿真
    print("\n[2/4] 运行仿真...")
    results = setup_scene.batch_run_simulations()
    
    # 3. 提取数据
    print("\n[3/4] 提取数据...")
    setup_scene.batch_extract_data()
    
    # 4. 保存运行统计
    print("\n[4/4] 保存运行统计...")
    import json
    stats_path = f"{setup_scene.OUTPUT_DIR}/data/run_statistics.json"
    
    total_time = sum(r['simulation_time'] for r in results)
    avg_time = total_time / len(results) if results else 0
    
    stats = {
        'total_experiments': len(results),
        'total_simulation_time': total_time,
        'avg_simulation_time': avg_time,
        'results': results,
    }
    
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n运行统计已保存: {stats_path}")
    print(f"总仿真时间: {total_time:.2f}秒")
    print(f"平均仿真时间: {avg_time:.2f}秒")
    
    print("\n" + "=" * 60)
    print("批量运行完成!")
    print("=" * 60)
    print("\n下一步:")
    print("1. 将 results/ 目录复制到分析环境")
    print("2. 运行 python analyze_results.py 分析数据")


if __name__ == "__main__":
    main()
