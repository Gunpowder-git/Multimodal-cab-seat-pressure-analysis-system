# 简单的压力传感器数据分析程序
# 专为新手设计，包含详细注释

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# 设置中文字体（如果系统没有中文字体，可能需要调整）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def safe_parse_matrix(matrix_str):
    """
    安全地解析m_data字段中的矩阵字符串
    
    参数:
        matrix_str: 字符串形式的矩阵，如 "[[0,0,0,...],[0,0,0,...],...]"
    
    返回:
        numpy数组表示的24x10矩阵
    """
    try:
        # 使用ast.literal_eval将字符串转换为Python列表
        matrix_list = ast.literal_eval(matrix_str)
        
        # 检查矩阵形状是否正确
        if isinstance(matrix_list, list) and len(matrix_list) == 24:
            # 确保每个子列表都有10个元素
            for i in range(len(matrix_list)):
                if len(matrix_list[i]) != 10:
                    # 如果元素数量不对，用0填充
                    matrix_list[i] = matrix_list[i] + [0] * (10 - len(matrix_list[i]))
            return np.array(matrix_list)
        else:
            # 如果格式不对，返回全零矩阵
            return np.zeros((24, 10))
    except:
        # 如果解析失败，返回全零矩阵
        return np.zeros((24, 10))

def calculate_matrix_mean(matrix):
    """
    计算24x10矩阵的平均值
    
    参数:
        matrix: 24x10的numpy数组
    
    返回:
        矩阵中所有元素的平均值
    """
    return np.mean(matrix)

def load_data(file_path):
    """
    加载和解析数据文件
    
    参数:
        file_path: 数据文件的路径
    
    返回:
        pandas DataFrame包含解析后的数据
    """
    print("正在加载数据文件...")
    
    # 读取CSV文件，使用制表符分隔
    data = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    
    print(f"成功读取 {len(data)} 行数据")
    
    # 解析m_data字段并计算平均压力
    print("正在解析压力矩阵数据...")
    data['pressure_matrix'] = data['m_data'].apply(safe_parse_matrix)
    data['avg_pressure'] = data['pressure_matrix'].apply(calculate_matrix_mean)
    
    print("数据解析完成！")
    return data

def create_weight_groups(data):
    """
    按体重将用户分为四组
    
    参数:
        data: 包含用户数据的DataFrame
    
    返回:
        添加了分组信息的DataFrame和分组边界
    """
    print("正在按体重分组...")
    
    # 获取所有不重复的用户信息（用户ID和体重）
    unique_users = data[['user_id', 'user_weight']].drop_duplicates()
    
    # 按体重排序
    unique_users = unique_users.sort_values('user_weight')
    
    # 计算四分位数（将数据分成四等分的边界值）
    q1 = unique_users['user_weight'].quantile(0.25)  # 25%分位数
    q2 = unique_users['user_weight'].quantile(0.50)  # 50%分位数（中位数）
    q3 = unique_users['user_weight'].quantile(0.75)  # 75%分位数
    
    print(f"体重分组边界: Q1={q1:.1f}kg, Q2={q2:.1f}kg, Q3={q3:.1f}kg")
    
    # 定义分组函数
    def get_weight_group(weight):
        if weight <= q1:
            return "轻量组"
        elif weight <= q2:
            return "中轻量组"
        elif weight <= q3:
            return "中重量组"
        else:
            return "重量组"
    
    # 为每个用户分配分组
    unique_users['weight_group'] = unique_users['user_weight'].apply(get_weight_group)
    
    # 将分组信息合并到原始数据
    data_with_groups = pd.merge(data, unique_users[['user_id', 'weight_group']], on='user_id')
    
    # 显示每组的人数
    group_counts = data_with_groups['weight_group'].value_counts()
    print("分组完成！每组人数:")
    for group, count in group_counts.items():
        print(f"  {group}: {count}条数据")
    
    return data_with_groups, (q1, q2, q3)

def calculate_derivative(pressure_values):
    """
    计算压力值的一阶导数（变化率）
    
    参数:
        pressure_values: 压力值序列
    
    返回:
        导数序列
    """
    # 使用中心差分法计算导数
    derivative = np.zeros(len(pressure_values))
    
    # 第一个点使用前向差分
    if len(pressure_values) > 1:
        derivative[0] = pressure_values[1] - pressure_values[0]
    
    # 中间点使用中心差分
    for i in range(1, len(pressure_values) - 1):
        derivative[i] = (pressure_values[i + 1] - pressure_values[i - 1]) / 2
    
    # 最后一个点使用后向差分
    if len(pressure_values) > 1:
        derivative[-1] = pressure_values[-1] - pressure_values[-2]
    
    return derivative

def find_slowdown_threshold(derivative, time_points, sensitivity=0.1):
    """
    找到导数开始缓慢下降的阈值点
    
    参数:
        derivative: 导数序列
        time_points: 对应的时间点
        sensitivity: 检测敏感度（0-1之间，越小越敏感）
    
    返回:
        阈值时间点和阈值值
    """
    # 平滑导数以减少噪声影响
    smooth_derivative = np.convolve(derivative, np.ones(5)/5, mode='same')
    
    # 计算导数的梯度（二阶导数）
    second_derivative = np.gradient(smooth_derivative)
    
    # 找到导数开始明显下降的点
    # 我们寻找二阶导数开始趋于平缓的点
    threshold = None
    threshold_value = None
    
    # 寻找拐点：二阶导数从负值开始接近零的点
    for i in range(10, len(second_derivative) - 5):
        # 检查当前点附近的二阶导数变化
        window = second_derivative[i:i+5]
        if np.all(np.abs(window) < sensitivity * np.max(np.abs(second_derivative))):
            # 找到了一个相对平缓的区域
            threshold = time_points[i]
            threshold_value = smooth_derivative[i]
            break
    
    return threshold, threshold_value

def analyze_temporal_data(data):
    """
    分析随时间变化的压力数据
    
    参数:
        data: 包含分组信息的DataFrame
    
    返回:
        按时间分组的分析结果
    """
    print("正在分析时序数据...")
    
    # 按unique_code和data_id排序，确保时间顺序
    data_sorted = data.sort_values(['unique_code', 'data_id'])
    
    # 为每个unique_code内的数据分配时间点（0-99帧）
    data_sorted['time_frame'] = data_sorted.groupby('unique_code').cumcount()
    
    # 检查每个unique_code的数据帧数
    frame_counts = data_sorted.groupby('unique_code').size()
    print(f"每个采样序列的平均帧数: {frame_counts.mean():.1f}")
    
    # 按时间和分组计算平均压力
    time_analysis = data_sorted.groupby(['time_frame', 'weight_group'])['avg_pressure'].mean().reset_index()
    
    # 按时间计算总体平均压力
    overall_time_analysis = data_sorted.groupby('time_frame')['avg_pressure'].agg(['mean', 'std']).reset_index()
    
    # 计算总体平均压力的导数
    overall_pressure = overall_time_analysis['mean'].values
    time_points = overall_time_analysis['time_frame'].values
    
    # 计算导数
    derivative = calculate_derivative(overall_pressure)
    overall_time_analysis['derivative'] = derivative
    
    # 找到阈值点
    threshold_frame, threshold_value = find_slowdown_threshold(derivative, time_points)
    
    print("时序分析完成！")
    return data_sorted, time_analysis, overall_time_analysis, threshold_frame, threshold_value

def create_plots(time_analysis, overall_time_analysis, weight_boundaries, threshold_frame=None, threshold_value=None):
    """
    创建分析图表，包含导数分析和阈值标记
    
    参数:
        time_analysis: 按时间分组的分析数据
        overall_time_analysis: 总体时间分析数据
        weight_boundaries: 体重分组边界
        threshold_frame: 阈值对应的时间帧
        threshold_value: 阈值对应的导数值
    """
    print("正在生成图表...")
    
    # 创建大图表（增加高度以容纳三个子图）
    plt.figure(figsize=(14, 12))
    
    # 第一个子图：不同体重组的压力变化
    plt.subplot(3, 1, 1)
    
    # 定义颜色和标签
    groups = ['轻量组', '中轻量组', '中重量组', '重量组']
    colors = ['blue', 'green', 'orange', 'red']
    
    # 为每个组绘制线图
    for i, group in enumerate(groups):
        group_data = time_analysis[time_analysis['weight_group'] == group]
        plt.plot(group_data['time_frame'], group_data['avg_pressure'], 
                label=group, color=colors[i], linewidth=2)
    
    plt.xlabel('时间帧 (0-100帧，每帧0.2秒)')
    plt.ylabel('平均压力')
    plt.title('不同体重组的平均压力随时间变化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 第二个子图：总体压力变化
    plt.subplot(3, 1, 2)
    
    plt.plot(overall_time_analysis['time_frame'], overall_time_analysis['mean'], 
             label='总体平均压力', color='purple', linewidth=2)
    
    # 添加标准差区域（显示数据的波动范围）
    plt.fill_between(overall_time_analysis['time_frame'],
                    overall_time_analysis['mean'] - overall_time_analysis['std'],
                    overall_time_analysis['mean'] + overall_time_analysis['std'],
                    alpha=0.3, color='purple', label='波动范围')
    
    plt.xlabel('时间帧 (0-100帧，每帧0.2秒)')
    plt.ylabel('平均压力')
    plt.title('总体平均压力随时间变化（含标准差）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 第三个子图：压力变化率（导数）分析
    plt.subplot(3, 1, 3)
    
    # 绘制导数曲线
    plt.plot(overall_time_analysis['time_frame'], overall_time_analysis['derivative'], 
             label='压力变化率（导数）', color='red', linewidth=2)
    
    # 如果找到阈值点，标记在图上
    if threshold_frame is not None:
        # 找到阈值点对应的导数
        threshold_index = np.where(overall_time_analysis['time_frame'] == threshold_frame)[0][0]
        threshold_derivative = overall_time_analysis['derivative'].iloc[threshold_index]
        
        # 标记阈值点
        plt.axvline(x=threshold_frame, color='green', linestyle='--', linewidth=2, 
                   label=f'阈值点 (帧{threshold_frame})')
        plt.scatter(threshold_frame, threshold_derivative, color='green', s=100, zorder=5)
        
        # 添加注释
        plt.annotate(f'阈值点\n帧: {threshold_frame}\n变化率: {threshold_derivative:.3f}', 
                    xy=(threshold_frame, threshold_derivative), 
                    xytext=(threshold_frame + 5, threshold_derivative + 0.1),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 添加快速下降和缓慢下降区域的标注
        plt.axvspan(0, threshold_frame, alpha=0.2, color='red', label='快速下降区域')
        plt.axvspan(threshold_frame, 100, alpha=0.2, color='blue', label='缓慢下降区域')
    
    plt.xlabel('时间帧 (0-100帧，每帧0.2秒)')
    plt.ylabel('压力变化率')
    plt.title('压力变化率（导数）随时间变化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('压力分析图表.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()
    
    # 打印分组信息和阈值检测结果
    q1, q2, q3 = weight_boundaries
    print("\n" + "="*60)
    print("分析结果详细信息:")
    print("="*60)
    print(f"轻量组 (蓝色): 体重 ≤ {q1:.1f} kg")
    print(f"中轻量组 (绿色): {q1:.1f} kg < 体重 ≤ {q2:.1f} kg")
    print(f"中重量组 (橙色): {q2:.1f} kg < 体重 ≤ {q3:.1f} kg")
    print(f"重量组 (红色): 体重 > {q3:.1f} kg")
    
    if threshold_frame is not None:
        threshold_time = threshold_frame * 0.2  # 转换为秒
        print(f"\n阈值检测结果:")
        print(f"开始缓慢下降的阈值帧: {threshold_frame}")
        print(f"对应时间: {threshold_time:.1f} 秒")
        print(f"阈值点的压力变化率: {threshold_value:.3f}")
        print(f"快速下降区域: 0-{threshold_frame}帧 (0-{threshold_time:.1f}秒)")
        print(f"缓慢下降区域: {threshold_frame}-100帧 ({threshold_time:.1f}-20.0秒)")
    else:
        print("\n未检测到明显的阈值点")
    print("="*60)

def analyze_stability_per_person(data_sorted, threshold_frame=30):
    """
    分析每个人的稳定期数据
    
    参数:
        data_sorted: 排序后的数据
        threshold_frame: 阈值帧数（默认为30帧）
    
    返回:
        包含每个人稳定期数据的DataFrame
    """
    print(f"正在分析每个人的稳定期数据（去除前{threshold_frame}帧）...")
    
    # 存储每个人的稳定期数据
    stability_data = []
    
    # 按用户分组
    user_groups = data_sorted.groupby('user_id')
    
    for user_id, user_data in user_groups:
        # 获取用户的基本信息（只取第一行，因为同一个用户有重复信息）
        user_info = user_data.iloc[0]
        
        # 对每个unique_code（采样序列）进行处理
        unique_codes = user_data['unique_code'].unique()
        
        for unique_code in unique_codes:
            # 获取该采样序列的数据
            sequence_data = user_data[user_data['unique_code'] == unique_code]
            
            # 确保数据按时间顺序排列
            sequence_data = sequence_data.sort_values('time_frame')
            
            # 计算稳定期的统计数据（去除前threshold_frame帧）
            stable_data = sequence_data[sequence_data['time_frame'] >= threshold_frame]
            
            if len(stable_data) > 0:
                # 计算稳定期的平均压力
                stable_avg_pressure = stable_data['avg_pressure'].mean()
                
                # 计算稳定期的标准差（稳定程度）
                stable_std = stable_data['avg_pressure'].std()
                
                # 计算稳定期的变异系数（标准差/均值，表示相对稳定性）
                stable_cv = stable_std / stable_avg_pressure if stable_avg_pressure > 0 else 0
                
                # 计算平均稳定度（稳定期的压力值的平均绝对偏差的倒数）
                # 平均稳定度 = 1 / (1 + 变异系数)，值越接近1表示越稳定
                stable_degree = 1 / (1 + stable_cv) if stable_cv >= 0 else 0
                
                # 收集数据
                stability_data.append({
                    'user_id': user_id,
                    'unique_code': unique_code,
                    'user_weight': user_info['user_weight'],
                    'user_height': user_info['user_height'],
                    'weight_group': user_info['weight_group'],
                    'stable_avg_pressure': stable_avg_pressure,
                    'stable_std': stable_std,
                    'stable_cv': stable_cv,
                    'stable_degree': stable_degree,  # 新增：平均稳定度
                    'stable_frames_count': len(stable_data),
                    'total_frames_count': len(sequence_data)
                })
    
    # 转换为DataFrame
    stability_df = pd.DataFrame(stability_data)
    
    print(f"稳定期分析完成！共分析 {len(stability_df)} 个采样序列")
    print(f"每个序列平均有 {stability_df['stable_frames_count'].mean():.1f} 帧稳定期数据")
    
    return stability_df

def create_stability_scatter_plot(stability_df):
    """
    创建稳定期数据的散点图
    
    参数:
        stability_df: 稳定期分析数据
    """
    print("正在生成稳定期分析散点图...")
    
    # 创建散点图
    plt.figure(figsize=(12, 8))
    
    # 定义颜色映射
    groups = ['轻量组', '中轻量组', '中重量组', '重量组']
    colors = ['blue', 'green', 'orange', 'red']
    
    # 为每个体重组创建散点图
    for i, group in enumerate(groups):
        group_data = stability_df[stability_df['weight_group'] == group]
        
        if len(group_data) > 0:
            plt.scatter(group_data['user_weight'], group_data['stable_avg_pressure'], 
                       label=group, color=colors[i], alpha=0.7, s=50)
    
    plt.xlabel('用户体重 (kg)')
    plt.ylabel('稳定期平均压力')
    plt.title('个人稳定期平均压力散点图（按体重分组）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.savefig('稳定期压力散点图.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建变异系数散点图（稳定性指标）
    plt.figure(figsize=(12, 8))
    
    for i, group in enumerate(groups):
        group_data = stability_df[stability_df['weight_group'] == group]
        
        if len(group_data) > 0:
            plt.scatter(group_data['user_weight'], group_data['stable_cv'], 
                       label=group, color=colors[i], alpha=0.7, s=50)
    
    plt.xlabel('用户体重 (kg)')
    plt.ylabel('稳定期变异系数 (标准差/均值)')
    plt.title('个人稳定期稳定性散点图（变异系数越小越稳定）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.savefig('稳定期稳定性散点图.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建平均稳定度散点图（新功能）
    plt.figure(figsize=(12, 8))
    
    for i, group in enumerate(groups):
        group_data = stability_df[stability_df['weight_group'] == group]
        
        if len(group_data) > 0:
            plt.scatter(group_data['user_weight'], group_data['stable_degree'], 
                       label=group, color=colors[i], alpha=0.7, s=50)
    
    plt.xlabel('用户体重 (kg)')
    plt.ylabel('平均稳定度 (1/(1+变异系数))')
    plt.title('个人平均稳定度散点图（值越接近1表示越稳定）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加水平参考线，表示稳定性等级
    plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='极稳定 (稳定度>0.9)')
    plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='中等稳定 (0.8-0.9)')
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='较不稳定 (稳定度<0.7)')
    
    # 保存图表
    plt.savefig('平均稳定度散点图.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印稳定性统计信息
    print("" + "="*60)
    print("稳定期分析统计信息:")
    print("="*60)
    
    for group in groups:
        group_data = stability_df[stability_df['weight_group'] == group]
        if len(group_data) > 0:
            print(f"{group}:")
            print(f"  平均稳定期压力: {group_data['stable_avg_pressure'].mean():.3f}")
            print(f"  平均变异系数: {group_data['stable_cv'].mean():.3f} (值越小越稳定)")
            print(f"  平均稳定度: {group_data['stable_degree'].mean():.3f} (值越接近1表示越稳定)")
            print(f"  样本数量: {len(group_data)}")
            print()

def save_results(data_sorted, time_analysis, stability_df=None):
    """
    保存分析结果到CSV文件
    
    参数:
        data_sorted: 排序后的数据
        time_analysis: 时间分析结果
        stability_df: 稳定期分析数据
    """
    print("正在保存结果...")
    
    # 保存完整分析数据
    data_sorted.to_csv('完整分析数据.csv', index=False, encoding='utf-8-sig')
    
    # 保存时间分析结果
    time_analysis.to_csv('时间分析结果.csv', index=False, encoding='utf-8-sig')
    
    # 如果存在稳定期分析数据，也保存
    if stability_df is not None:
        stability_df.to_csv('稳定期分析数据.csv', index=False, encoding='utf-8-sig')
    
    print("结果已保存到以下文件:")
    print("  - 完整分析数据.csv")
    print("  - 时间分析结果.csv")
    if stability_df is not None:
        print("  - 稳定期分析数据.csv")
    print("  - 压力分析图表.png")
    print("  - 稳定期压力散点图.png")
    print("  - 稳定期稳定性散点图.png")

def main():
    """
    主函数 - 程序的入口点
    """
    print("="*60)
    print("椅子压力传感器数据分析程序（含导数分析）")
    print("="*60)
    
    try:
        # 数据文件路径
        file_path = "c:/Users/gunpo/OneDrive/Desktop/data.txt"
        
        # 第一步：加载数据
        data = load_data(file_path)
        
        # 第二步：按体重分组
        data_with_groups, weight_boundaries = create_weight_groups(data)
        
        # 第三步：分析时序数据（包含导数计算）
        data_sorted, time_analysis, overall_time_analysis, threshold_frame, threshold_value = analyze_temporal_data(data_with_groups)
        
        # 第四步：创建图表（包含导数分析和阈值标记）
        create_plots(time_analysis, overall_time_analysis, weight_boundaries, threshold_frame, threshold_value)
        
        # 第五步：分析每个人的稳定期数据
        # 使用自动检测的阈值帧，如果没有检测到则使用默认值30
        stability_threshold = threshold_frame if threshold_frame is not None else 30
        stability_df = analyze_stability_per_person(data_sorted, stability_threshold)
        
        # 第六步：创建稳定期散点图
        create_stability_scatter_plot(stability_df)
        
        # 第七步：保存结果
        save_results(data_sorted, time_analysis, stability_df)
        
        print("\n" + "="*60)
        print("分析完成！")
        print("="*60)
        
    except FileNotFoundError:
        print("错误：找不到数据文件！")
        print("请检查文件路径是否正确:", file_path)
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        print("请检查数据文件格式是否正确")

# 程序入口
if __name__ == "__main__":
    main()
