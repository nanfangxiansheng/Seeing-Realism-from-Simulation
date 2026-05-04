import json
import torch
import numpy as np
import random
import os
from tqdm import tqdm

# 假设你提供的类保存在 coreset_lib.py 中
# 如果在一个文件中，直接忽略这行 import
from Coreset import CoresetSelection 

# --- 模拟参数类 (修正版) ---
class Args:
    def __init__(self, coreset_key='score', stratas=50, budget_mode='confidence', sampling_mode='kcenter'):
        # 基础参数
        self.coreset_key = coreset_key
        self.stratas = stratas
        self.budget_mode = budget_mode
        self.sampling_mode = sampling_mode
        
        # --- 补全 Coreset.py 需要的额外参数，防止 AttributeError ---
        self.graph_score = False    # <--- 解决你当前的报错
        self.aucpr = False          # 防止后面报 aucpr 缺失
        self.gamma = None           # 图采样参数，设为 None 即可
        self.n_neighbor = 10        # 图采样邻居数，设为默认值
        self.graph_mode = None      # 占位
        self.precomputed_dists = None # 占位
        self.precomputed_neighbors = None # 占位
def load_and_process_json(json_path):
    """
    读取JSON并将层级数据转换为平铺的 Tensor/Numpy 格式
    """
    print(f"正在加载数据: {json_path} ...")
    
    # 假设 JSON 结构: { "Task_A": [ {video_info}, ... ], "Task_B": ... }
    # 如果是列表结构，请相应调整
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    all_features = []
    all_scores = []      # 对应 difficulty/loss
    all_targets = []     # 对应 task_id (0~31)
    
    # 元数据列表，用于通过索引找回视频ID或路径
    meta_info_list = [] 

    # 遍历 32 个任务
    task_names = sorted(list(raw_data.keys()))
    task_to_id = {name: i for i, name in enumerate(task_names)}

    for task_name, videos in raw_data.items():
        task_id = task_to_id[task_name]
        
        for vid in videos:
            # 1. 提取 Embedding (假设是 list 格式)
            emb = vid.get('embedding') 
            # 2. 提取难度分数 (Loss)
            score = vid.get('difficulty') 
            
            # 简单的数据校验
            if emb is None or score is None:
                continue

            all_features.append(emb)
            all_scores.append(score)
            all_targets.append(task_id)
            
            # 保存元数据以便后续检索
            meta_info_list.append({
                "task": task_name,
                "video_id": vid.get('id', 'unknown'),
                "path": vid.get('path', ''),
                "original_data": vid # 如果内存够大，可以存整个对象
            })

    # 转换为 Numpy/Tensor
    # Features: (N_samples, D_dim)
    features_np = np.array(all_features, dtype=np.float32)
    
    # Scores: (N_samples,)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
    
    # Targets: (N_samples,)
    targets_tensor = torch.tensor(all_targets, dtype=torch.long)

    # 构造 confidence (可选)
    # 假设 score 是 Loss，那么 Confidence ~= 1 / (1 + Loss) 或者简单归一化
    # 这里我们简单模拟一个 confidence，如果你的 budget_mode='uniform' 则不需要它
    max_loss = torch.max(scores_tensor)
    confidence_tensor = 1.0 - (scores_tensor / (max_loss + 1e-6))

    data_score = {
        'score': scores_tensor,      # 你的 difficulty
        'targets': targets_tensor,   # 你的 32 个任务 ID
        'confidence': confidence_tensor 
    }

    print(f"数据加载完毕。总样本数: {len(meta_info_list)}, 特征维度: {features_np.shape[1]}")
    return data_score, features_np, meta_info_list

# def save_selection(selected_indices, meta_info_list, output_path):
#     """
#     根据选中的索引，保存结果为新的 JSON
#     """
#     selected_data = []
#     for idx in selected_indices:
#         # 注意：idx 可能是 tensor，转为 int
#         i = int(idx)
#         selected_data.append(meta_info_list[i]['original_data'])
    
#     print(f"正在保存 {len(selected_data)} 个精选样本到 {output_path} ...")
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(selected_data, f, indent=4)


def save_selection(selected_indices, meta_info_list, output_path, data_score=None):
    """
    根据选中的索引，保存结果为新的 JSON，并附带采样分数
    """
    selected_data = []
    
    # 获取所有分数的 Tensor (如果传入了 data_score)
    all_scores = data_score['score'] if data_score else None
    
    for idx in selected_indices:
        # 注意：idx 可能是 tensor，转为 int
        i = int(idx)
        
        # 1. 复制原始数据 (使用 copy 防止修改原始内存中的对象)
        item = meta_info_list[i]['original_data'].copy()
        
        # 2. 注入分数 (如果可用)
        if all_scores is not None:
            # .item() 将 tensor 转为 Python float，否则 JSON 无法序列化
            score_value = all_scores[i].item()
            
            # 你可以把它存为 'sampling_score'，或者覆盖原来的 'difficulty'
            item['sampling_score'] = round(score_value, 6) # 保留6位小数
            
            # 如果你想确保 'difficulty' 字段一定存在且是最新的：
            # item['difficulty'] = round(score_value, 6)

        selected_data.append(item)
    
    print(f"正在保存 {len(selected_data)} 个精选样本到 {output_path} ...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(selected_data, f, indent=4)

# --- 主程序逻辑 ---

# --- 主程序逻辑 (修改版) ---

def main():
    # 1. 设置路径
    # 输入：指向你第一个脚本生成的那个文件
    json_file = "/home/intern/huangxiaodi2/RoboTwin/d2pruning/core/data/merged_dataset_input.json"
    
    # 输出：采样后的结果保存位置
    output_file = "/home/intern/huangxiaodi2/RoboTwin/d2pruning/core/data/coreset_subset_32tasks_20.json" 
    
    # 2. 加载数据
    # data_score 包含: 'score' (loss), 'targets' (task_id), 'confidence'
    # features 包含: embedding numpy array
    if not os.path.exists(json_file):
        print(f"错误：找不到输入文件 {json_file}")
        return

    data_score, features, meta_info = load_and_process_json(json_file)
    
    # 3. 设置采样目标数量
    # 假设总数据约 9600 条，你想选多少？例如 30% (约2880条) 或 固定 1000 条
    target_subset_size = 2000
    print(f"总样本数: {len(meta_info)} -> 目标采样数: {target_subset_size}")
    
    # 4. 配置采样策略
    # 推荐组合：Stratified + Confidence Budget + K-Center
    # 含义：按难度分层，难的层给更多名额，层内按几何特征选最具代表性的
    args = Args(
        coreset_key='score',     # 使用 difficulty (loss) 进行分层
        stratas=5,              # 将难度切分为 50 个区间
        budget_mode='confidence',# 【重要】根据 Loss 动态分配名额 (难样本多采，简单样本少采)
        sampling_mode='kcenter'  # 【重要】使用 Embedding 进行 K-Center 聚类采样 (去重/多样性)
    )
    
    # 如果你想用均匀分布（不管难易，每种难度都采一样多），把 budget_mode 改为 'uniform'
    
    print(f"开始执行 Stratified Sampling...")
    print(f"策略: Stratas={args.stratas}, Budget={args.budget_mode}, Sampling={args.sampling_mode}")

    selected_indices, _ = CoresetSelection.stratified_sampling(
        data_score, 
        target_subset_size, 
        args, 
        data_embeds=features # 传入 Embedding 用于 K-Center 计算
    )

    # 5. 保存结果
    # 传入 data_score 以便在输出中记录 sampling_score
    save_selection(selected_indices, meta_info, output_file, data_score=data_score)
    print(f"全部完成！结果已保存至: {output_file}")

if __name__ == "__main__":
    main()