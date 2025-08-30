# -*- coding: utf-8 -*-

import json
import random
from tqdm import tqdm

# =====================================================================================
# --------------------------------- 【配置区】 ---------------------------------
#         请根据你的实际情况和偏好，仔细修改这里的参数
# =====================================================================================

# 1. 文件路径
INPUT_FILE_PATH = "ft_dataset.jsonl"      # 你的原始数据集文件
OUTPUT_FILE_PATH = "ft_dataset_filtered.jsonl" # 筛选后输出的精华数据集文件

# 2. 目标数量
# 你希望从原始数据中筛选出多少条高质量的样本？
TARGET_SAMPLE_COUNT = 3000

# 3. 好友的独特用词 (！！！关键！！！)
# 列出你好友常用、具有标志性的词语、口头禅或表情符号。
# 列表越丰富，筛选的针对性越强。
KEYWORD_LIST = [
    "xs", "何意", "还真是", "ww", "呜呜", "女人", "感觉", "哎", "个么",
    "😂", "😅", "🤔"
]

# 4. 评分权重 (你可以调整不同策略的重要性)
# 这是启发式筛选的核心，决定了我们如何评价一条数据的好坏。
WEIGHTS = {
    "split_token": 8,   # 包含 <|split|> 的样本
    "keywords": 7,      # 每出现一个好友的常用关键词，就加分
    "length": 9,        # 回复长度在理想范围内，给予奖励
    "context": 2        # 对话历史越长，上下文越丰富，给予奖励
}

# 5. 长度评分的理想范围 (字符数)
# 我们不希望回复太短（如“嗯”）或太长（可能是大段复制粘贴）。
IDEAL_LENGTH_RANGE = (10, 80)


# =====================================================================================
# --------------------------------- 【脚本主逻辑】 ---------------------------------
#                      一般情况下，你不需要修改下面的代码
# =====================================================================================

def load_jsonl(file_path):
    """从JSONL文件加载数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 假设每行是一个JSON对象，且包含 "messages" 键
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"警告：跳过格式错误的行: {line.strip()}")
    return data

def save_jsonl(data, file_path):
    """将数据保存为JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def score_sample(sample):
    """
    为单个数据样本（一条多轮对话）计算分数。
    """
    total_score = 0
    
    # 确保 'messages' 键存在且不为空
    if not sample.get("messages"):
        return 0

    messages = sample["messages"]
    
    # 1. 上下文评分：对话轮次越多，上下文越丰富
    # messages 列表的长度代表了总的对话回合数
    num_turns = len(messages)
    total_score += num_turns * WEIGHTS["context"]

    # 我们主要分析最后一条助手的回复，这是模型要学习模仿的重点
    last_assistant_reply = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            last_assistant_reply = msg.get("content", "")
            break
            
    if not last_assistant_reply:
        return total_score # 如果没有助手回复，则只计算上下文分数

    # 2. <|split|> 行为评分
    if "<|split|>" in last_assistant_reply:
        total_score += WEIGHTS["split_token"]

    # 3. 关键词评分
    keyword_count = sum(1 for keyword in KEYWORD_LIST if keyword in last_assistant_reply)
    total_score += keyword_count * WEIGHTS["keywords"]

    # 4. 长度评分
    reply_length = len(last_assistant_reply)
    if IDEAL_LENGTH_RANGE[0] <= reply_length <= IDEAL_LENGTH_RANGE[1]:
        total_score += WEIGHTS["length"]
        
    return total_score

def main():
    """主执行函数"""
    print("="*50)
    print("开始执行启发式筛选脚本...")
    print(f"输入文件: {INPUT_FILE_PATH}")
    print(f"目标样本数: {TARGET_SAMPLE_COUNT}")
    print("="*50)

    # 加载数据
    print("\n[步骤 1/4] 正在加载原始数据集...")
    all_data = load_jsonl(INPUT_FILE_PATH)
    if not all_data:
        print("错误：未加载到任何数据，请检查输入文件路径和格式。")
        return
    print(f"加载完成，共 {len(all_data)} 条数据。")

    # 为每条数据评分
    print("\n[步骤 2/4] 正在为每条数据评分...")
    scored_data = []
    for sample in tqdm(all_data, desc="评分进度"):
        score = score_sample(sample)
        scored_data.append((score, sample))
    print("评分完成。")

    # 按分数排序
    print("\n[步骤 3/4] 正在根据分数排序...")
    scored_data.sort(key=lambda x: x[0], reverse=True)
    print("排序完成。")

    # 筛选出分数最高的 N 条
    print(f"\n[步骤 4/4] 正在筛选分数最高的 {TARGET_SAMPLE_COUNT} 条数据...")
    
    # 防止目标数量超过总数
    actual_count = min(TARGET_SAMPLE_COUNT, len(scored_data))
    
    # 提取样本，去除分数
    filtered_data = [sample for score, sample in scored_data[:actual_count]]
    
    # 为了保证多样性，最后再随机打乱一次
    random.shuffle(filtered_data)
    
    print("筛选完成。")

    # 保存到新文件
    save_jsonl(filtered_data, OUTPUT_FILE_PATH)
    
    print("\n" + "="*50)
    print("🎉 脚本执行成功！")
    print(f"已从 {len(all_data)} 条原始数据中筛选出 {len(filtered_data)} 条高质量数据。")
    print(f"精华数据集已保存到: {OUTPUT_FILE_PATH}")
    print("现在你可以使用这个新文件进行微调了。")
    print("="*50)

if __name__ == "__main__":
    main()