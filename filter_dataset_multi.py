# 分桶-评分-采样
# 1. 分桶 (Bucketing)：首先，脚本会遍历所有多轮对话，根据它们的对话轮数（messages列表的长度）
# 将它们分为“短对话”、“中等对话”、“长对话”三个桶。
# 2. 评分 (Scoring)：在每个桶内部，脚本会为每一段对话打一个“质量分”。分数基于我们定义的启发式
# 规则（如关键词密度、<|split|>使用频率、角色转换频率等）。
# 3. 采样 (Sampling)：最后，脚本会根据你设定的目标比例（比如，我想要20%的短对话，50%的中等对话，
# 30%的长对话），从每个桶中挑选出分数最高的样本，组合成最终的精华数据集。

import json
import random
from tqdm import tqdm
import math

# =====================================================================================
# --------------------------------- 【 expert configuration 】 ---------------------------------
# =====================================================================================

# 1. File Paths
INPUT_FILE = "ft_dataset_multi.jsonl"
OUTPUT_FILE = "ft_dataset.jsonl"

MESSAGE_SPLIT_TOKEN = "<|split|>"

SYSTEM_PROMPT = "你将扮演用户的好友'There'与用户在线聊天。"


# 2. 目标数量与分布
TARGET_SAMPLE_COUNT = 6000

LENGTH_BUCKETS = {
    "short": (2, 3),
    "medium": (4, 6),
    "long": (7, 100)
}

TARGET_DISTRIBUTION = {
    "short": 0.50,
    "medium": 0.30,
    "long": 0.20,
}
assert math.isclose(sum(TARGET_DISTRIBUTION.values()), 1.0), "TARGET_DISTRIBUTION 的总和必须为 1.0"


# 3. 带权重的关键词字典
# 为你好友的独特用词分配权重分数。
# 建议：越是独特、罕见的口头禅，权重越高。越是常用的表情或词语，权重越低。
KEYWORD_WEIGHTS = {
    # -- 高权重 (独特口头禅/行为) --
    "个么": 8, "女人": 8, "何意": 8, 
    "草拟的": 8, "无所谓": 8,
    
    # -- 中等权重 (常用词) --
    "还真是": 6, "ww": 6, "呜呜": 6,
    "好看": 4, "感觉": 4, "哎": 4, "妈的": 6,

    # -- 低权重 (常用表情/语气词) --
    "[Awkward]": 2, "[Angry]": 2, "[Sob]": 2, "🤔": 2,
    "啊": 1
}

# 4. 评分权重
SCORING_WEIGHTS = {
    "keywords_master_weight": 1.5, # 作为所有关键词得分的“总乘数”
    "split_token_freq": 20,
    "turn_alternation": 10,
    "variance_in_length": 5
}


# =====================================================================================
# --------------------------------- 【 script logic 】 ---------------------------------
# =====================================================================================

def load_jsonl(file_path):
    """从JSONL文件加载数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, file_path):
    """将数据保存为JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def score_conversation(conversation):
    """为一整段多轮对话计算质量分 (使用加权关键词)"""
    messages = conversation["messages"]
    num_turns = len(messages)
    if num_turns == 0:
        return 0

    assistant_replies = [msg['content'] for msg in messages if msg['role'] == 'assistant']
    if not assistant_replies:
        return 0

    # 1. 计算加权关键词得分
    total_keyword_score = 0
    for reply in assistant_replies:
        # 累加在回复中找到的所有关键词的权重
        total_keyword_score += sum(weight for keyword, weight in KEYWORD_WEIGHTS.items() if keyword in reply)
    # 按总轮数进行归一化，得到“每轮平均关键词得分”
    keyword_score_per_turn = total_keyword_score / num_turns
    
    # 2. <|split|> 使用频率
    split_count = sum(reply.count(MESSAGE_SPLIT_TOKEN) for reply in assistant_replies)
    split_freq = split_count / len(assistant_replies)

    # 3. 角色转换频率
    alternations = sum(1 for i in range(1, num_turns) if messages[i]['role'] != messages[i-1]['role'])
    turn_alternation_ratio = alternations / (num_turns - 1) if num_turns > 1 else 0

    # 4. 回复长度多样性
    reply_lengths = [len(reply) for reply in assistant_replies]
    mean_len = sum(reply_lengths) / len(reply_lengths)
    variance = sum((l - mean_len) ** 2 for l in reply_lengths) / len(reply_lengths)
    variance_score = math.log1p(variance)

    # 计算总分
    total_score = (
        (keyword_score_per_turn * SCORING_WEIGHTS["keywords_master_weight"]) +
        (split_freq * SCORING_WEIGHTS["split_token_freq"]) +
        (turn_alternation_ratio * SCORING_WEIGHTS["turn_alternation"]) +
        (variance_score * SCORING_WEIGHTS["variance_in_length"])
    )
    
    return total_score

def main():
    """主执行函数"""
    print("="*60)
    print("启动高级启发式筛选脚本 (带独立关键词权重)")
    print("="*60)

    # 1. 加载所有已生成的多轮对话数据
    print(f"\n[Step 1/4] 从 {INPUT_FILE} 加载数据...")
    all_conversations = load_jsonl(INPUT_FILE)
    print(f"加载了 {len(all_conversations)} 段完整对话。")

    # 2. 评分并放入对应的长度桶
    print("\n[Step 2/4] 正在评分并进行分桶...")
    buckets = {name: [] for name in LENGTH_BUCKETS}
    other_bucket = []

    for conv in tqdm(all_conversations, desc="评分和分桶"):
        score = score_conversation(conv)
        conv["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}] + conv["messages"]
        num_turns = len(conv["messages"])
        
        placed = False
        for name, (min_len, max_len) in LENGTH_BUCKETS.items():
            if min_len <= num_turns <= max_len:
                buckets[name].append((score, conv))
                placed = True
                break
        if not placed:
            other_bucket.append((score, conv))
            
    print("分桶完成。各桶内样本数量:")
    for name, bucket_items in buckets.items():
        print(f"  - {name.capitalize()} 对话: {len(bucket_items)} 条")
    print(f"  - 其他 (不符合长度定义): {len(other_bucket)} 条")

    # 3. 从每个桶中按比例筛选出分数最高的样本
    print("\n[Step 3/4] 正在根据目标分布进行分层采样...")
    final_dataset = []
    for name, percentage in TARGET_DISTRIBUTION.items():
        num_to_take = int(TARGET_SAMPLE_COUNT * percentage)
        buckets[name].sort(key=lambda x: x[0], reverse=True)
        selected_samples = [conv for score, conv in buckets[name][:num_to_take]]
        final_dataset.extend(selected_samples)
        print(f"  - 从 '{name}' 桶中采样了 {len(selected_samples)}/{num_to_take} 条 (目标/实际)。")

    # 4. 保存最终的数据集
    print("\n[Step 4/4] 正在打乱数据并保存...")
    random.shuffle(final_dataset)
    save_jsonl(final_dataset, OUTPUT_FILE)

    print("\n" + "="*60)
    print("🎉 最终精华数据集构建成功！")
    print(f"总计筛选出 {len(final_dataset)} 条符合分布的高质量对话。")
    print(f"文件已保存至: {OUTPUT_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()