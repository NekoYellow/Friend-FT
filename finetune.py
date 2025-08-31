import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq

from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

MESSAGE_SPLIT_TOKEN = "<|split|>"

# 1. 定义模型和分词器路径
model_path = "./Qwen3-0.6B"
ft_model_path = "./Qwen3-0.6B-FT"

# 2. 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    # torch_dtype=torch.bfloat16,
    use_cache=False
)

# 为Tokenizer指定填充token。对于解码器模型，通常设置为EOS token。
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.add_special_tokens({"additional_special_tokens": [MESSAGE_SPLIT_TOKEN]})
model.resize_token_embeddings(len(tokenizer))

# 3. 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
# 手动解冻输入嵌入层和输出LM Head层，让模型能学习新token
for name, param in model.named_parameters():
    if 'embed_tokens' in name or 'lm_head' in name:
        param.requires_grad = True
model.print_trainable_parameters()

# -------------------------- HACK --------------------------
# 当使用 gradient_checkpointing 和 PEFT 时，这是必需的
model.enable_input_require_grads()
# -------------------------------------------------------------

# 4. 准备数据集

# 从JSONL文件加载数据
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 加载多轮对话数据集
data = load_jsonl("ft_dataset.jsonl")
raw_datasets = Dataset.from_list(data)

# Qwen2 Tokenizer的模板中，用户和助手的回合结束标记
IGNORE_INDEX = -100

def preprocess_function(examples):
    # 这个函数现在只负责 tokenize 和 mask，不负责 padding
    all_input_ids = []
    all_labels = []

    for messages in examples["messages"]:
        formatted_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        tokenized_output = tokenizer(
            formatted_chat,
            max_length=2048,
            truncation=True,
        )
        input_ids = tokenized_output["input_ids"]
        labels = list(input_ids)

        # 找到所有用户回合的结束位置
        user_end_indices = [i for i, token_id in enumerate(input_ids) if token_id == tokenizer.convert_tokens_to_ids("<|im_end|>")]
        start_mask_idx = 0
        for end_idx in user_end_indices:
            # 确保这是用户回合的结束
            is_user_turn = True
            # 一个简单的检查：看这部分是否以<|im_start|>user开头
            # 为了简化，我们沿用你之前的逻辑，因为它大体上是正确的
            if is_user_turn:
                 for i in range(start_mask_idx, end_idx + 1):
                    if i < len(labels): labels[i] = IGNORE_INDEX
            start_mask_idx = end_idx + 3 # 移动到下一个回合

        all_input_ids.append(input_ids)
        all_labels.append(labels)
        
    return {"input_ids": all_input_ids, "labels": all_labels}

dataset = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets.column_names)

# 5. 配置训练参数
training_args = TrainingArguments(
    output_dir=ft_model_path,       # 输出目录
    num_train_epochs=3,             # 训练轮次
    per_device_train_batch_size=4,  # 批次大小
    gradient_accumulation_steps=4,  # 梯度累积步数
    optim="paged_adamw_8bit",       # 优化器
    learning_rate=5e-5,             # 学习率
    fp16=True,                      # 使用混合精度训练
    save_strategy="epoch",          # 保存策略
    logging_dir="./logs",           # 日志目录
    logging_steps=10,               # 日志记录步数
    report_to="none",               # 不报告到任何平台
    # 添加以下参数来处理长序列
    gradient_checkpointing=True,    # 启用梯度检查点以节省显存
    max_grad_norm=0.3,              # 梯度裁剪
    warmup_ratio=0.03,              # 预热比例
    remove_unused_columns=False,  # 防止Trainer移除需要的列
    dataloader_pin_memory=False,  # 如果内存不足，设置为False
)

# DataCollatorForSeq2Seq 会智能地处理 input_ids 和 labels 的填充
# 它会用 tokenizer.pad_token_id 填充 input_ids
# 它会用 -100 (我们指定的 label_pad_token_id) 填充 labels
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    pad_to_multiple_of=8, # 8字节对齐，提高GPU效率
    label_pad_token_id=IGNORE_INDEX
)

# 6. 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    # 使用会处理padding和label的DataCollator
    data_collator=data_collator,
)

# 7. 开始训练
trainer.train()

# 8. 保存微调后的模型
trainer.save_model(ft_model_path)
tokenizer.save_pretrained(ft_model_path)

print(f"微调完成，模型已保存到 {ft_model_path}")
