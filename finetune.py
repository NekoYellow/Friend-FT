import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

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
    torch_dtype=torch.bfloat16,
    use_cache=False
)
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
# 注意：这可能需要根据你的tokenizer版本微调，但通常是这样
IGNORE_INDEX = -100
USER_END_TOKEN_ID = tokenizer.convert_tokens_to_ids("<|im_end|>")

def preprocess_and_mask_labels(examples):
    all_input_ids = []
    all_labels = []

    for messages in examples["messages"]:
        # 使用模板格式化对话
        formatted_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False # 微调时不要加 assistant 起始符
        )
        
        # Tokenize
        tokenized_output = tokenizer(
            formatted_chat,
            max_length=2048,
            truncation=True,
        )

        input_ids = tokenized_output["input_ids"]
        labels = list(input_ids)

        # 找到所有用户回合的结束位置
        user_end_indices = [i for i, token_id in enumerate(input_ids) if token_id == USER_END_TOKEN_ID]
        
        # 掩码所有非助手角色的部分
        # 每次掩码从对话开始到第一个 <|im_end|>，以及上一个助手说完到下一个 <|im_end|>
        start_mask_idx = 0
        for end_idx in user_end_indices:
            # 从上一个掩码结束的位置到当前用户回合结束的位置，都设置为-100
            for i in range(start_mask_idx, end_idx + 1):
                labels[i] = IGNORE_INDEX
            # 更新下一个掩码的起始位置，跳过 <|im_start|> assistant\n
            start_mask_idx = end_idx + 3 

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    # 返回结果需要 padding，交给 DataCollator 处理
    return {"input_ids": all_input_ids, "labels": all_labels}

dataset = raw_datasets.map(preprocess_and_mask_labels, batched=True, remove_columns=raw_datasets.column_names)

# 5. 配置训练参数
training_args = TrainingArguments(
    output_dir=ft_model_path,       # 输出目录
    num_train_epochs=3,             # 训练轮次
    per_device_train_batch_size=1,  # 由于序列更长，减小批次大小
    gradient_accumulation_steps=8,  # 增加梯度累积步数
    optim="adamw_torch",            # 优化器
    learning_rate=2e-4,             # 学习率
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

# 6. 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    # 使用会处理padding和label的DataCollator
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

# 7. 开始训练
trainer.train()

# 8. 保存微调后的模型
trainer.save_model(ft_model_path)
tokenizer.save_pretrained(ft_model_path)

print(f"微调完成，模型已保存到 {ft_model_path}")
