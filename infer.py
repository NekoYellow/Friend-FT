import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. 先加载tokenizer
ft_model_path = "./Qwen3-0.6B-FT"
tokenizer = AutoTokenizer.from_pretrained(ft_model_path, trust_remote_code=True)

# 2. 加载基础模型并调整词表大小
base_model_path = "./Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model.resize_token_embeddings(len(tokenizer))

# 3. 加载adapter
model = PeftModel.from_pretrained(model, ft_model_path)

# 检查 tokenizer 是否有 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
# -----------------------------------------

messages = [
    {"role": "user", "content": "哎\n相当女人"}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
)

# 因为我们的输入没有 padding，所以 attention_mask 就是一个和 input_ids 同样形状、内容全为 1 的张量
attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
# --------------------------------

print("Generating response...")
outputs = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

response_ids = outputs[0] # [input_ids.shape[-1]:]
response = tokenizer.decode(response_ids) #, skip_special_tokens=True)

print(response)