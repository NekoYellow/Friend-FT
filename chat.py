import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- 1. 配置模型路径、设备和自定义Token ---
FT_MODEL_PATH = "./Qwen3-0.6B-FT"
BASE_MODEL_PATH = "./Qwen3-0.6B"
MESSAGE_SPLIT_TOKEN = "<|split|>" # 你新增的自定义Token
SYSTEM_PROMPT = "你将扮演用户的好友'There'与用户在线聊天。"

# 自动选择设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- 2. 加载模型和Tokenizer ---
print("Loading model and tokenizer...")

# 2.1 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(FT_MODEL_PATH, trust_remote_code=True)

# 2.2 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)

# 2.3 调整词表大小
model.resize_token_embeddings(len(tokenizer))

# 2.4 加载LoRA/PEFT Adapter
model = PeftModel.from_pretrained(model, FT_MODEL_PATH)

# 2.5 将模型移动到指定设备
model.to(DEVICE)
model.eval()

# 2.6 设置pad_token_id
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

print("Model loaded successfully. You can start chatting now.")
print("Type 'exit' or 'quit' to end the conversation.")
print("-" * 30)

# --- 3. 主聊天循环 ---
messages = []

while True:
    try:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        print()

        messages.append({"role": "user", "content": user_input})

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.6,           # 提高温度以增加多样性，但不过高以防胡言乱语
                top_p=0.9,                 # 使用 top_p 进行核采样，动态选择候选词
                repetition_penalty=1.1,    # 对重复词进行轻微惩罚
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")],
            )

        response_ids = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=False)
        response = response.replace("<|im_end|>", "")

        # 处理自定义的分割Token，将其替换为换行符
        for msg in response.split(MESSAGE_SPLIT_TOKEN):
            clean_msg = msg.strip()
            if clean_msg:
                print(f"There: {clean_msg}")
        print()
        messages.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
        break