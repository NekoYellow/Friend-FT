# 微信好友聊天微调

在消费级显卡上用好友的聊天记录微调一个模仿好友聊天风格的语言模型。

使用开源的[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)作为基础模型，加LoRA并解冻Embedding和LMHead（以支持新引入的表示消息间隔的特殊token）进行微调。

## 步骤

1. 下载[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)模型参数到当前目录：`hf download Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B`

2. 用[chatlog](https://github.com/sjzar/chatlog)导出JSON格式的聊天记录

3. 用`filter_text.py`和`process_chat.py`处理文本

4. 用`finetune.py`微调模型

5. 用`infer.py`测试模型（单步生成），或用`chat.py`和模型在命令行多轮对话

## 优化

### 数据

数据质量直接决定模型表现，值得进行针对性优化。为保证结果质量大致需要至少5000条问答文本。可以修改`process_chat.py`中的参数来适配特定好友的聊天特征，或者尝试改用`process_chat_multi.py`来抽取多轮对话数据。

如果数据量过于庞大，可以考虑使用`filter_dataset.py``filter_dataset_multi.py`来对上一步抽取的对话数据进行筛选，同样根据好友聊天特征自定义。

可以尝试在微调数据和推理时都加一个system prompt，`filter_dataset_multi.py`里已实现。

这里还引入了新的特殊token`<|split|>`来表示一条消息的结束（不是一句话的结束）。这样模型可以学到具有好友风格的断句。

### 超参数

训练和推理用的超参数也推荐自定义。为了达到更佳表现，`num_train_epochs`可以适当增加；显存充裕的情况下，`optim`可以换成使用标准32位float的优化器，适当调大`per_device_train_batch_size`。显卡支持BFloat16的可以改用`bf16`来加载模型和训练。推理时的生成参数也可以视情况调整。

## 效果

### 原始生成结果

用户问“在干嘛”

基础模型（Qwen3-0.6B）：

```
<|im_start|>user
在干嘛<|im_end|>
<|im_start|>assistant
<think>
好的，我现在需要处理用户的问题“在干嘛”。首先，我要确定用户是在问我在做什么。作为AI助手，我的主要功能是提供帮助和解答问题， 但有时用户可能误以为我有其他功能。我需要礼貌地回应，同时明确说明我的实际用途。

接下来，我应该检查用户可能的意图。用户可能想了解我的功能，或者遇到了某个问题需要帮助。我需要确保回答友好且专业，避免让用户 感到被误解。同时，要保持语言简洁，避免冗长。

另外，考虑到用户可能希望得到更具体的回答，我应该提供一些通用的信息，比如我的功能范围，以及如何帮助用户。这样既展示了我的能 力，又保持了对话的开放性。

最后，要确保回答符合平台的规定，比如使用正确的格式和礼貌的语气。检查是否有任何可能的错误，比如拼写或语法问题，确保回答的准 确性。
</think>

我是在帮助用户解答问题、提供支持或提供相关的信息。如果您有任何问题或需要帮助，请随时告诉我！ 😊<|im_end|>
```

微调模型：

```
<|im_start|>user
在干嘛<|im_end|>
<|im_start|>assistant
哎<|split|>我感觉<|split|>我也不怎么忙<|split|>我最近是<|split|>在准备高考<|im_end|>
```

### 对话效果

<img width="512" alt="chat" src="https://github.com/user-attachments/assets/1b284f90-ac30-48ca-9188-d3309dfadfda" />
