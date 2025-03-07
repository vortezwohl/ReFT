import os

# from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import torch

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model/models--Qwen--Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, cache_dir="./model/models--Qwen--Qwen2.5-0.5B-Instruct")

# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,  # 任务类型为因果语言模型
#     r=16,  # LoRA 的秩
#     lora_alpha=32,  # LoRA 的 alpha 参数
#     target_modules=["q_proj", "v_proj"],  # 需要应用 LoRA 的模块
#     lora_dropout=0.05,  # LoRA 的 dropout 概率
#     bias="none"  # 不对偏置项进行 LoRA
# )
#
# # 应用 LoRA 到模型
# model = get_peft_model(model, lora_config)

ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=2e-6,  # 学习率
    batch_size=1,  # 批量大小
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    ppo_epochs=4,  # PPO轮数
    max_grad_norm=0.3,  # 最大梯度范数
    init_kl_coef=0.05,  # 初始KL散度系数
    target_kl=0.1,  # 目标KL散度
    gamma=0.8,  # 折扣因子
    lam=0.95,  # 优势函数的lambda参数
    cliprange=0.2,  # PPO裁剪范围
    cliprange_value=0.2,  # 价值函数的裁剪范围
    vf_coef=1.0,  # 价值函数的系数
    # gen_kwargs={"max_new_tokens": 32},  # 生成配置
)

# 初始化PPO训练器
ppo_trainer = PPOTrainer(
    model=model,
    config=ppo_config,
    tokenizer=tokenizer,
)


def reward(response):
    # if 'user' in response.lower():
    #     return torch.tensor(-1.0)
    if len(response) >= 50:
        if len(response) < 100:
            return torch.tensor(0.5)
        else:
            return torch.tensor(1.0)
    else:
        if len(response) > 30:
            return torch.tensor(0.0)
        else:
            return torch.tensor(-0.1)


epochs = 1024
prompts = [
    "你好啊.",
    "你叫什么名字?",
    "很高兴认识你.",
    "你是什么样的人?",
    "你吃了吗.",
    "今天天气不错.",
    "认识你很开心.",
    "你在忙什么?"
]

for i, prompt in enumerate(prompts):
    prompts[i] = f'User:{prompt}\nAssistant:'

for epoch in range(epochs):
    for prompt in prompts:
        query_tensor = tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)
        response_tensor = ppo_trainer.generate(query_tensor).squeeze(0)
        response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)
        reward_value = reward(response_text)
        response_text = response_text.replace('\n', '\\n')
        print(f"Epoch {epoch}, Prompt: {prompt}, Response: {response_text}, Reward: {reward_value}")
        ppo_trainer.step([query_tensor], [response_tensor], [reward_value])

model.save_pretrained("./output/test")
tokenizer.save_pretrained("./output/test")
