from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# 定义PPO配置
ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-5,  # 学习率
    batch_size=1,  # 批量大小
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    ppo_epochs=4,  # PPO轮数
    max_grad_norm=0.3,  # 最大梯度范数
    init_kl_coef=0.05,  # 初始KL散度系数
    target_kl=6.0,  # 目标KL散度
    gamma=1.0,  # 折扣因子
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


# 定义基于规则的奖励函数
def reward_function(response):
    target_string = "目标字符串"  # 你可以根据需要修改目标字符串
    if target_string in response:
        return 1.0  # 如果检测到目标字符串，给予正奖励
    else:
        return -1.0  # 如果未检测到，给予负奖励


# 示例输入文本
prompts = [
    "这是一个需要生成包含目标字符串的文本。",
    "另一个需要生成包含目标字符串的文本。",
]

# 强化微调的训练循环
for epoch in range(10):  # 训练10个epoch
    for prompt in prompts:
        # 生成模型的输出
        query_tensors = tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)
        response_tensors = ppo_trainer.generate(query_tensors).squeeze(0)
        response_texts = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # 计算奖励
        rewards = [reward_function(response) for response in response_texts]

        # 打印奖励和生成的文本
        print(f"Epoch {epoch}, Prompt: {prompt}, Response: {response_texts[0]}, Reward: {rewards[0]}")

        # 优化模型
        ppo_trainer.step([query_tensors], [response_tensors], [torch.tensor(rewards)])

# 保存微调后的模型
model.save_pretrained("./output/test")
tokenizer.save_pretrained("./output/test")
