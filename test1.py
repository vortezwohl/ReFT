# imports
import os

import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# get models
model = AutoModelForCausalLMWithValueHead.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', cache_dir="./model")
ref_model = create_reference_model(model)

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', cache_dir="./model")
tokenizer.pad_token = tokenizer.eos_token

# initialize trainer
ppo_config = PPOConfig(batch_size=1, mini_batch_size=1)

# encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt")

# get model response
response_tensor = respond_to_batch(model, query_tensor)
print(tokenizer.decode(response_tensor[0]))

# create a ppo trainer
ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer)

# define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0)]

# train model for one step with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)

print(train_stats)
