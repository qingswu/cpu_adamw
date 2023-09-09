import torch
from cpu_adamw import DeepSpeedCPUAdam


model_size = 64
dtype = torch.float

cpu_data = torch.randn(model_size, device="cpu").to(dtype)
cpu_param = torch.nn.Parameter(cpu_data)
cpu_optimizer = DeepSpeedCPUAdam([cpu_param])

cpu_param.grad = torch.randn(model_size, device=cpu_param.device).to(cpu_param.dtype)

cpu_optimizer.step()

# python seems unloaded _cpu_adamw (and set it to None) before calling __del__ of DeepSpeedCPUAdam,
# explicitly del cpu_optimizer to avoid _cpu_adamw None error
del cpu_optimizer
