# Mixture of Experts

## Introduction

## 主要架构

### 1. FFN MoE

在传统的MoE中，每个token会被路由到K个专家，然后再将这K个专家的输出进行加权求和。这种方法可以理解为，针对单个词元这种细粒度进行dispatch是不同专家考虑某一词元中不同含义的部分（因为专家没办法看到全部的序列），然后对经过专家处理的多个结果进行整合，实现词义调整。

```python
zeros = torch.zeros_like(x)     # Shape: (bsz * seq_len, hidden_size)
        
# Compute the expert output
for i, expert in enumerate(self.experts):
    # Create a expert mask
    expert_mask = (ids == i).any(dim=-1)
    # Mask the input tensor
    x_masked = x[expert_mask]
    
    if x_masked.size(0) == 0:
        # Skip the expert if the mask is empty
        continue
    
    # Compute the expert output
    expert_out = expert(x_masked)
    # Get expert score
    scores = logits[expert_mask][:, i]
    # Scale the expert output
    expert_out = expert_out * scores.unsqueeze(-1)
    # Update the expert output
    zeros[expert_mask] += expert_out
```

还有另一种做法可以尝试，即K个专家处理某一词元时，最终结果由K个专家的输出进行拼接

### 2. Chain of Experts

原作对不同层的MoE使用了不同的Route，但所有层的Expert共享相同的参数。

- 参考Blog：[Chain-of-Experts: 释放MoE专家的沟通潜能](https://sandy-server-87f.notion.site/Chain-of-Experts-MoE-1ab9bb750b79801bbfebf01ae9a77b3f)
- 参考代码实现：[ZihanWang314/CoE/config/models/coe_deepseekv2/modeling_coe.py](https://github.com/ZihanWang314/CoE/blob/main/config/models/coe_deepseekv2/modeling_coe.py)

### 3. Deepseek 无损负载均衡

[DeepSeek-V3 解读3：无辅助损耗的负载均衡](https://zhuanlan.zhihu.com/p/25228000281)
