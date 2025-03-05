# Mixture of Experts

## Introduction

## 主要架构

### 1. FFN MoE

### 2. Chain of Experts

原作对不同层的MoE使用了不同的Route，但所有层的Expert共享相同的参数。

- 参考Blog：[Chain-of-Experts: 释放MoE专家的沟通潜能](https://sandy-server-87f.notion.site/Chain-of-Experts-MoE-1ab9bb750b79801bbfebf01ae9a77b3f)
- 参考代码实现：[ZihanWang314/CoE/config/models/coe_deepseekv2/modeling_coe.py](https://github.com/ZihanWang314/CoE/blob/main/config/models/coe_deepseekv2/modeling_coe.py)
