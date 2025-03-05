# MyLLM: My Large Language Model in action

记录一下动手边做边学Large Language Model的过程。不要当一个调包侠，要知道细节，以及为什么这么做。

## 更新日志

- [2025.3.5]
修改MoE，增加共享专家支持，增加[Chain of Experts](https://github.com/ZihanWang314/CoE/tree/main)支持

- [2025.3.4]
更新学习计划，将任务进行分类；

- [2025.3.2]
增加了对DynamicKVCache的支持；增加了对FFN MoE的支持；将原模型文件按照不同模块进行拆分；完成了基于SDPA实现的模型训练，但是发现似乎收敛难度有点大，再做个对比实验看看是不是学习率调度器的问题

- [2025.2.15]
修复PretrainDataset在生成labels时会比input_ids多一个的问题，增加对SDPA实现的FlashAttention支持

- [2025.2.13]
完成Tokenizer训练，Debug跑通基础模型

## TODO List

### ✅ 已完成

- [x] [2025.2.11] 训练一个分词器

- [x] [2025.2.13] 参考minimind复现大模型预训练，并参考qwen2和llama将实现迁移到HF的Transformers规范

- [x] [2025.2.15] 改进Attention机制以及引入MoE
  - [x] [2025.2.15] SDPA支持
    - [x] SDPA模型训练中，同样batch峰值显存消耗与训练时间下降了一半，增加batch
  - [ ] ~~FlashAttention支持~~(暂时搁置，SDPA也挺快的)
  - [x] 基础MoE与专家负载均衡训练
    - [x] FFN MoE

- [x] [2025.2.20] Generation相关
  - [x] use_cache支持：支持DynamicCache

### 🕥 进行中

- [ ] [2025.3.2] 训练
  - [ ] SDPA训练结果不太理想，估计是OneCycle调度器的问题，试试看修改一下重新训练

- [ ] [2025.3.2] 改进Attention机制以及引入MoE
  - [ ] 基础MoE与专家负载均衡训练
  - [ ] Kimi Mixture of Block Attention
    - Kimi MoBA具体实现方式是什么？会不会影响KVCache？
    - 切分QKV的时候能不能用一个低秩矩阵计算Q分块和K分块的相关值？
    - 能不能逐层对原序列长度进行缩减？让模型把一段内容读薄？

- [ ] [2025.3.2] 做数据做数据！！！！：
  - [ ] 清洗预训练数据
  - [ ] 找点免费的api，根据种子合成领域迁移数据以及短CoT数据，执行全量微调

- [ ] 测试模型

### 🗓 计划内

- 精细化训练
  - [ ] SFT：领域迁移全量微调、LoRA微调
    - 模拟退火防止过拟合？
  - [ ] 黑盒、白盒蒸馏
  - [ ] 强化学习
    - [ ] 以往研究结果发现使用蒙特卡洛树进行采样并不能使得模型得到更好训练，需要渐进式训练
      - Kimi的训练方法？Deepseek的训练方法？
    - [ ] 由模型生成Policy，再利用Deepseek R1进行评分？即R1作为Critic？
  
- 模型改进
  - [ ] 对MoE、Attention进行花式修改
    - [ ] 稀疏注意力与线性注意力
    - [ ] Deepseek V3的MLA和MoE

- 工程化
  - [ ] 训练与推理加速工程
    - [ ] DeepSpeed MoE接口？
    - [ ] Deepseek FlashMLA推理加速？
    - [ ] Deepseek 3FS加速超大规模训练数据集加载？

- 其他
  - [ ] 如何实现长文本输入？如何实现长文本的大海捞针？RoPE有没有必要更平滑？
  - [ ] 手动配置DeepSpeed分布式计算，因为现在的运行方法在`io_operation`存在问题
    - [ ] 是否需要自定义Trainer（自定义DeepSpeed分布式训练）？

- 待续...

## 🧑‍💻 QA

1. 怎么让token的embedding在某种语义环境下呈现出特定的可加性呢？比如：

    - 迷你版的番茄是圣女果：Vector[迷你版的] + Vector[番茄] = Vector[圣女果] ≠ Vector[奇异果]
    - 也就是说，不能仅仅将向量表示学习局限于共现性约束

2. Debug中发现，由`GenerationMixin`生成的`position_ids`中，`attention_mask == 0`的部分全为1，这个目的是什么？

3. [2025.2.15] Padding token的嵌入应该随着模型的更新而更新吗？

   [2025.2.16] 看了一下Qwen2ForCausalLM的embed_tokens对应的padding嵌入，并不为0向量，而在初始化时，padding是全0向量

   [2025.3.5] 之前手动把padding在反向传播时mask掉会造成不稳定，可能的原因参考视频[Bilibili 良睦路程序员：transformers一个非常严重的bug——在使用梯度累计的时候 loss不等效](https://www.bilibili.com/video/BV1oY1aYzEVi)，具体原因待调查

4. [2025.2.15] 直接修改eager_attn为sdpa在训练过程中出现了nan值

   1. [2025.2.16] 原来写的`attention_mask(bsz, n, seq_len, seq_len)`，在`padding`的行全为`mask`，这个使得在计算`softmax`出现除0，参照Qwen2的处理方式，如果前k个`token`是`padding`，那么`attention mask`对应的`[k, seq_len]`全为0，而不是最小值填充，以规避上述情况。
   2. [2025.2.16] 解决上述问题后，Forward Passing没有再出现Nan，而Backward引发了Nan，在把`MyLLMRMSNorm`去掉的情况下不会出现Nan，问题还在排查。
   3. [2025.2.19] Backward的Nan问题已解决，在Debug过程中，使用了Qwen2的模型文件进行逐个模块替换，最后将问题定位在`nn.Embedding(..., padding_idx=0)`参数设置上，该设置使得在`MyLLMForCausalLM`的`lm_head`在反向传播时`grad_output`（该参数通过backward钩子函数获得）出现了Nan，把这个参数去掉就好了，但是作用机理仍不明。模块替换实验参考：[modeling_qwen.py](./test/modeling_qwen.py)

5. [2025.3.4] MoE训练出现了问题，会莫名其妙卡在某个step然后一动不动，也不报任何bug

## 📚 参考资料

- [Tokenzier](./notes/tokenzier.md)
- [Pretrain](./notes/pretrain.md)
- [Reinforcement Learning](./notes/reinforcement.md)
