# MyLLM: My Large Language Model in action

记录一下动手边做边学Large Language Model的过程。不要当一个调包侠，要知道细节，以及为什么这么做。

## TODO List

### ✅ 已完成

- [x] [2025.2.11] 训练一个分词器
- [x] [2025.2.13] 参考minimind复现大模型预训练，并参考qwen2和llama将实现迁移到HF的Transformers规范

### 🕥 进行中

- [ ] [2025.2.15] 改进Attention机制以及引入MoE
  - [x] [2025.2.15] SDPA支持
    - [ ] SDPA模型训练中，同样batch峰值显存消耗与训练时间下降了一半，增加batch
  - [ ] FlashAttention支持
  - [ ] 稀疏注意力与线性注意力
  - [ ] Deepseek V3的MLA和MoE

### 🗓 计划内

- [ ] Generation相关？
- [ ] 修改RoPE
- [ ] 添加shortcut MoE
- [ ] 强化学习与蒸馏
- [ ] 手动配置DeepSpeed分布式计算，因为现在的运行方法在`io_operation`存在问题
  - [ ] 是否需要自定义Trainer（自定义DeepSpeed分布式训练）？
- [ ] ...

### 更新日志

- [2025.2.13]
完成Tokenizer训练，Debug跑通基础模型

- [2025.2.15]
修复PretrainDataset在生成labels时会比input_ids多一个的问题，增加对SDPA实现的FlashAttention支持

## 🧑‍💻 QA

1. 怎么让token的embedding在某种语义环境下呈现出特定的可加性呢？比如：

    - 迷你版的番茄是圣女果：Vector[迷你版的] + Vector[番茄] = Vector[圣女果] ≠ Vector[奇异果]
    - 也就是说，不能仅仅将向量表示学习局限于共现性约束

2. Debug中发现，由`GenerationMixin`生成的`position_ids`中，`attention_mask == 0`的部分全为1，这个目的是什么？

3. [2025.2.15] Padding token的嵌入应该随着模型的更新而更新吗？

   [2025.2.16] 看了一下Qwen2ForCausalLM的embed_tokens对应的padding嵌入，并不为0向量

4. [2025.2.15] 直接修改eager_attn为sdpa在训练过程中出现了nan值

   1. [2025.2.16] 原来写的`attention_mask(bsz, n, seq_len, seq_len)`，在`padding`的行全为`mask`，这个使得在计算`softmax`出现除0，参照Qwen2的处理方式，如果前k个`token`是`padding`，那么`attention mask`对应的`[k, seq_len]`全为0，而不是最小值填充，以规避上述情况。
   2. [2025.2.16] 解决上述问题后，Forward Passing没有再出现Nan，而Backward引发了Nan，在把`MyLLMRMSNorm`去掉的情况下不会出现Nan，问题还在排查。
   3. [2025.2.19] Backward的Nan问题已解决，在Debug过程中，使用了Qwen2的模型文件进行逐个模块替换，最后将问题定位在`nn.Embedding(..., padding_idx=0)`参数设置上，该设置使得在`MyLLMForCausalLM`的`lm_head`在反向传播时`grad_output`（该参数通过backward钩子函数获得）出现了Nan，把这个参数去掉就好了，但是作用机理仍不明。模块替换实验参考：[modeling_qwen.py](./test/modeling_qwen.py)

## 1. Tokenizer

使用BPE算法进行分词，参照[Minimind](https://github.com/jingyaogong/minimind)分词器的训练.

**使用数据**:

1. [Minimind](https://github.com/jingyaogong/minimind)分词器训练语料

2. [IndustryCorpus 2.0](https://data.baai.ac.cn/details/BAAI-IndustryCorpus-v2)的编程、人工智能-机器学习部分

**数据处理**:

- 核心步骤：

    1. 直接训练Tokenizer
    2. 检查Merges合并规则，人工提取不合理的合并规则
    3. 将连续符号使用正则表达式进行替换

- 参考： [0-tokenizer.ipynb](./notebooks/0-tokenizer.ipynb)

**其他参考资料**：

[Huggingface NLP Course - 6.🤗TOKENIZERS库](https://huggingface.co/learn/nlp-course/zh-CN/chapter6/1?fw=pt)

### **1.2 NOTES:**

2025.2.11

1. 在不进行原始数据清洗的情况下进行分词器训练会出现很多的连`空格`格和`=`等符号

2. 分词器解码在特殊token后会自动附加空格，这个bug目前还没解决

## 2. Pre-Train

参照[Qwen2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py)实现模型主要架构，参照[LLAMA](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)实现Eager GQA，参照[Qwen2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py)实现SDPA，参照[十分钟读懂旋转编码（RoPE）](https://zhuanlan.zhihu.com/p/647109286)实现旋转位置编码。

**使用数据:**

**数据处理:**

- 核心步骤（试了一下预处理速度太慢，目前训练速度也太慢了，所以先做模型结构去了）：

    1. 计算文本hash值，对重复部分进行再hash
    2. 对再hash重复部分检查其是否相同，相同则删除
    3. 对超长部分文本进行分段

- 参考： [1-pretrain.ipynb](notebooks/1-pretrain.ipynb)

### **2.2 NOTES:**

2025.2.15

**①** 基础实验已跑通，6卡训练耗时约41小时，Loss如图：

  ![alt text](assets/pretrain-loss.png)

**②** Debug Qwen2模型推理过程，对模型文件进行修改，增加对SDPA的支持

1. 关注GenerationMixin做的n件事
   1. 生成position_ids等

   2. 初始化Cache实例

      - 检查是否有传入的Cache实例
      - 检查模型支持的Cache类型
      - 没有支持类型时检查是否支持动态Cache类型

   3. 将准备好的参数传入生成方法，由生成方法调用模型的`forward`方法

      `GenerationMixin`集成了多种采样方法，如`GreedySearch`，`BeamSearch`以及`Sample`等；

2. Flash Attention

   - 参考资料

    [Bilibili: Flash Attention 为什么那么快？原理讲解](https://www.bilibili.com/video/BV1UT421k7rA)

    [Bilibili: 【7】Flash Attention 原理讲解](https://www.bilibili.com/video/BV17CPkeEEHH)

   - 实现部分
