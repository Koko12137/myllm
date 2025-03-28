# MyLLM: My Large Language Model in action

记录一下动手边做边学Large Language Model的过程。不要当一个调包侠，要知道细节，以及为什么这么做。

## 更新日志

- [2025.3.28]
  完成预训练，进入Response-Based蒸馏阶段；完成蒸馏模型代码实现，开始蒸馏模型训练；训练中加入困惑度测试；上阶段实验结果整理中

- [2025.3.23]
  对比实验发现直接CoE训练存在难以收敛问题；参照[阶跃星辰调参实验](https://arxiv.org/abs/2503.04715)将训练batch_size和学习率增大，预训练预计7天完成，计划将对困惑度进行测试，并增加上下文长度进行全量微调与模型蒸馏；参考[阅读笔记](./notes/reinforcement.md)完成数据合成（谨防**阿里云百炼大模型**的批量推理超出预算）

- [2025.3.17]
  
  - BUG修复：
    1. 修复MoE代码错误，修复MoE训练过程中某些专家未激活时反向传播梯度不存在而造成的`all_gather`出现死锁的问题（QA4）
    2. 修复使用Qwen2分词器的`vocab_size`初始化`nn.Embedding`时索引错误引发的一系列问题
  
  - 修改：
    1. tokenizer训练使用bytelevel分词会对中文产生影响，而使用unidcode分词会大幅增加训练时间，目前手搓了一个训练脚本但Python速度太慢，考虑到白盒蒸馏模型需要使用教师模型的词表，本项目暂定使用Qwen2的分词器

  - 完成：
    1. 增加[Deepseek无损MoE负载均衡](./notes/moe)支持
    2. 增加了一篇蛮有意思的文章的阅读笔记：[为什么Qwen能自我改进推理，Llama却不行？斯坦福找到了原理](./notes/reinforcement.md)
    3. 完成SDPA对比训练，但损失函数仍止步于2.6左右，因此修改了训练数据集和调度器以及学习率，再次测试预训练
    4. 清洗[BAAI/IndustryCorpus2](./scripts/clean_data.py)数据集

  - 进行中：
    训练MoE模型

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

- 数据工程
  - [x] 清洗预训练数据
    - [2025.3.23] **在阿里云百炼平台提交了几个[数据合成](notebooks/1-data-compose.ipynb)的批量推理任务，运行完后被短信告知`欠费870元`，天塌了**

- 训练
  - [x] [2025.2.11] 训练一个分词器
  - [x] [2025.2.13] 参考minimind复现大模型预训练，并参考qwen2和llama将实现迁移到HF的Transformers规范
  - [x] [2025.3.2] SDPA训练结果不太理想，估计是OneCycle调度器的问题，试试看修改一下重新训练
  - [x] [2025.3.13] 修改调度器为Warmup，但效果仍没有明显提升，推测是数据多元性的问题或者是学习率的问题
  - [x] [2025.3.14] 将padding从Backward中剔除（实验发现会造成训练的不稳定）
  - [x] [2025.3.23] 训练MoE模型
    - [x] MoE专家负载均衡训练
  - [x] [2025.3.23] 对学习率、batch_size、CoE、MoE、Dense等参数进行对比实验

- 模型改进
  - [x] [2025.2.15] 改进Attention机制以及引入MoE
    - [x] [2025.2.15] SDPA支持
      - [x] SDPA模型训练中，同样batch峰值显存消耗与训练时间下降了一半，增加batch
    - [ ] ~~FlashAttention支持~~(暂时搁置，SDPA也挺快的)
    - [x] [2025.3.13] 基础MoE与专家负载均衡训练
      - [x] FFN MoE、共享专家、Chain of Experts
      - [x] Temperature平滑与Top-K采样专家
      - [x] 专家负载均衡训练

- 工程化
  - [x] [2025.2.20] Generation相关
    - [x] use_cache支持：支持DynamicCache

- 测试模型
  - [x] [2025.3.28] 在蒸馏学习中加入了困惑度测试
  
- 其他
  - [x] [2025.3.14] 解耦Dataset和Tokenizer，Batch后使用Collator进行填充
  - [x] [2025.3.28] 增加对DeepNorm的支持

### 🕥 进行中

- [ ] [2025.3.2] 做数据做数据！！！！：
  - [ ] 找点免费的api，根据种子合成领域迁移数据以及短CoT数据，执行全量微调

- [ ] 模型改进
  - [ ] [2025.3.5] 改进Attention机制以及引入MoE
    - [ ] Kimi Mixture of Block Attention
      - Kimi MoBA具体实现方式是什么？会不会影响KVCache？
      - 切分QKV的时候能不能用一个低秩矩阵计算Q分块和K分块的相关值？
      - 能不能逐层对原序列长度进行缩减？让模型把一段内容读薄？
    - [ ] 基于已训练的Attention参数矩阵，直接划分Mixture of Block Attention
  - [ ] [2025.3.28] 蒸馏模型
    - [ ] [2025.3.28] 使用1.5B模型作为教师模型，使用原数据集进行一个Epoch蒸馏模型
  - [ ] 其他改进
    - [ ] 增加一个工厂类用于指定替换模型的部分模块，并返回一个可训练的模型

- [ ] [2025.3.14] 训练
  - [ ] 使用合成数据集与Deepseek-R1蒸馏数据集，加入教师模型，进行白盒蒸馏
    - [ ] Response-based 知识蒸馏
    - [ ] 部分中间层知识蒸馏，如Attention层对齐教师模型
  - [ ] SFT
    - [ ] **教师-学生**混合模型有监督微调

- [ ] 测试模型
  - [ ] 对不同模型进行困惑度测试
  - [ ] 检查专家负载均衡情况
  - [ ] 对比不同Norm的训练难度与最终效果差异

### 🗓 计划内

- 精细化训练
  - [ ] SFT：领域迁移全量微调、LoRA微调
    - 模拟退火防止过拟合？
    - 将新的专家插入MoE进行SFT？
  - [ ] 黑盒、白盒蒸馏
    - [ ] 仅微调学生模型，使得学生模型配合教师模型输出？
  - [ ] 强化学习
    - [ ] 以往研究结果发现使用蒙特卡洛树进行采样并不能使得模型得到更好训练，需要渐进式训练
      - Kimi的训练方法？Deepseek的训练方法？
    - [ ] 由模型生成Policy，再利用Deepseek R1进行评分？即R1作为Critic？
    - [ ] 合成深度思考数据集
      - [ ] 弱智吧
      - [ ] 海龟汤、狼人杀
  
- 模型改进
  - [ ] 对MoE、Attention进行花式修改
    - [ ] Block MoE
    - [ ] 稀疏注意力与线性注意力
    - [ ] Deepseek V3的MLA

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

2. [2025.2.15] Padding token的嵌入应该随着模型的更新而更新吗？

   [2025.2.16] 看了一下Qwen2ForCausalLM的embed_tokens对应的padding嵌入，并不为0向量，而在初始化时，padding是全0向量

   [2025.3.5] 之前手动把padding在反向传播时mask掉会造成不稳定，可能的原因参考视频[Bilibili 良睦路程序员：transformers一个非常严重的bug——在使用梯度累计的时候 loss不等效](https://www.bilibili.com/video/BV1oY1aYzEVi)，具体原因待调查

3. [2025.2.15] 直接修改eager_attn为sdpa在训练过程中出现了nan值

   1. [2025.2.16] 原来写的`attention_mask(bsz, n, seq_len, seq_len)`，在`padding`的行全为`mask`，这个使得在计算`softmax`出现除0，参照Qwen2的处理方式，如果前k个`token`是`padding`，那么`attention mask`对应的`[k, seq_len]`全为0，而不是最小值填充，以规避上述情况。
   2. [2025.2.16] 解决上述问题后，Forward Passing没有再出现Nan，而Backward引发了Nan，在把`MyLLMRMSNorm`去掉的情况下不会出现Nan，问题还在排查。
   3. [2025.2.19] Backward的Nan问题已解决，在Debug过程中，使用了Qwen2的模型文件进行逐个模块替换，最后将问题定位在`nn.Embedding(..., padding_idx=0)`参数设置上，该设置使得在`MyLLMForCausalLM`的`lm_head`在反向传播时`grad_output`（该参数通过backward钩子函数获得）出现了Nan，把这个参数去掉就好了，但是作用机理仍不明。模块替换实验参考：[modeling_qwen.py](./test/modeling_qwen.py)

4. [2025.3.4] MoE训练出现了问题，会莫名其妙卡在某个step然后一动不动，也不报任何bug

    [2025.3.17] 在MoE模型中，通过Top-K获取每个Token指定的专家，而在原来的代码中，当某些专家没有被任何Token激活时，其反向传播的梯度不存在，这样会造成在分布式执行all_gather时某些进程持续等待该部分梯度，造成死锁。参考[[BUG] The NCCL timed out while using the zero3 model. How can I solve this problem? #5066](https://github.com/deepspeedai/DeepSpeed/issues/5066)

5. [2025.3.23] 训练设置了一个对照试验，实验对比`num_logits_to_keep`设置为100与不设置的区别，结果发现设置为100时训练过程极其不稳定，且收敛难度加大，推测原因如下：

   - 在模型训练阶段，假设我们每个batch都只考虑最后一个输出token的loss，且每个batch的上下文不一致，则每个参数的梯度为所有batch的平均值；而如果我们考虑最后k个tokens的loss，在causal attention的前提下，单个batch中越靠前的token所造成的loss会被放得越大，使得模型难以聚焦到最后一个token的生成中，这造成了训练容易进入局部最优
   - 训练时我们对数据集进行洗牌的目的在于尽可能保证每个sample不一样，避免进入局部最优状态，而在同一上下文中计算多个tokens的loss与这一做法的目的背道而驰，因此在训练时应该尽量避免这种情况
   - 验证：使用钩子函数抓取每个batch的梯度，对比其均值和方差

## 📚 参考资料

- [Tokenzier](./notes/tokenzier.md)
- [Pretrain](./notes/pretrain.md)
  - [MoE](./notes/moe.md)
- [Reinforcement Learning](./notes/reinforcement.md)
