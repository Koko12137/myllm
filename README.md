# MyLLM: My Large Language Model in action

记录一下动手边做边学Large Language Model的过程

## TODO List

### 2025.2.11

- [x] 训练一个分词器

### 2025.2.13

- [ ] 参考minimind和qwen2复现大模型预训练，并将其迁移到HF的Transformers规范

### 未完成

- [ ] 添加MLA和MoE
- [ ] 修改RoPE
- [ ] 添加shortcut MoE
- [ ] 强化学习与蒸馏
- [ ] ...

## QA

1. 怎么让token的embedding在某种语义环境下呈现出特定的可加性呢？比如：

    - 迷你版的番茄是圣女果：Vector[迷你版的] + Vector[番茄] = Vector[圣女果] ≠ Vector[奇异果]
    - 也就是说，不能仅仅将向量表示学习局限于共现性约束

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

## 2. Pre-Train

参照[Qwen2]()实现模型主要架构，参照[]()实现GQA，参照[]()实现旋转位置编码

**使用数据**

**数据处理**

- 核心步骤：

    1. 计算文本hash值，对重复部分进行再hash
    2. 对再hash重复部分检查其是否相同，相同则删除
    3. 对超长部分文本进行分段

- 参考： []()
