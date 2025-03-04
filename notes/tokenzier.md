# 1. Tokenizer

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

## **1.2 NOTES:**

### [2025.2.11]

1. 在不进行原始数据清洗的情况下进行分词器训练会出现很多的连`空格`格和`=`等符号

2. 分词器解码在特殊token后会自动附加空格，这个bug目前还没解决

## 1.3 TODO

- [ ] 补一个基于Python的手搓BPE算法及速度优化
