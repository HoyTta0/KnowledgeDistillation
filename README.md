# 知识蒸馏在文本方向上的应用

模型相关等内容在[**我的博客**](https://blog.csdn.net/HoyTra0/article/details/106238382)有具体介绍。

## 目录

- [更新日志](#更新日志)
- [运行环境](#运行环境)
- [使用说明](#使用说明)
- [模型实现](#模型实现)
  - [代码结构](#代码结构)
- [模型效果](#模型效果)
- [公开数据集测试效果](#TNEWS测试效果)
- [已知问题](#已知问题)
- [TODO](#TODO)
- [参考链接](#参考链接)

## 更新日志

### 2021.11.10

>  大幅改动，整理数据格式，优化代码，使用最新第三方包，旧版本代码在v0.1分支查看。

### 2020.08.28

>  整理代码结构，抛弃借鉴的Bert模型，增加xlnet模型，预训练xlnet模型效果较差，可以在模型基础上再进行预训练，因此添加了模型预训练代码。

### 2020.07.15

>  修复bug，添加textGCN模型（单独训练，模型效果较差）。

### 2020.07.06

>  移除模型介绍＆部分模型实现，增加使用说明及运行环境。

### 2020.05.28

>  增加了直接使用学生模型训练代码，并使用公开测试集完成测试。

## 运行环境

python 3.7

transformers 4.11.3

torch 1.10.0

## 使用说明

下载[预训练BERT模型参数 pytorch_model.bin](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz)放入KnowledgeDistillation/bert_pretrain



运行 python distill.py

config.py中train_teacher、train_student分别表示是否训练教师模型、训练学生模型。

```python
  单独训练教师模型      
  self.train_teacher = 1
  self.train_student = 0
  教师模型与学生模型串行训练（蒸馏）
  self.train_teacher = 1
  self.train_student = 1
  想要单独训练学生模型，需配置student.py中损失函数a=1，T=0，
  且saved_dict中存在训练好的教师模型teacher.ckpt，否则需要对代码进行改动
  self.train_teacher = 0
  self.train_student = 1
```

想要单独训练学生模型，只需将student.py中损失函数的a=1，T=0即可。

## 模型实现

模型基本上是对论文[**Distilling Task-Specific Knowledge from BERT into Simple Neural Networks**](https://arxiv.org/abs/1903.12136
)的复现

### 代码结构

Teacher模型：BERT模型

Student模型：一层的biLSTM

LOSS函数：交叉熵 、MSE LOSS

知识函数：用最后一层的softmax前的logits作为知识表示

## 模型效果

内部数据集测试效果。

Teacher

Running time: 116.05915258956909 s

|            | precision | recall | F1-score | support |
| :--------: | :-------: | :----: | :------: | :-----: |
|     0      |   0.91    |  0.84  |   0.87   |  2168   |
|     1      |   0.82    |  0.90  |   0.86   |  1833   |
|  accuracy  |           |        |   0.86   |  4001   |
| macro avg  |   0.86    |  0.87  |   0.86   |  4001   |
| weight avg |   0.87    |  0.86  |   0.86   |  4001   |

Student

Running time: 0.155623197555542 s

|            | precision | recall | F1-score | support |
| :--------: | :-------: | :----: | :------: | :-----: |
|     0      |   0.87    |  0.85  |   0.86   |  2168   |
|     1      |   0.83    |  0.85  |   0.84   |  1833   |
|  accuracy  |           |        |   0.85   |  4001   |
| macro avg  |   0.85    |  0.85  |   0.85   |  4001   |
| weight avg |   0.85    |  0.85  |   0.85   |  4001   |

可以看出student模型与teacher模型相比精度有一定的丢失，这也可以理解，毕竟student模型结构简单。而在运行时间上大模型是小模型的746倍（cpu）。

## TNEWS测试效果

在数据集中选了5类并做了下采样。（此部分具体说明后续完善）

Student alone

|            | precision | recall | F1-score | support |
| :--------: | :-------: | :----: | :------: | :-----: |
|   story    |  0.6489   | 0.7907 |  0.7128  |   215   |
|   sports   |  0.7669   | 0.7849 |  0.7758  |   767   |
|   house    |  0.7350   | 0.7778 |  0.7558  |   378   |
|    car     |  0.8162   | 0.7522 |  0.7829  |   791   |
|    game    |  0.7319   | 0.7041 |  0.7177  |   659   |
|  accuracy  |           |        |  0.7562  |  2810   |
| macro avg  |  0.7398   | 0.7619 |  0.7490  |  2810   |
| weight avg |  0.7592   | 0.7562 |  0.7567  |  2810   |

Teacher

|            | precision | recall | F1-score | support |
| :--------: | :-------: | :----: | :------: | :-----: |
|   story    |  0.6159   | 0.8651 |  0.7195  |   215   |
|   sports   |  0.8423   | 0.7940 |  0.8174  |   767   |
|   house    |  0.8030   | 0.8519 |  0.8267  |   378   |
|    car     |  0.8823   | 0.7863 |  0.8316  |   791   |
|    game    |  0.7835   | 0.8073 |  0.7952  |   659   |
|  accuracy  |           |        |  0.8082  |  2810   |
| macro avg  |  0.7854   | 0.8209 |  0.7981  |  2810   |
| weight avg |  0.8172   | 0.8082 |  0.8100  |  2810   |

Student 

|            | precision | recall | F1-score | support |
| :--------: | :-------: | :----: | :------: | :-----: |
|   story    |  0.5207   | 0.8186 |  0.6365  |   215   |
|   sports   |  0.8411   | 0.7040 |  0.7665  |   767   |
|   house    |  0.7678   | 0.7698 |  0.7688  |   378   |
|    car     |  0.8104   | 0.7459 |  0.7768  |   791   |
|    game    |  0.6805   | 0.7466 |  0.7120  |   659   |
|  accuracy  |           |        |  0.7434  |  2810   |
| macro avg  |  0.7241   | 0.7570 |  0.7321  |  2810   |
| weight avg |  0.7604   | 0.7434 |  0.7470  |  2810   |

## 已知问题

1. ~~直接用student模型训练效果如何，未做测试。~~ （在公开数据集上完成测试，并上传了训练代码）
2. ~~学生模型用了句向量表征，~~原论文用的词向量，后续工作将换回。
3. ~~教师模型参考了别人的代码，后续会自己搭BERT~~。

## TODO

1. 学生模型加载预训练词向量（目前随机初始化）

## 参考链接

1. [如何理解soft target这一做法？  知乎 YJango的回答](https://www.zhihu.com/question/50519680?sort=created)

2. [【经典简读】知识蒸馏(Knowledge Distillation) 经典之作](https://zhuanlan.zhihu.com/p/102038521?utm_source=wechat_timeline)
3. [**Distilling the Knowledge in a Neural Network**](https://arxiv.org/abs/1503.02531  )
4. [**Distilling Task-Specific Knowledge from BERT into Simple Neural Networks**](https://arxiv.org/abs/1903.12136
   )
5. [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)