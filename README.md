# 知识蒸馏在文本方向上的应用

虽然说做文本不像图像对gpu依赖这么高，但是当需要训练一个大模型或者拿这个模型做预测的时候，也是耗费相当多资源的，尤其是BERT出来以后，不管做什么用BERT效果都能提高，万物皆可BERT。

然而想要在线上部署应用，大公司倒还可以烧钱玩，毕竟有钱任性，小公司可玩不起，成本可能都远大于效益。这时候，模型压缩的重要性就体现出来了，如果一个小模型能够替代大模型，而这个小模型的效果又和大模型差不多，换做你晚上做梦也会笑醒。

## 目录

- [知识蒸馏介绍](#知识蒸馏介绍)
  - [模型结构](#模型结构)
- [模型实现](#模型实现)
  - [代码结构](#代码结构)
  - [学生模型输入](#学生模型输入)
  - [学生模型结构](#学生模型结构)
  - [教师模型结构](#教师模型结构)
  - [损失函数](#损失函数)
- [模型效果](#模型效果)
- [已知问题](#已知问题)
- [参考链接](#参考链接)

## 知识蒸馏介绍

在讲知识蒸馏时一定会提到的Geoffrey Hinton开山之作[**Distilling the Knowledge in a Neural Network**](https://arxiv.org/abs/1503.02531  )当然也是在图像中开的山，下面简单做一个介绍。

知识蒸馏使用的是Teacher—Student模型，其中teacher是“知识”的输出者，student是“知识”的接受者。知识蒸馏的过程分为2个阶段:

1.原始模型训练: 训练"Teacher模型", 它的特点是模型相对复杂，也可以由多个分别训练的模型集成而成。

2.精简模型训练: 训练"Student模型", 它是参数量较小、模型结构相对简单的单模型。

### 模型结构

![蒸馏结构图](https://pic3.zhimg.com/v2-271064c09d53934a346cff1fafcc466b_r.jpg#pic_center)

​		借用YJango大佬的图，这里我简单解释一下我们怎么构建这个模型

1. 训练大模型

   首先我们先对大模型进行训练，得到训练参数保存，这一步在上图中并未体现，上图最左部分是使用第一步训练大模型得到的参数。

2. 计算大模型输出

   训练完大模型之后，我们将计算soft target，不直接计算output的softmax，这一步进行了一个divided by T蒸馏操作。（注：这时候的输入数据可以与训练大模型时的输入不一致，但需要保证与训练小模型时的输入一致）

3. 训练小模型

   小模型的训练包含两部分，并通过调节λ的大小来调整两部分损失函数的权重。

   - soft target loss
   - hard target loss

4. 小模型预测

   预测就没什么不同了，按常规方式进行预测。

## 模型实现

模型基本上是对论文[**Distilling Task-Specific Knowledge from BERT into Simple Neural Networks**](https://arxiv.org/abs/1903.12136
)的复现，完整项目代码在我的[GitHub仓库](https://github.com/HoyTta0/KnowledgeDistillation)下，下面介绍部分代码实现

### 代码结构

Teacher模型：BERT模型

Student模型：一层的biLSTM

LOSS函数：交叉熵 、MSE LOSS

知识函数：用最后一层的softmax前的logits作为知识表示

### 学生模型输入

Student模型的输入句向量由句中每一个词向量求和取平均得到，[词向量](https://github.com/Embedding/Chinese-Word-Vectors)为预训练好的300维中文向量，训练数据集为Wikipedia_zh中文维基百科。

```python
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('sgns.wiki.word')
# 生成句向量
def build_sentence_vector(sentence,w2v_model):

    sen_vec = [0]*300
    count = 0
    for word in sentence:
        try:
            sen_vec += w2v_model[word]
            count += 1
        except KeyError:
            continue
    if count != 0:
        sen_vec /= count
    return sen_vec
```

### 学生模型结构

学生模型为单层biLSTM，再接一层全连接。

```python
class biLSTM(nn.Module):
    def __init__(self):
        super(biLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=300, hidden_size=256,
                         num_layers=1, batch_first=True, dropout=0, bidirectional= True)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)     
        out = self.fc1(lstm_out)
        activated_t = F.relu(out)
        linear_out = self.fc2(activated_t)

        return linear_out, hidden
```

### 教师模型结构

教师模型为BERT，并对最后四层进行微调，后面也接了一层全连接。

```python
class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in list(self.bert.parameters())[:-4]:
            param.requires_grad = False
        self.fc = nn.Linear(config.hidden_size, 192)
        # self.fc1 = nn.Linear(192, 48)
        self.fc2 = nn.Linear(192, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers= False)
        out = self.fc(pooled)
        out = F.relu(out)
        # out = self.fc1(out)
        out = self.fc2(out)
        return out
```

### 损失函数

损失函数为学生输出s_logits和教师输出t_logits的MSE损失与学生输出与真实标签的交叉熵。

```python
# 损失函数
def get_loss(t_logits, s_logits, label, a, T):
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.MSELoss()
    loss = a * loss1(s_logits, label) + (1 - a) * loss2(t_logits, s_logits)
    return loss
```

## 模型效果

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

​		可以看出student模型与teacher模型相比精度有一定的丢失，这也可以理解，毕竟student模型结构简单。而在运行时间上大模型是小模型的746倍（cpu）。

## 已知问题

1. 没有写蒸馏过程，就是divided by T是如何实现蒸馏（其实是懒）
2. 直接用student小模型训练数据的效果如何，并未做测试。
3. 数据集是公司项目的数据集，量并不是很大，自己也只标注了几千条数据，后续会在CLUE的TNEWS短文本分类数据集上做测试，再出一个对比结果。

## 参考链接

1. [如何理解soft target这一做法？  知乎 YJango的回答](https://www.zhihu.com/question/50519680?sort=created)

2. [【经典简读】知识蒸馏(Knowledge Distillation) 经典之作](https://zhuanlan.zhihu.com/p/102038521?utm_source=wechat_timeline)
3. [**Distilling the Knowledge in a Neural Network**](https://arxiv.org/abs/1503.02531  )
4. [**Distilling Task-Specific Knowledge from BERT into Simple Neural Networks**](https://arxiv.org/abs/1903.12136
   )
5. [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)