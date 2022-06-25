- [1. BERT最后一层mean-pooling](#1-bert最后一层mean-pooling)
- [2. RDrop策略](#2-rdrop策略)
- [3. 最后四层拼接](#3-最后四层拼接)
- [4. 模型层之间的差分学习率](#4-模型层之间的差分学习率)
- [5. xlnet分类处理长文本](#5-xlnet分类处理长文本)
- [6. EMA指数权重平均](#6-ema指数权重平均)
- [7. DiceLoss](#7-diceloss)
- [8. FocalLoss](#8-focalloss)
- [9. FGM对抗训练](#9-fgm对抗训练)
- [10. 生成模型中的各种解码方式](#10-生成模型中的各种解码方式)
- [11. mseloss中引入相关性损失](#11-mseloss中引入相关性损失)


# 1. BERT最后一层mean-pooling
```python
class LastHiddenModel(nn.Module):
    def __init__(self, model_name, n_classes):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name)

        self.model = AutoModel.from_pretrained(model_name, config=config)
        self.linear = nn.Linear(config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(input_ids, attention_mask, token_type_ids)# last_hidden_state和pooler out
        last_hidden_state = outputs[0] # 所有字符最后一层hidden state # 32 400 768 ，但是PAD PAD
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        logits = self.linear(mean_embeddings)
        return logits
```
# 2. RDrop策略
```python
import torch.nn.functional as F

# define your task model, which outputs the classifier logits
model = TaskModel()

def compute_kl_loss(self, p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

# keep dropout and forward twice
logits = model(x)

logits2 = model(x)

# cross entropy loss for classifier
ce_loss = 0.5 * (cross_entropy_loss(logits, label) + cross_entropy_loss(logits2, label))

kl_loss = compute_kl_loss(logits, logits2)

# carefully choose hyper-parameters
loss = ce_loss + α * kl_loss
```
# 3. 最后四层拼接
```python
class LastFourModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        config = AutoConfig.from_pretrained(PRE_TRAINED_MODEL_NAME)
        config.update({'output_hidden_states':True})
        self.model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME, config=config)
        self.linear = nn.Linear(4*768, n_classes)
        
    def forward(self, input_ids, attention_mask):
        
        outputs = self.model(input_ids, attention_mask)
        all_hidden_states = torch.stack(outputs[2])
        concatenate_pooling = torch.cat(
            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]), -1
        )
        concatenate_pooling = concatenate_pooling[:,0]
        output = self.linear(concatenate_pooling)
        
        return soutput
```
# 4. 模型层之间的差分学习率
```python
def get_parameters(model, model_init_lr, multiplier, classifier_lr):
    parameters = []
    lr = model_init_lr
    for layer in range(12,-1,-1):
        layer_params = {
            'params': [p for n,p in model.named_parameters() if f'encoder.layer.{layer}.' in n],
            'lr': lr
        }
        parameters.append(layer_params)
        lr *= multiplier
    classifier_params = {
        'params': [p for n,p in model.named_parameters() if 'layer_norm' in n or 'linear' in n 
                   or 'pooling' in n],
        'lr': classifier_lr
    }
    parameters.append(classifier_params)
    return parameters
parameters=get_parameters(model,2e-5,0.95, 1e-4)
optimizer=AdamW(parameters)
```
# 5. xlnet分类处理长文本
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import XLNetModel


class MyXLNet(nn.Module):
    def __init__(self, num_classes=35, alpha=0.5):
        self.alpha = alpha
        super(MyXLNet, self).__init__()
        self.net = XLNetModel.from_pretrained(xlnet_cfg.xlnet_path).cuda()
        for name, param in self.net.named_parameters():
            if 'layer.11' in name or 'layer.10' in name or 'layer.9' in name or 'layer.8' in name or 'pooler.dense' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.MLP = nn.Sequential(
            nn.Linear(768, num_classes, bias=True),
        ).cuda()

    def forward(self, x):
        x = x.long()
        x = self.net(x, output_all_encoded_layers=False).last_hidden_state
        x = F.dropout(x, self.alpha, training=self.training)
        x = torch.max(x, dim=1)[0]
        x = self.MLP(x)
        return torch.sigmoid(x)
```
# 6. EMA指数权重平均
```python
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# 初始化
ema = EMA(model, 0.999)
ema.register()

# 训练过程中，更新完参数后，同步update shadow weights
def train():
    optimizer.step()
    ema.update()

# eval前，apply shadow weights；eval之后，恢复原来模型的参数
def evaluate():
    ema.apply_shadow()
    # evaluate
    ema.restore()

# 这里强调一下  在保存模型的之前 需要ema.apply_shadow()一下，即把ema后的权重更新到模型上，然后再保存。
# 另外: 模型权重的指数滑动平均, 不参加梯度更新，只是记录滑动平均的参数，给预测使用，注意区别于类似adam一类的自适应学习率优化器, 针对一阶二阶梯度的指数滑动平均, 两者完全不同
```
# 7. DiceLoss
```python
# ref: https://zhuanlan.zhihu.com/p/68748778
import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        # 上面计算的dice为dice系数  1-dice系数  就是diceloss
        return 1 - dice


if __name__ == '__main__':
    target = torch.tensor([[0], [1], [2], [1], [0]])   # 三分类

    # 这里必须将标签转为one-hot形式。
    batch_size, label_num = 5, 3
    target_onthot = torch.zeros(batch_size, label_num).scatter_(1, target, 1)

    logits = torch.rand(5, 3)
    loss_func = DiceLoss()
    loss = loss_func(logits, target_onthot)
    print(loss)
```
# 8. FocalLoss
```python
import torch
from torch import nn
import torch.nn.functional as F


class BCEFocalLoss(nn.Module):
    # 可用于二分类和多标签分类
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, labels):
        '''
        假设是三个标签的多分类
        loss_fct = BCEFocalLoss()
        labels = torch.tensor([[0, 1, 1], [1, 0, 1]])
        logits = torch.tensor([[0.3992, 0.2232, 0.6435],[0.3800, 0.3044, 0.3241]])
        loss = loss_fct(logits, labels)
        print(loss)  # tensor(0.0908)
   
        '''
        probs = torch.sigmoid(logits)
         
        loss = -self.alpha * (1 - probs) ** self.gamma * labels * torch.log(probs) - (1 - self.alpha) * probs ** self.gamma * (1 - labels) * torch.log(1 - probs)
       
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class MultiCEFocalLoss(nn.Module):
    # 可以用于多分类 (注: 不是多标签分类)
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num = class_num
   
    def forward(self, logits, labels):
        '''
        logits: (batch_size, class_num)
        labels: (batch_size,)
        '''
        probs = F.softmax(logits, dim=1) 
        class_mask = F.one_hot(labels, self.class_num)   # 将真实标签转为one-hot
        ids = labels.view(-1, 1)   # (batch_size, 1)
        alpha = self.alpha[ids.data.view(-1)]   # 每一类的权重因子
        
        probs = (probs * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
         
        loss = -alpha * (torch.pow((1-probs), self.gamma)) * log_p
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


if __name__ == '__main__':
    # loss_fct = BCEFocalLoss()
    # labels = torch.tensor([[0, 1, 1], [1, 0, 1]])
    # logits = torch.tensor([[0.3992, 0.2232, 0.6435],[0.3800, 0.3044, 0.3241]])
    # loss = loss_fct(logits, labels)
    # print(loss)
    
    # 举例四分类
    loss_fct = MultiCEFocalLoss(class_num=4)
    labels = torch.tensor([1, 3, 0, 0, 2])
    logits = torch.randn(5, 4)
    loss = loss_fct(logits, labels)
    print(loss)

```
# 9. FGM对抗训练
```python
import torch
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# 初始化
fgm = FGM(model)
for batch_input, batch_label in data:
    # 正常训练
    loss = model(batch_input, batch_label)
    loss.backward() # 反向传播，得到正常的grad
    # 对抗训练
    fgm.attack() # 在embedding上添加对抗扰动
    loss_adv = model(batch_input, batch_label)
    loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    fgm.restore() # 恢复embedding参数
    # 梯度下降，更新参数
    optimizer.step()
    model.zero_grad()
```

# 10. 生成模型中的各种解码方式
```python
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from transformers.models.gpt2 import GPT2LMHeadModel


def greedy_decode():
    input_txt = "最美不过下雨天"
    input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"]
    # print(input_ids)   # tensor([[ 101, 3297, 5401,  679, 6814,  678, 7433, 1921,  102]])
    # print(input_ids.size())   # torch.Size([1, 9])
    iterations = []
    n_steps = 8  # 进行8步解码
    choices_per_step = 5  # 每一步候选数量

    with torch.no_grad():
        for i in range(n_steps):
            iteration = dict()
            iteration['Input'] = tokenizer.decode(input_ids[0])
            output = model(input_ids=input_ids)
            # print(output.logits.size())  # torch.Size([1, 9, 21128])

            # 取最后一个token的输出
            next_token_logits = output.logits[0, -1, :]

            # print(next_token_logis.size())   # torch.Size([1, 21128])
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
            # print(sorted_ids)   # tensor([ 5682,   741,   691,  ..., 12518, 11888, 10980])

            for choice_idx in range(choices_per_step):
                # 选取概率最大的五个
                token_id = sorted_ids[choice_idx]
                token_prob = next_token_probs[token_id].cpu().numpy()
                token_choice = (
                    # 得到对应的字符 + 概率值
                    f"{tokenizer.decode(token_id)} ({100 * token_prob:.2f}%)"  # 取百分号两位数
                )
                iteration[f"Choice {choice_idx + 1}"] = token_choice
            input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)  # 将概率最大的字符拼接到提示文本
            iterations.append(iteration)
        print(iterations)


def default_decode():
    # greedy search
    max_length = 50
    input_txt = "最美不过下雨天"
    input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"]
    output_greedy = model.generate(input_ids, max_length=max_length, do_sample=False)
    print(output_greedy)
    print(tokenizer.decode(output_greedy[0]))
    # [CLS] 最 美 不 过 下 雨 天 [SEP] 色 彩 斑 斓 的 雨 天 ， 是 不 是 很 美 ？ 这 个 秋 天 ，
    # 是 不 是 很 美 ？ 这 个 秋 天 ， 是 不 是 很 美 ？ 这 个 秋 天 ，


def beam_search_decode():
    def log_probs_from_logits(logits, labels):
        logp = F.log_softmax(logits, dim=-1)
        logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
        return logp_label

    def sequence_logprob(model, labels, input_len=0):
        with torch.no_grad():
            output = model(labels)
            log_probs = log_probs_from_logits(
                output.logits[:, :-1, :], labels[:, 1:])
            seq_log_prob = torch.sum(log_probs[:, input_len:])
        return seq_log_prob.cpu().numpy()

    # 贪婪搜索
    max_length = 50
    input_txt = "最美不过下雨天"
    input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"]
    output_greedy = model.generate(input_ids, max_length=max_length, do_sample=False)
    print(output_greedy)
    logp = sequence_logprob(model, output_greedy, input_len=len(input_ids[0]))
    print(tokenizer.decode(output_greedy[0]))
    print(f"\nlog-prob: {logp:.2f}")

    # beam_search搜索
    output_beam = model.generate(input_ids, max_length=max_length, num_beams=3,
                                 do_sample=False, no_repeat_ngram_size=2)  # no_repeat_ngram_size缓解重复
    print(output_beam)
    logp = sequence_logprob(model, output_beam, input_len=len(input_ids[0]))
    print(tokenizer.decode(output_beam[0]))
    print(f"\nlog-prob: {logp:.2f}")


def temperature_sampling_decode():
    torch.manual_seed(42)
    max_length = 50
    input_txt = "最美不过下雨天"
    input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"]
    output_temp = model.generate(input_ids, max_length=max_length, do_sample=True,
                                 temperature=0.5, top_k=10)
    print(output_temp)
    print(tokenizer.decode(output_temp[0]))


if __name__ == '__main__':
    tokenizer = BertTokenizer(vocab_file='./gpt2_pretrain/vocab.txt')
    model = GPT2LMHeadModel.from_pretrained('./gpt2_pretrain')
    # default_decode()    # 默认的贪婪搜索
    # greedy_decode()   # 手写greedy search
    # beam_search_decode()   # 贪婪搜索和beam_search对比
    temperature_sampling_decode()
```
# 11. mseloss中引入相关性损失
```python
from torch import nn
import torch


class MSECorrLoss(nn.Module):
    def __init__(self, p=1.5):
        super(MSECorrLoss, self).__init__()
        self.p = p
        self.mseLoss = nn.MSELoss()

    def forward(self, logits, target):
        assert (logits.size() == target.size())
        mse_loss = self.mseLoss(logits, target)   # 均方误差损失

        logits_mean = logits.mean(dim=0)
        logits_std = logits.std(dim=0)
        logits_z = (logits - logits_mean) / logits_std   # 正态分布标准化

        target_mean = target.mean(dim=0)
        target_std = target.std(dim=0)
        target_z = (target - target_mean) / target_std   # 正态分布标准化
        corr_loss = 1 - ((logits_z * target_z).mean(dim=0))   # 后面的减数 就是计算两个分布的相关系数，然后用1减。越相关损失越小。
        loss = mse_loss + self.p * corr_loss
        return loss


if __name__ == '__main__':
    logits = torch.tensor([[0.1], [0.2], [0.2], [0.8], [0.4]], dtype=torch.float32)
    label = torch.tensor([[0], [1], [0], [0], [1]], dtype=torch.float32)
    print(logits.size())   # torch.Size([5, 1])   (batch_size, 1)
    print(label.size())  # torch.Size([5, 1])   (batch_size, 1)
    loss_func = MSECorrLoss()
    loss = loss_func(logits, label)
    print(loss)   # tensor([1.9949])

```
