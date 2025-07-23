#一、知识蒸馏代码实战


#1、环境准备与模型加载


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

# 加载预训练Teacher模型（BERT-base）
teacher_model = BertModel.from_pretrained("bert-base-uncased")
teacher_model.eval()  # 冻结Teacher参数

# 定义Student模型（更小架构）
student_config = BertConfig(
    num_hidden_layers=4,     # 减少层数（原12层）
    hidden_size=512,         # 缩小特征维度（原768）
    num_attention_heads=8    # 减少注意力头（原12头）
)
student_model = BertModel(student_config)


#2、蒸馏损失函数实现
def distillation_loss(
    student_logits,  # Student模型输出logits
    teacher_logits,  # Teacher模型输出logits
    labels,         # 真实标签
    temperature=5.0, # 温度参数（软化概率分布）
    alpha=0.3       # 蒸馏损失权重
):
    # 1. 计算原始任务损失（交叉熵）
    loss_ce = F.cross_entropy(student_logits, labels)
    
    # 2. 计算蒸馏损失（KL散度）
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    loss_kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
    
    # 3. 组合损失
    total_loss = alpha * loss_kl + (1 - alpha) * loss_ce
    return total_loss


#3、训练循环（关键步骤）
# 初始化优化器
optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)

for epoch in range(3):  # 示例训练3轮
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        
        # Teacher模型前向传播（不计算梯度）
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids, attention_mask)
            teacher_logits = teacher_outputs.last_hidden_state.mean(dim=1)  # 示例任务：池化后分类
        
        # Student模型前向传播
        student_outputs = student_model(input_ids, attention_mask)
        student_logits = student_outputs.last_hidden_state.mean(dim=1)
        
        # 计算蒸馏损失
        loss = distillation_loss(
            student_logits, 
            teacher_logits, 
            labels,
            temperature=5.0,
            alpha=0.3
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


#4、中间层特征蒸馏（增强版）
    #若需对齐中间层特征（如TinyBERT策略），可扩展损失函数：
def feature_distillation(student_hiddens, teacher_hiddens):
    # 对齐特定层的隐藏状态（如Student第1层对Teacher第3层）
    s_layer1, s_layer4 = student_hiddens[1], student_hiddens[3]
    t_layer3, t_layer6 = teacher_hiddens[2], teacher_hiddens[5]
    loss_mse = F.mse_loss(s_layer1, t_layer3) + F.mse_loss(s_layer4, t_layer6)
    return loss_mse

# 在训练循环中追加损失
student_all_hiddens = student_model(input_ids, output_hidden_states=True).hidden_states
teacher_all_hiddens = teacher_model(input_ids, output_hidden_states=True).hidden_states
loss += 0.2 * feature_distillation(student_all_hiddens, teacher_all_hiddens)  # 加权融合


#关键点解析

    #温度参数T的作用：

        #T>1时软化概率分布，让小模型学习Teacher的类别间关系（如"猫→狗"相似性）。

        #典型值：T=3~10（文本任务常用5.0）。

    #损失权重alpha的平衡：

        #alpha=0：仅用真实标签训练（普通训练）。

        #alpha=0.3：30%依赖Teacher知识（常用初始值）。

    #中间层蒸馏的意义：

        #强制Student模仿Teacher的中间表示，提升小模型的特征提取能力
