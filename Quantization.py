#二、模型量化代码实战


#1、动态量化（快速上手）


import torch
import os
from torch.quantization import quantize_dynamic

# 加载预训练FP32模型
model_fp32 = torch.load('bert_fp32.pth')  # 假设是已蒸馏的小模型
model_fp32.eval()

# 动态量化（仅量化权重，推理时动态量化激活值）
model_int8 = quantize_dynamic(
    model_fp32,
    {torch.nn.Linear, torch.nn.MultiheadAttention},  # 量化这些模块
    dtype=torch.qint8
)

# 保存量化模型
torch.save(model_int8.state_dict(), 'bert_dynamic_int8.pth')
print(f"模型体积对比: {os.path.getsize('bert_fp32.pth')/1e6:.1f}MB → "
      f"{os.path.getsize('bert_dynamic_int8.pth')/1e6:.1f}MB")

#关键点：

    #仅量化Linear和Attention层（对Transformer最有效）。

    #体积减少4倍（FP32的50MB → INT8的12.5MB）。

    #无需校准数据，但速度提升有限（适合快速实验）。


#2、静态量化（更高精度）：


    #2.1、插入量化观察器
from torch.quantization import prepare, convert

# 定义量化配置（x86 CPU适用）
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# 插入观察节点（统计激活值分布）
model_prepared = torch.quantization.prepare(model_fp32)

# 用校准数据运行前向传播（500-1000条样本）
with torch.no_grad():
    for data in calibration_dataset:
        model_prepared(data)  # 记录激活值范围


    #2.2转换为静态量化模型
# 转换为最终INT8模型（权重+激活值全量化）
model_int8_static = torch.quantization.convert(model_prepared)

# 验证量化效果
input_sample = torch.randn(1, 128)  # 示例输入
output_fp32 = model_fp32(input_sample)
output_int8 = model_int8_static(input_sample)
print(f"输出误差：{torch.mean(torch.abs(output_fp32 - output_int8)):.4f}")

#关键点：

    #需校准数据（覆盖真实输入分布）。

    #误差更小（相比动态量化）。

    #体积减少4倍，速度提升2-3倍。


#3、量化感知训练（QAT，最高精度）
from torch.ao.quantization import QuantStub, DeQuantStub

class QuantizableBERT(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.quant = QuantStub()    # 量化入口
        self.model = original_model
        self.dequant = DeQuantStub()  # 反量化出口

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        return self.dequant(x)

# 包装原始模型
qat_model = QuantizableBERT(model_fp32)
qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

# 训练阶段模拟量化
qat_model_prepared = torch.quantization.prepare_qat(qat_model)
for epoch in range(3):
    for data, label in train_loader:
        optimizer.zero_grad()
        output = qat_model_prepared(data)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()

# 转换为最终量化模型
model_int8_qat = torch.quantization.convert(qat_model_prepared)

#关键点：

    #训练时模拟量化噪声，提升最终量化精度。

    #适合对精度要求严苛的场景（精度损失<0.5%）。

    #流程复杂，需额外训练时间。


#4、量化模型部署验证


    #4.1、量化模型部署验证


import time

def benchmark(model, input_tensor, num_runs=100):
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    return (time.time() - start) / num_runs * 1000  # 毫秒/次

latency_fp32 = benchmark(model_fp32, input_sample)
latency_int8 = benchmark(model_int8_static, input_sample)
print(f"延迟对比：FP32={latency_fp32:.1f}ms, INT8={latency_int8:.1f}ms")


    #4.2、精度验证
# 在测试集上评估
correct = 0
total = 0
for data, label in test_loader:
    output = model_int8_static(data)
    pred = output.argmax(dim=1)
    correct += (pred == label).sum().item()
    total += len(label)
print(f"量化模型准确率：{correct/total*100:.2f}% (原模型：92.1%)")

# 一键选择最佳量化方式
def auto_quantize(model, data_loader=None):
    if data_loader:  # 有校准数据 → 静态量化
        model_prepared = prepare(model)
        for data in data_loader: 
            model_prepared(data)
        return convert(model_prepared)
    else:            # 无数据 → 动态量化
        return quantize_dynamic(model, {torch.nn.Linear})