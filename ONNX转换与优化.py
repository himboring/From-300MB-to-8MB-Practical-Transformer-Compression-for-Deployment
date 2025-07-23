#三、ONNX转换与优化代码实战


#1、基础模型导出为ONNX
import torch
from transformers import BertModel

# 加载预训练模型（假设已量化）
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # 设置为推理模式

# 生成虚拟输入（示例：batch=1, seq_len=128）
dummy_input = torch.randint(0, 10000, (1, 128))  # 模拟token_id输入

# 导出ONNX模型
torch.onnx.export(
    model,
    dummy_input,
    "bert_model.onnx",
    input_names=["input_ids"],      # 输入节点名
    output_names=["last_hidden_state"],  # 输出节点名
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},  # 动态维度
        "last_hidden_state": {0: "batch_size"}
    },
    opset_version=13,  # 必须≥11以支持Transformer算子
    do_constant_folding=True  # 启用常量折叠优化
)


#2、ONNX模型优化（算子融合/常量折叠）
from onnxruntime.tools import optimize_model
from onnxruntime.quantization import quantize_dynamic, QuantType

# 加载原始ONNX模型并优化
onnx_model = optimize_model("bert_model.onnx")
onnx_model.save("bert_model_optimized.onnx")

# 可选：进一步INT8量化（需安装onnxruntime>=1.8）
quantize_dynamic(
    "bert_model_optimized.onnx",
    "bert_model_quantized.onnx",
    weight_type=QuantType.QInt8,  # 权重量化类型
    optimize_model=True           # 启用图优化
)


#3、跨平台部署验证


    #3.1、Python端推理（ONNX Runtime）
import onnxruntime as ort

# 创建推理会话（启用所有优化）
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4  # 设置并行线程数

# 加载量化模型
session = ort.InferenceSession(
    "bert_model_quantized.onnx",
    sess_options,
    providers=["CPUExecutionProvider"]  # 指定CPU/GPU
)

# 准备输入（需与导出时形状一致）
input_ids = dummy_input.numpy()
inputs = {"input_ids": input_ids}

# 运行推理
outputs = session.run(None, inputs)
print(outputs[0].shape)  # 输出形状：(1, 128, 768)


    #3.2、Android端部署（NNAPI加速）
    #这里为java代码，请用java编辑
// Android代码示例（需ONNX Runtime Mobile库）
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;

SessionOptions options = new SessionOptions();
options.setOptimizationLevel(SessionOptions.OptLevel.ORT_ENABLE_EXTENDED);  // 启用优化
options.addNnapi();  // 使用Android NPU加速（API Level 27+）

OrtSession session = new OrtSession(assetManager, "bert_model_quantized.onnx", options);

// 准备输入
OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputIds, new long[]{1, 128});
OrtSession.Result result = session.run(Collections.singletonMap("input_ids", inputTensor));

// 获取输出
float[][] lastHiddenState = (float[][]) result.get(0).getValue();


#4、高级技巧与问题排查


    #4.1、自定义算子支持
    #若遇到不支持的算子（如自定义Layer），需注册自定义实现：
# 定义符号化函数（示例：自定义GELU）
class CustomGelu(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input):
        return g.op("CustomGelu", input, attr_schema="alpha:float=1.0")

    @staticmethod
    def forward(ctx, input):
        return input * 0.5 * (1.0 + torch.erf(input / 1.41421))

# 导出时指定custom_opsets
torch.onnx.export(..., custom_opsets={"custom_domain": 1})


    #4.2动态轴与内存优化
    #限制动态轴范围避免内存爆炸：
dynamic_axes={
    "input_ids": {
        0: "batch_size(max=8)",  # 限制最大batch=8
        1: "sequence_length(max=512)"
    }
}


    #4.3、性能分析工具
    #使用ONNX Runtime的Profiler定位瓶颈：
session = ort.InferenceSession(..., enable_profiling=True)
session.run(...)
session.end_profiling()  # 生成profile.json