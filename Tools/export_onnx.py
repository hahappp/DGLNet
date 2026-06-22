import torch
import os
# 导入你的 DGLNet 模型
from model import convnext_tiny_ultimate


def main():
    # 1. 设置路径 (请根据你的实际情况修改)
    weights_path = r"填写你的实际路径"
    output_onnx_path = r"dglnet_rafdb.onnx"

    # 2. 实例化模型并加载权重
    print("⏳ 正在加载 PyTorch 模型...")
    model = convnext_tiny_ultimate(num_classes=7)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()  # 必须设置为评估模式！

    # 3. 创建一个 Dummy Input (假输入)，告诉 ONNX 你的输入张量长什么样
    # batch_size=1, channels=3, height=224, width=224
    dummy_input = torch.randn(1, 3, 224, 224, device='cpu')

    # 4. 执行“另存为”操作 (导出为 ONNX)
    print("🚀 正在导出为 ONNX 格式...")
    torch.onnx.export(
        model,  # 要导出的模型
        dummy_input,  # 假输入
        output_onnx_path,  # 输出路径
        export_params=True,  # 将权重参数也存入
        opset_version=11,  # ONNX 算子集版本，11是一个非常稳定兼容的版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input'],  # 为输入节点命名
        output_names=['output'],  # 为输出节点命名
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 允许动态 Batch Size
    )

    print(f"✅ 导出成功！ONNX 模型已保存至: {output_onnx_path}")
    print("你现在可以脱离 PyTorch，在任何支持 ONNX 的设备上运行它了！")


if __name__ == '__main__':
    main()
