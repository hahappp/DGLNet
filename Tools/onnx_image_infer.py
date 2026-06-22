import numpy as np
import onnxruntime as ort
from PIL import Image


def preprocess_image(image_path):
    """手动进行图像预处理，完全脱离 PyTorch 和 torchvision"""
    # 1. 读取图片并转换为 RGB
    img = Image.open(image_path).convert('RGB')

    # 2. Resize 到 256
    img = img.resize((256, 256), Image.BICUBIC)

    # 3. CenterCrop 到 224
    width, height = img.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    img = img.crop((left, top, right, bottom))

    # 4. 转换为 Numpy 数组并 Normalize
    img_np = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std

    # 5. HWC 转为 CHW，并增加 Batch 维度: (1, 3, 224, 224)
    img_np = np.transpose(img_np, (2, 0, 1))
    img_tensor = np.expand_dims(img_np, axis=0)

    return img_tensor


def main():
    # ================= 配置路径 =================
    # 1. 刚才导出的 ONNX 模型路径
    onnx_model_path = "dglnet_rafdb.onnx"
    # 2. 随便找一张测试集里的图片路径
    test_image_path = r"D:\dataset\RAF-DB\val\happy\test_0275.jpg"
    # ============================================

    class_names = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    print(f"⏳ 正在加载 ONNX 推理引擎...")
    # 启动 ONNX 运行环境 (仅使用 CPU 进行极其轻量化的推理)
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    print(f"✅ 引擎加载完毕！\n")

    print(f"🖼️ 正在处理测试图片: {test_image_path}")
    input_tensor = preprocess_image(test_image_path)

    # 核心步骤：直接传入 numpy 数组进行推理，完全不需要 torch!
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]

    # 计算 Softmax 获取概率
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    # 获取最高概率的类别
    pred_idx = np.argmax(probs)
    pred_class = class_names[pred_idx]
    confidence = probs[pred_idx]

    print("-" * 40)
    print(f"🎯 推理结果: {pred_class}")
    print(f"📊 置信度:   {confidence * 100:.2f}%")
    print("-" * 40)


if __name__ == '__main__':
    main()