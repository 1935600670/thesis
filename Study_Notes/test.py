import torch

# 查看 PyTorch 版本
print("PyTorch 版本：", torch.__version__)
# 检查 CUDA 是否可用
print("CUDA 是否可用：", torch.cuda.is_available())
# 查看 CUDA 版本（如果可用）
if torch.cuda.is_available():
    print("CUDA 版本：", torch.version.cuda)
