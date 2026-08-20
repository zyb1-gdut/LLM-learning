import torch.cuda
# 默认没有设置过cuda时,CUDA available结果是False,如果设置后是True
print("CUDA是否可用:", torch.cuda.is_available())
print("CUDA版本:", torch.version.cuda)
print("PyTorch版本:", torch.__version__)
print("电脑上GPU数量:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("当前使用的GPU版本:",
torch.cuda.get_device_name(torch.cuda.current_device()))