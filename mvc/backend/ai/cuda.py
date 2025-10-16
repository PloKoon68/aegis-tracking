import torch
print(torch.cuda.is_available())  # True olmalı
print(torch.cuda.get_device_name(0))  # GPU adını yazmalı
