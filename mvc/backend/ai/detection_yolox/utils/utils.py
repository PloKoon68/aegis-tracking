import numpy as np
import cv2
from PIL import Image

#---------------------------------------------------------#
# OpenCV (NumPy) veya PIL Image girişini destekler
#---------------------------------------------------------#
def cvtColor(image):
    # Eğer NumPy array ise (OpenCV BGR veya RGB)
    if isinstance(image, np.ndarray):
        # 3 kanal varsa BGR -> RGB çevir
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # NumPy array'i PIL Image'e çevir
        image = Image.fromarray(image)
        return image
    # Eğer zaten PIL Image ise
    elif isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    else:
        raise TypeError(f"Desteklenmeyen görüntü tipi: {type(image)}")

#---------------------------------------------------#
# NumPy veya PIL Image girişini destekler
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    # NumPy array geldiyse önce PIL'e çevir
    if isinstance(image, np.ndarray):
        image = cvtColor(image)
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Desteklenmeyen görüntü tipi: {type(image)}")

    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def preprocess_input(image):
    image = image / 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image

#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
