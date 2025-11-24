# utils.py
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2

def pil_to_cv2(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def crop_image(pil_image, x, y, w, h):
    return pil_image.crop((x, y, x + w, y + h))

def detect_blur(pil_image):
    img_cv = pil_to_cv2(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var

def enhance_image(pil_image, strength=1.5):
    enhancer = ImageEnhance.Sharpness(pil_image)
    img_sharp = enhancer.enhance(strength)
    img_cv = pil_to_cv2(img_sharp)
    try:
        img_cv = cv2.detailEnhance(img_cv, sigma_s=10, sigma_r=0.15)
    except Exception:
        pass
    pil_out = cv2_to_pil(img_cv)
    return pil_out

def unsharp_mask(pil_image, radius=2, percent=150, threshold=3):
    return pil_image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
