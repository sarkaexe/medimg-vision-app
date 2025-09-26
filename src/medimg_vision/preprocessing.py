import numpy as np, cv2
def to_gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
def normalize(img):
    img=img.astype(np.float32); m,M=img.min(),img.max()
    return (np.zeros_like(img,dtype=np.uint8) if M-m<1e-8 else ((img-m)/(M-m)*255).astype(np.uint8))
def median_denoise(img, ksize=3): return cv2.medianBlur(img, max(3,int(ksize)|1))
def bilateral_denoise(img,d=7,sigma_color=50,sigma_space=50): return cv2.bilateralFilter(img,d,sigma_color,sigma_space)
def clahe(img,clip_limit=2.0,tile_grid_size=8):
    g=to_gray(img); c=cv2.createCLAHE(clipLimit=clip_limit,tileGridSize=(tile_grid_size,tile_grid_size))
    e=c.apply(g); return cv2.cvtColor(e, cv2.COLOR_GRAY2BGR) if img.ndim==3 else e
