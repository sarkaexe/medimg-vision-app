import numpy as np, cv2
from .preprocessing import to_gray
def sobel_edges(img, ksize=3):
    g=to_gray(img).astype(np.float32)
    sx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=ksize); sy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=ksize)
    m=cv2.magnitude(sx,sy); m=cv2.normalize(m,None,0,255,cv2.NORM_MINMAX); return m.astype(np.uint8)
def canny_edges(img, low=50, high=150): return cv2.Canny(to_gray(img), low, high)
