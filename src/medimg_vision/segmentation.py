import numpy as np, cv2
from .preprocessing import to_gray
def otsu_threshold(img):
    g=to_gray(img); _,th=cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU); return th
def adaptive_threshold(img, block_size=31, C=5):
    g=to_gray(img); bs=max(3, block_size|1)
    return cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, bs, C)
def watershed_segment(img):
    g=to_gray(img); blur=cv2.medianBlur(g,3)
    _,th=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    import numpy as np
    k=np.ones((3,3),np.uint8); bg=cv2.dilate(th,k,3)
    dist=cv2.distanceTransform(th,cv2.DIST_L2,5); _,fg=cv2.threshold(dist,0.5*dist.max(),255,0); fg=fg.astype(np.uint8)
    unk=cv2.subtract(bg,fg); col=img if img.ndim==3 else cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    _,markers=cv2.connectedComponents(fg); markers=markers+1; markers[unk==255]=0
    markers=cv2.watershed(col,markers); return ((markers>1).astype(np.uint8)*255)
