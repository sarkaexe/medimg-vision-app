import numpy as np, cv2
def opening(img,k=3): ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)); return cv2.morphologyEx(img,cv2.MORPH_OPEN,ker)
def closing(img,k=3): ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)); return cv2.morphologyEx(img,cv2.MORPH_CLOSE,ker)
def connected_components(mask): return cv2.connectedComponentsWithStats(mask, connectivity=8)
