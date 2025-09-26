import numpy as np
from medimg_vision.segmentation import otsu_threshold
def test_otsu_simple():
 img=np.zeros((64,64),np.uint8); img[16:48,16:48]=200; th=otsu_threshold(img); assert th[32,32]==255 and th[2,2]==0
