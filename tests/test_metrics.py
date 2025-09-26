import numpy as np
from medimg_vision.metrics import psnr, ssim
def test_psnr_identical():
 a=np.zeros((16,16),dtype=np.uint8); assert psnr(a,a)>90
def test_ssim_identical():
 a=(np.random.rand(32,32)*255).astype(np.uint8); assert ssim(a,a)>0.99
