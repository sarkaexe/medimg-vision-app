import numpy as np
from medimg_vision.preprocessing import normalize
def test_normalize_constant():
 a=(np.ones((8,8))*7).astype(np.uint8); out=normalize(a); assert out.max()==0 and out.min()==0
