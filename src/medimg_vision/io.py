from pathlib import Path
from typing import List
import numpy as np, cv2
try:
    import pydicom; _HAS=True
except Exception:
    _HAS=False
EXT={'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}
def imread(path:str):
    p=Path(path)
    if p.suffix.lower() in EXT:
        img=cv2.imdecode(np.fromfile(str(p),dtype=np.uint8),cv2.IMREAD_UNCHANGED)
        if img is None: raise ValueError('Failed to read')
        return img
    elif p.suffix.lower()=='.dcm' and _HAS:
        ds=pydicom.dcmread(str(p)); a=ds.pixel_array.astype(np.float32)
        a=(a-a.min())/(a.ptp()+1e-8)*255; return a.astype(np.uint8)
    else:
        raise ValueError('Unsupported extension')
def imwrite(path,img):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    ext=p.suffix.lower().lstrip('.') or 'png'
    ok,buf=cv2.imencode('.'+ext, img); assert ok, 'encode failed'
    buf.tofile(str(p))
def list_images(folder:str)->List[str]:
    p=Path(folder)
    return sorted([str(f) for f in p.rglob('*') if f.suffix.lower() in EXT or f.suffix.lower()=='.dcm'])
