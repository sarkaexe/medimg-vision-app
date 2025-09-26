from pathlib import Path
import numpy as np, cv2, random
def generate_circle(h,w):
    img=np.zeros((h,w),np.uint8)
    r=random.randint(min(h,w)//10, min(h,w)//4)
    cx=random.randint(r+10, w-r-10); cy=random.randint(r+10, h-r-10)
    cv2.circle(img,(cx,cy),r,255,-1)
    img=cv2.GaussianBlur(img,(0,0),1.0)
    noise=np.random.normal(0,15,img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16)+noise, 0, 255).astype(np.uint8)
def generate_dataset(out_dir, n=10, h=256, w=256):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(out/f"synthetic_{i:03d}.png"), generate_circle(h,w))
    print(f"Saved {n} synthetic images to {out_dir}")
