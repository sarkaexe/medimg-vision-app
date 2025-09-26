import numpy as np, cv2
def psnr(a,b):
    a=a.astype(np.float32); b=b.astype(np.float32)
    mse=( (a-b)**2 ).mean()
    import numpy as np
    return 100.0 if mse==0 else 20*np.log10(255.0/np.sqrt(mse))
def ssim(a,b,K1=0.01,K2=0.03, win_size=11):
    if a.ndim==3: a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
    if b.ndim==3: b=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    a=a.astype(np.float32); b=b.astype(np.float32)
    C1=(K1*255)**2; C2=(K2*255)**2
    g=cv2.getGaussianKernel(win_size,1.5); w=g@g.T
    mu1=cv2.filter2D(a,-1,w); mu2=cv2.filter2D(b,-1,w)
    mu1_sq=mu1*mu1; mu2_sq=mu2*mu2; mu1_mu2=mu1*mu2
    sig1=cv2.filter2D(a*a,-1,w)-mu1_sq; sig2=cv2.filter2D(b*b,-1,w)-mu2_sq; sig12=cv2.filter2D(a*b,-1,w)-mu1_mu2
    s=((2*mu1_mu2+C1)*(2*sig12+C2))/((mu1_sq+mu2_sq+C1)*(sig1+sig2+C2)+1e-8)
    return float(s.mean())
