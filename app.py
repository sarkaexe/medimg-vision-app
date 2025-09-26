import sys, io, zipfile, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.append(str(ROOT/'src')); sys.path.append(str(ROOT))
import streamlit as st, numpy as np, cv2
from medimg_vision import io as mio, preprocessing as pre, segmentation as seg, edges as ed, morphology as morph
from medimg_vision.metrics import psnr, ssim
from examples.generate_synthetic_data import generate_dataset

st.set_page_config(page_title='medimg-vision demo', layout='wide')
st.title('🩺 medimg-vision — demo przetwarzania obrazów')

st.sidebar.header('Ustawienia')
with st.sidebar.expander('📤 Wejście'):
    upload = st.file_uploader('Wgraj obrazy', type=['png','jpg','jpeg','tif','tiff','bmp','dcm'], accept_multiple_files=True)
    gen_n = st.number_input('Liczba obrazów syntetycznych', 1, 200, 8)
    gen_btn = st.button('Wygeneruj obrazy syntetyczne')

with st.sidebar.expander('🧹 Preprocessing'):
    use_median = st.checkbox('Median denoise', True); median_k = st.slider('Kernel (median)', 3, 11, 3, 2)
    use_bilateral = st.checkbox('Bilateral denoise', False); bilateral_d = st.slider('d (bilateral)', 3, 15, 7, 2)
    bilateral_sc = st.slider('sigmaColor', 10, 150, 50, 10); bilateral_ss = st.slider('sigmaSpace', 10, 150, 50, 10)
    use_clahe = st.checkbox('CLAHE', True); clahe_clip = st.slider('clipLimit', 1.0, 4.0, 2.0, 0.1); clahe_grid = st.slider('tileGrid', 4, 16, 8, 2)

with st.sidebar.expander('✂️ Segmentacja'):
    seg_method = st.selectbox('Metoda', ['otsu','adaptive','watershed'], 0)
    block = st.slider('block_size (adaptive)', 3, 63, 31, 2)
    C = st.slider('C (adaptive)', -10, 10, 5, 1)

with st.sidebar.expander('📈 Krawędzie'):
    edge_method = st.selectbox('Metoda', ['canny','sobel'], 0)
    canny_low = st.slider('Canny low', 0, 200, 50, 5)
    canny_high = st.slider('Canny high', 0, 300, 150, 5)
    sobel_k = st.slider('Sobel ksize', 3, 7, 3, 2)

with st.sidebar.expander('🔧 Morfologia'):
    do_open = st.checkbox('Opening', False)
    do_close = st.checkbox('Closing', False)
    morph_k = st.slider('Kernel', 3, 15, 5, 2)

tmp = Path(tempfile.mkdtemp(prefix='medimg-streamlit-'))
def save_files(files):
    paths=[]
    for f in files:
        p = tmp/f.name
        open(p,'wb').write(f.read())
        paths.append(str(p))
    return paths

def run(img):
    w = img.copy()
    if use_median: w = pre.median_denoise(pre.to_gray(w), ksize=median_k)
    if use_bilateral: w = pre.bilateral_denoise(pre.to_gray(w), d=bilateral_d, sigma_color=bilateral_sc, sigma_space=bilateral_ss)
    if use_clahe: w = pre.clahe(w, clip_limit=clahe_clip, tile_grid_size=clahe_grid)
    if seg_method=='otsu': m = seg.otsu_threshold(w)
    elif seg_method=='adaptive': m = seg.adaptive_threshold(w, block_size=block, C=C)
    else: m = seg.watershed_segment(w)
    mm = m.copy()
    if do_open: mm = morph.opening(mm, k=morph_k)
    if do_close: mm = morph.closing(mm, k=morph_k)
    e = ed.canny_edges(w, low=canny_low, high=canny_high) if edge_method=='canny' else ed.sobel_edges(w, ksize=sobel_k)
    return w, m, mm, e

def to_bgr(x):
    return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if x.ndim==2 else x

paths = []
if gen_btn:
    sd = tmp/'synth'; sd.mkdir(parents=True, exist_ok=True); generate_dataset(str(sd), n=int(gen_n))
    paths = mio.list_images(str(sd))
elif upload:
    paths = save_files(upload)

if not paths:
    st.info('Wgraj obrazy lub wygeneruj syntetyczne.'); st.stop()

st.success(f'Znaleziono {len(paths)} obraz(ów).')
c = st.columns(4)
for i,n in enumerate(['Wejście','Po preprocessing','Maska (po segm.)','Krawędzie']): c[i].markdown(f'**{n}**')

buf = io.BytesIO()
with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        img = mio.imread(p)
        w, m, mm, e = run(img)
        c1, c2, c3, c4 = st.columns(4)
        c1.image(to_bgr(img), caption=Path(p).name, channels='BGR', use_column_width=True)
        c2.image(to_bgr(w), caption='preprocessing', channels='BGR', use_column_width=True)
        c3.image(to_bgr(mm), caption=f'segment ({seg_method})', channels='BGR', use_column_width=True)
        c4.image(to_bgr(e), caption=f'edges ({edge_method})', channels='BGR', use_column_width=True)
        try:
            st.caption(f'PSNR(pre|in) = {psnr(pre.to_gray(img), pre.to_gray(w)):.2f} dB • SSIM(pre|in) = {ssim(img,w):.4f}')
        except Exception: pass
        stem = Path(p).stem
        _,b1=cv2.imencode('.png', w); z.writestr(f'{stem}_pre.png', b1.tobytes())
        _,b2=cv2.imencode('.png', mm); z.writestr(f'{stem}_mask.png', b2.tobytes())
        _,b3=cv2.imencode('.png', e); z.writestr(f'{stem}_edges.png', b3.tobytes())
st.download_button('⬇️ Pobierz wyniki (ZIP)', data=buf.getvalue(), file_name='results.zip', mime='application/zip')
st.toast('Gotowe! Możesz pobrać ZIP z wynikami.', icon='✅')
