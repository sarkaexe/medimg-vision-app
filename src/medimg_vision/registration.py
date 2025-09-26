import numpy as np, cv2
from .preprocessing import to_gray
def ecc_align(ref, moving, warp_mode=cv2.MOTION_EUCLIDEAN, number_of_iterations=100, termination_eps=1e-6):
    ref_g=to_gray(ref).astype(np.float32); mov_g=to_gray(moving).astype(np.float32)
    warp=np.eye(2,3,dtype=np.float32) if warp_mode in (cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN) else np.eye(3,3,dtype=np.float32)
    cc, warp = cv2.findTransformECC(ref_g, mov_g, warp, warp_mode, (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps), None, 5)
    if warp_mode==cv2.MOTION_HOMOGRAPHY:
        aligned=cv2.warpPerspective(moving, warp, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
    else:
        aligned=cv2.warpAffine(moving, warp, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
    return aligned, warp, cc
