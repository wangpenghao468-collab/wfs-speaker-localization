import cv2, numpy as np, torch

_sift = cv2.SIFT_create(nfeatures=12000)


def sift_feats(gray):
    """RootSIFT keypoints/descriptors."""
    kp, d = _sift.detectAndCompute(gray, None)
    d = np.sqrt(d / (d.sum(1, keepdims=True) + 1e-7))
    return np.array([k.pt for k in kp], np.float32), d.astype(np.float16)


def mutual_ratio(Da, Db, ratio=0.85):
    """Mutual nearest neighbours passing Lowe's ratio test; Da, Db are cuda tensors."""
    S = (Da @ Db.T).float()
    v, i = S.topk(2, dim=1)
    dist = torch.sqrt(torch.clamp(2 - 2 * v, min=0))
    ok = dist[:, 0] < ratio * dist[:, 1]
    ok &= S.argmax(0)[i[:, 0]] == torch.arange(len(Da), device=Da.device)
    q = torch.where(ok)[0]
    return q.cpu().numpy(), i[q, 0].cpu().numpy()


def sony_camera(cam):
    """SIMPLE_RADIAL -> (K, dist) for OpenCV."""
    f, cx, cy, k = cam["params"]
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]), np.array([k, 0, 0, 0, 0.0])
