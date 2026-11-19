#!/usr/bin/env python
"""OpenCV's own calibration solvers on T. Schöps's Siemens-star detections (2026-11-06; OpenCV has no Siemens-star detector, so the
detections are Schöps's camera_calibration dataset.bin, the solver is OpenCV's): cv2.calibrateCamera (model pinhole: OpenCV's default
k1 k2 p1 p2 k3), cv2.calibrateCamera with CALIB_RATIONAL_MODEL | CALIB_THIN_PRISM_MODEL | CALIB_TILTED_MODEL (model ocv14), or
cv2.fisheye.calibrate (model fisheye: Kannala-Brandt k1..k4, flags RECOMPUTE_EXTRINSIC | FIX_SKEW, as opencv_calibrate_aprilgrid.py).
Board points: star id -> texture position (--texfeat, the board's texture_features.bin on its --texfeat-size texture) -> the nominal
flat page (--page-mm, metres). Image points: Schöps's (COLMAP pixel frame) - 0.5 = OpenCV's frame. Views with >= --min-features.
pinhole / ocv14 start from OpenCV's own initialisation (initIntrinsicParams2D); --f-init (f px, centre = image centre) adds
USE_INTRINSIC_GUESS — needed by cv2.fisheye.calibrate (OpenCV 5.0's own start does not converge, 2026-10-28). No robust refinement.
  python opencv_calibrate_schoeps.py --dataset <dataset.bin> --texfeat <texture_features.bin> --texfeat-size 4000 4000 --page-mm 1000 1000
                                     --model pinhole|ocv14|fisheye --out <dir> [--min-features 100] [--f-init px]
Writes <out>/calibration.json (K, D = dist, image_size, OpenCV's rms, per-point error median / mean / p90 / p95 / max, per view)."""
import argparse, json, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # siemens_board.py beside this script (tools/)
from siemens_board import read_dataset_bin
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--texfeat", required=True)
ap.add_argument("--texfeat-size", type=float, nargs=2, required=True); ap.add_argument("--page-mm", type=float, nargs=2, required=True)
ap.add_argument("--model", choices=["pinhole", "ocv14", "fisheye"], required=True); ap.add_argument("--out", required=True)
ap.add_argument("--min-features", type=int, default=100); ap.add_argument("--f-init", type=float, default=0.0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
(W, H), sets, _ = read_dataset_bin(a.dataset)
_, tsets, _ = read_dataset_bin(a.texfeat)
XB = {i: (x / a.texfeat_size[0] * a.page_mm[0] * 1e-3, y / a.texfeat_size[1] * a.page_mm[1] * 1e-3, 0.0) for x, y, i in tsets[0][1]}
names, objs, imgs = [], [], []
for name, feats in sets:
    o = [XB[i] for x, y, i in feats if i in XB]; q = [(x - 0.5, y - 0.5) for x, y, i in feats if i in XB]
    if len(o) >= a.min_features: names.append(name); objs.append(np.array(o, np.float64)); imgs.append(np.array(q, np.float64))
print(f"{len(sets)} views in {a.dataset}, {len(names)} with >= {a.min_features} features, {sum(len(o) for o in objs)} features; image {W} x {H}; model {a.model}")
K0 = np.array([[a.f_init, 0, W / 2 - 0.5], [0, a.f_init, H / 2 - 0.5], [0, 0, 1]]) if a.f_init > 0 else None
crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-10)
if a.model in ("pinhole", "ocv14"):
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS if K0 is not None else 0) | ((cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_TILTED_MODEL) if a.model == "ocv14" else 0)
    ob = [o.reshape(-1, 1, 3).astype(np.float32) for o in objs]; im = [q.reshape(-1, 1, 2).astype(np.float32) for q in imgs]   # (OpenCV 5: Point3f / Point2f)
    rms, K, D, rv, tv = cv2.calibrateCamera(ob, im, (W, H), K0, None, flags=flags, criteria=crit)
    proj = [cv2.projectPoints(o, r, t, K, D)[0] for o, r, t in zip(ob, rv, tv)]
else:
    flags = cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW | (cv2.CALIB_USE_INTRINSIC_GUESS if K0 is not None else 0)
    ob = [o.reshape(1, -1, 3) for o in objs]; im = [q.reshape(1, -1, 2) for q in imgs]
    rms, K, D, rv, tv = cv2.fisheye.calibrate(ob, im, (W, H), K0 if K0 is not None else np.zeros((3, 3)), np.zeros((4, 1)), flags=flags, criteria=crit)
    proj = [cv2.fisheye.projectPoints(o.reshape(-1, 1, 3), r, t, K, D)[0] for o, r, t in zip(ob, rv, tv)]
e = [np.linalg.norm(q.reshape(-1, 2) - p.reshape(-1, 2), axis=1) for q, p in zip(imgs, proj)]; ea = np.concatenate(e)
st = dict(median=float(np.median(ea)), mean=float(ea.mean()), p90=float(np.quantile(ea, 0.9)), p95=float(np.quantile(ea, 0.95)), max=float(ea.max()), rms=float(np.sqrt(np.mean(ea ** 2))))
print(f"RESULT OpenCV {a.model} on Schöps's detections: rms {rms:.4f} px (OpenCV), per point median {st['median']:.4f} mean {st['mean']:.4f} p95 {st['p95']:.4f} max {st['max']:.3f} px; "
      f"f ({K[0, 0]:.2f}, {K[1, 1]:.2f}) c ({K[0, 2]:.2f}, {K[1, 2]:.2f}) dist {' '.join('%.5g' % v for v in np.ravel(D))}")
json.dump(dict(model=a.model, opencv=cv2.__version__, K=K.tolist(), D=np.ravel(D).tolist(), dist=np.ravel(D).tolist(), image_size=[W, H], rms_opencv=float(rms),
               error_px=st, n_views=len(names), n_points=int(ea.size), flags=int(flags), f_init=a.f_init or None, dataset=a.dataset,
               per_view=[dict(image=n, n=int(ek.size), rms=float(np.sqrt(np.mean(ek ** 2))), median=float(np.median(ek))) for n, ek in zip(names, e)],
               convention="OpenCV pixel centres (colmap = + 0.5); points = Schöps's star features on the nominal flat page"),
          open(os.path.join(a.out, "calibration.json"), "w"), indent=1)
print("->", os.path.join(a.out, "calibration.json"))
