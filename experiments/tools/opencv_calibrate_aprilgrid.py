#!/usr/bin/env python
"""OpenCV's own calibration of the AprilGrid photos (2026-10-28): OpenCV's AprilTag detector (cv2.aruco.ArucoDetector, DICT_APRILTAG_36h11)
and board model (cv2.aruco.GridBoard: 8 x 8 tags of 90 mm, 27 mm gaps, ids 85-148), then cv2.calibrateCamera (model pinhole: OpenCV's
default k1 k2 p1 p2 k3) or cv2.fisheye.calibrate (model fisheye: Kannala-Brandt k1..k4), or cv2.calibrateCamera with
CALIB_RATIONAL_MODEL | CALIB_THIN_PRISM_MODEL | CALIB_TILTED_MODEL (model ocv14: OpenCV's full 14 coefficients). Settings at OpenCV's defaults except
  markerBorderBits = 2           these tags carry a 2-bit black border; the default 1 decodes nothing (detect_aprilgrid.py)
  cornerRefinementMethod SUBPIX  OpenCV's recommended refinement for calibration (the default is none)
  fisheye flags RECOMPUTE_EXTRINSIC | FIX_SKEW   (the documented usage; --f-init adds USE_INTRINSIC_GUESS with f = --f-init px, c = centre:
                 OpenCV 5.0's own start does not converge even on noise-free synthetic fisheye points, rms 353 px — 2026-10-28)
The printed id layout (which way a row of ids runs, which way the next row lies) is read off the detections in each tag's own frame,
and the GridBoard ids are arranged to match. The points are OpenCV's tag corners (the outer corners of the black border), not the
junctions between tags. Pixel coordinates in OpenCV's convention (pixel centre i; colmap = i + 0.5).
  python opencv_calibrate_aprilgrid.py --images <png dir> --out <dir> --model pinhole|fisheye [--min-markers 4] [--f-init px] [--jobs 8]
Writes <out>/detections.json (cached), <out>/calibration.json (K, D, OpenCV's rms, per-point error median / mean / p90 / max, per view).
rc-results night_2026-11-06/tools copy (06.11, real sets): model ocv14 added; the fisheye corner screen centres on --circle-centre (default
the image centre; the 05.11 synthetic set used its field stop's centre 2141.0 1442.1); calibration.json also carries "dist" (= D).
pinhole / ocv14 run from OpenCV's own start; --f-init (USE_INTRINSIC_GUESS) only when that start does not converge (reported).
rc-results main_2026-11-05/tools copy (05.11): model fisheye drops the outermost corners (radius from the image circle's centre, in
20 px steps from the whole set) until OpenCV 5.0's InitExtrinsics stops asserting fabs(norm_u1) > 0 inside the iterations
(calibration.json corner_radius_max_px; a view left with < 8 corners: excluded_views)."""
import argparse, glob, json, os
from multiprocessing import Pool
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True); ap.add_argument("--out", required=True); ap.add_argument("--model", choices=["pinhole", "ocv14", "fisheye"], required=True)
ap.add_argument("--min-markers", type=int, default=4); ap.add_argument("--f-init", type=float, default=0.0); ap.add_argument("--jobs", type=int, default=8)
ap.add_argument("--ids", default="85-148"); ap.add_argument("--cols", type=int, default=8); ap.add_argument("--rows", type=int, default=8)
ap.add_argument("--circle-centre", type=float, nargs=2, default=None)
ap.add_argument("--tag-m", type=float, default=0.09); ap.add_argument("--gap-m", type=float, default=0.027)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
ID0, ID1 = [int(v) for v in a.ids.split("-")]
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

def detect(f):
    cv2.setNumThreads(1)                                  # (one image per worker process)
    p = cv2.aruco.DetectorParameters(); p.markerBorderBits = 2; p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.ArucoDetector(DICT, p).detectMarkers(g)
    keep = [(int(i), c.reshape(4, 2).tolist()) for c, i in zip(corners, ids.ravel())] if ids is not None else []
    return os.path.splitext(os.path.basename(f))[0], g.shape[::-1], [(i, c) for i, c in keep if ID0 <= i <= ID1]

files = sorted(glob.glob(os.path.join(a.images, "*.png")))
DJ = os.path.join(a.out, "detections.json")
if os.path.isfile(DJ):
    det = json.load(open(DJ))
else:
    with Pool(a.jobs) as pool: res = pool.map(detect, files)
    det = dict(size=list(res[0][1]), images={n: [[i, c] for i, c in m] for n, _, m in res},
               settings="cv2.aruco.ArucoDetector DICT_APRILTAG_36h11, markerBorderBits 2, CORNER_REFINE_SUBPIX, OpenCV " + cv2.__version__)
    json.dump(det, open(DJ, "w"))
W, H = det["size"]
print(f"{len(det['images'])} images, {sum(len(m) for m in det['images'].values())} tags (ids {ID0}-{ID1}); image {W} x {H}")

# ---- the printed layout: the next id along a row and the next row (id + cols), seen in each tag's own frame (x: corner 0 -> 1, y: 0 -> 3)
sx, sy = [], []
for m in det["images"].values():
    D = {i: np.array(c) for i, c in m}
    for i, c in D.items():
        ex = c[1] - c[0]; ey = c[3] - c[0]; ctr = c.mean(0)
        if (i - ID0) % a.cols < a.cols - 1 and i + 1 in D: sx.append(np.sign(np.dot(D[i + 1].mean(0) - ctr, ex)))
        if i + a.cols in D: sy.append(np.sign(np.dot(D[i + a.cols].mean(0) - ctr, ey)))
col_dir, row_dir = int(np.sign(np.sum(sx))), int(np.sign(np.sum(sy)))
print(f"layout: next id along a row lies at tag-frame {'+' if col_dir > 0 else '-'}x ({int(np.sum(np.array(sx) == col_dir))}/{len(sx)} pairs agree), "
      f"next row at {'+' if row_dir > 0 else '-'}y ({int(np.sum(np.array(sy) == row_dir))}/{len(sy)})")
ids = np.array([[ID0 + (r if row_dir > 0 else a.rows - 1 - r) * a.cols + (c if col_dir > 0 else a.cols - 1 - c) for c in range(a.cols)] for r in range(a.rows)]).ravel()
board = cv2.aruco.GridBoard((a.cols, a.rows), a.tag_m, a.gap_m, DICT, ids)

names, objs, imgs = [], [], []
for n, m in sorted(det["images"].items()):
    if len(m) < a.min_markers: continue
    o, p = board.matchImagePoints([np.array(c, np.float32).reshape(1, 4, 2) for _, c in m], np.array([i for i, _ in m], np.int32).reshape(-1, 1))
    names.append(n); objs.append(o.reshape(-1, 1, 3).astype(np.float64)); imgs.append(p.reshape(-1, 1, 2).astype(np.float64))
print(f"{len(names)} views with >= {a.min_markers} tags, {sum(len(o) for o in objs)} corners; model {a.model}")

K0 = np.array([[a.f_init, 0, W / 2 - 0.5], [0, a.f_init, H / 2 - 0.5], [0, 0, 1]]) if a.f_init > 0 else None
if a.model in ("pinhole", "ocv14"):
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS if K0 is not None else 0) | ((cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_TILTED_MODEL) if a.model == "ocv14" else 0)
    objs = [o.astype(np.float32) for o in objs]; imgs = [q.astype(np.float32) for q in imgs]     # (OpenCV 5: Point3f / Point2f only)
    rms, K, D, rv, tv = cv2.calibrateCamera(objs, imgs, (W, H), K0, None, flags=flags)
    proj = [cv2.projectPoints(o, r, t, K, D)[0] for o, r, t in zip(objs, rv, tv)]
else:
    flags = cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW | (cv2.CALIB_USE_INTRINSIC_GUESS if K0 is not None else 0)   # (OpenCV 5: the fisheye flags live in cv2)
    objs = [o.reshape(1, -1, 3) for o in objs]; imgs = [q.reshape(1, -1, 2) for q in imgs]     # (OpenCV 5.0's fisheye.calibrate: 1 x N only)
    # OpenCV 5.0's InitExtrinsics (re-run every iteration under CALIB_RECOMPUTE_EXTRINSIC) asserts fabs(norm_u1) > 0 when a view's
    # corners near 90° of the axis undistort degenerately under the current k1..k4: the outermost corners (radius from the image
    # circle's centre) are dropped in steps until the calibration runs through — the largest radius OpenCV's fisheye model takes
    CC = np.array(a.circle_centre if a.circle_centre else [W / 2, H / 2])   # the screen's centre (COLMAP; OpenCV − 0.5 is immaterial here)
    full = (objs, imgs, names); excluded = []; r_used = None
    for rmax in [np.inf] + list(np.arange(1300.0, 600.0, -20.0)):
        ob, im, nm = [], [], []
        for o, q, n in zip(*full):
            m = np.hypot(q[0, :, 0] - CC[0], q[0, :, 1] - CC[1]) < rmax
            if m.sum() >= 8: ob.append(np.ascontiguousarray(o[:, m])); im.append(np.ascontiguousarray(q[:, m])); nm.append(n)
        try:
            rms, K, D, rv, tv = cv2.fisheye.calibrate(ob, im, (W, H), K0 if K0 is not None else np.zeros((3, 3)), np.zeros((4, 1)), flags=flags,
                                                      criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-10))
            r_used = rmax; objs, imgs = ob, im; excluded = [n for n in full[2] if n not in nm]; names = nm
            print(f"cv2.fisheye.calibrate ran through with the corners within {rmax:.0f} px of the circle centre: {len(nm)} views, {sum(o.shape[1] for o in ob)} of {sum(o.shape[1] for o in full[0])} corners")
            break
        except cv2.error as err:
            print(f"  corners within {rmax:.0f} px: {str(err).strip().splitlines()[-1][:100]}")
    else:
        raise SystemExit("cv2.fisheye.calibrate failed at every radius")
    proj = [cv2.fisheye.projectPoints(o.reshape(-1, 1, 3), r, t, K, D)[0] for o, r, t in zip(objs, rv, tv)]
e = [np.linalg.norm(q.reshape(-1, 2) - p.reshape(-1, 2), axis=1) for q, p in zip(proj, imgs)]; ea = np.concatenate(e)
st = dict(median=float(np.median(ea)), mean=float(ea.mean()), p90=float(np.quantile(ea, 0.9)), max=float(ea.max()), rms=float(np.sqrt(np.mean(ea ** 2))))
print(f"RESULT OpenCV {a.model}: rms {rms:.4f} px (OpenCV), per point median {st['median']:.4f} mean {st['mean']:.4f} p90 {st['p90']:.4f} max {st['max']:.3f} px; "
      f"f ({K[0, 0]:.2f}, {K[1, 1]:.2f}) c ({K[0, 2]:.2f}, {K[1, 2]:.2f}) dist {' '.join('%.5g' % v for v in np.ravel(D))}")
json.dump(dict(model=a.model, opencv=cv2.__version__, K=K.tolist(), D=np.ravel(D).tolist(), dist=np.ravel(D).tolist(), image_size=[W, H], rms_opencv=float(rms), error_px=st,
               n_views=len(names), n_points=int(ea.size), excluded_views=(excluded if a.model == 'fisheye' else []), corner_radius_max_px=(None if a.model != 'fisheye' or r_used in (None, np.inf) else float(r_used)), flags=int(flags), f_init=a.f_init or None, layout=dict(col_dir=col_dir, row_dir=row_dir),
               per_view=[dict(image=n, n=int(ek.size), rms=float(np.sqrt(np.mean(ek ** 2))), median=float(np.median(ek))) for n, ek in zip(names, e)],
               convention="OpenCV pixel centres (colmap = + 0.5); points = the tag corners of cv2.aruco (outer border corners)", images=a.images),
          open(os.path.join(a.out, "calibration.json"), "w"), indent=1)
print("->", os.path.join(a.out, "calibration.json"))
