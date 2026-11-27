#!/usr/bin/env python3
"""texture_learned_photograph.py <evaluate dir> <poses.json> <frames dir> <out dir> --roles=<page.json> [--bands=a,b,…] [--read-on] [--plain-cut]

The band "training photograph" of fig:texture-learned (the paper's Fig. 7, right): the first three lines of the printed page's
text as ONE photograph of the texture run shows them. AUTHOR-SIDE: it needs rawpy (LibRaw) and OpenCV (cv2, with SIFT), which no
machine of the run of record has, so no driver stage runs it; what it writes is an input of record of scripts/texture_recovery.jl's
stage `bands` (configs/inputs/texture_learned_photograph_20261121.toml records the step: the tool's versions, its inputs, its
outputs). This file is the photograph part of the step `tables` of T. Harder's texture_learned.py (the figure folder's tools/,
2026-11-16), taken out of it line for line; the rest of that step is ported to Julia (src/paper_figures/texture_learned.jl).

  photograph  a TRAINING photograph: one the run's texture is optimised on (role "fit" in the refine stage's page.json — a third
              of the run's photographs are validation and test photographs and are not eligible). Of those the one whose sheet is
              closest to fronto-parallel and near the image centre (of the eight with the least tilt the one with the least
              tilt + off-axis angle). The band is the photograph itself, NOT rectified: its RAW decoded linearly (camera white
              balance, no auto-brightening, Rec. 709 luminance) and a cut of it at its own scale, without any perspective
              correction. The camera was rolled by about 4 deg against the page in every near-frontal photograph of the set, so
              the cut is TURNED in the image plane by the text lines' angle, about the band's centre, so that the lines are level
              (a rigid rotation, bicubic); --plain-cut keeps the axis-aligned cut of the untouched pixels instead (the lines
              slanted). The band's place is found through a homography design -> photograph (SIFT matches, RANSAC) that is used
              for nothing else; the grey is scaled such that the paper's white matches the design's.
  the lines   the cuts lie in the middle of the white gaps between two text lines of the design; a band begins half a line gap
              above its first line. Every band of the figure shows the first three lines; with --read-on the text reads on from
              band to band (lines 1-3, 4-6, …) and the photograph shows the lines of its place among --bands (any of truth,
              photograph, pinhole, ours, in the figure's order from the top; the default is that order).

Reads   <evaluate dir>: the evaluate stage's product of the texture run (design.png — the printed design, the reference the
        photograph's place is found by — and report.json: tex_mm, TW, TH); <poses.json>: the init stage's (the photographs'
        poses); --roles=<page.json>: the refine stage's (photographs, roles); <frames dir>: the folder of the RAW photographs.
Writes  <out dir>/photograph.png — the cut, 8-bit grey (design units 0 … 1 as 0 … 255), its own pixels;
        <out dir>/meta.toml — the record of the cut: the layout it was cut for (text block, region, lines per band, order),
        [bands.photograph] (its text lines, the cut's size and its bounding box in the photograph) and [photograph] (the file,
        its number among the training photographs, its pose, the lines' angle, the scale, the registration's counts)."""
import json, os, sys
import numpy as np
from PIL import Image

BOX = (100, 100, 727, 1065)         # the page's text block in texels (x0, y0, x1, y1): the white box inside the blob border
PER = 3                             # text lines per band
INK = 0.85                          # a row of the text block lies in a gap between two lines when the design's darkest texel in it is lighter than this
DESIGN = "design"                   # the run's file of the printed design (the ground truth): the reference
ALL = {s[0]: s for s in (           # slug, the run's texture file (None: the photograph)
    ("truth", DESIGN), ("photograph", None), ("pinhole", "texture_pinhole"), ("ours", "texture_learned"))}
ORDER = ("truth", "photograph", "pinhole", "ours")   # the bands from the top (--bands=… selects others)
SOURCES = tuple(ALL[s] for s in ORDER)
TRAINING = "fit"                    # the role (page.json) of a photograph the texture is optimised on


def write(path, lines):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def photograph(edir, poses, frames, tdir, page, read_on=False, level=True):
    import cv2, rawpy                                                          # the photograph's decoding and its place only
    rep = json.load(open(f"{edir}/report.json")); tex = float(rep["tex_mm"]); x0, y0, x1, y1 = BOX
    def load(n):
        im = Image.open(f"{edir}/{n}.png"); assert im.mode == "L", (n, im.mode); return np.asarray(im).astype(np.float32) / 255
    design = load(DESIGN); TH, TW = design.shape; T = {}
    assert (TW, TH) == (rep["TW"], rep["TH"]) and x1 <= TW and y1 <= TH, (TW, TH)
    box = (slice(y0, y1), slice(x0, x1))
    # the bands: cut in the middle of the gaps between the text lines, PER lines each; rows of the text block. Every band the first
    # PER lines; with read_on band k the lines PER (k − 1) + 1 … PER k
    white = design[box].min(axis=1) > INK; edges = np.flatnonzero(np.diff(white.astype(int)))
    gaps = [(a + 1, b + 1) for a, b in zip(edges[:-1], edges[1:]) if white[a + 1]]                                                  # (first, past-last) row of each gap between two lines
    assert len(gaps) >= PER * (len(SOURCES) if read_on else 1), len(gaps)
    mid = lambda i: int(gaps[i - 1][0] + gaps[i - 1][1]) // 2                                                                       # the middle of the gap after text line i (1-based)
    top = max(0, int(np.flatnonzero(~white)[0]) - (gaps[0][1] - gaps[0][0]) // 2)
    if read_on: cuts = [mid(PER * k) for k in range(1, len(SOURCES) + 1)]; bands = list(zip([top] + cuts[:-1], cuts))
    else: bands = [(top, mid(PER))] * len(SOURCES)
    lines = [(PER * k + 1, PER * (k + 1)) if read_on else (1, PER) for k in range(len(SOURCES))]
    region = (x0, y0 + top, x1, y0 + max(b for _, b in bands)); os.makedirs(tdir, exist_ok=True)
    # the photograph: a training photograph (role TRAINING); the sheet's normal (world z) against the optical axis, and the sheet near the axis
    views = json.load(open(poses))["per_view"]; pg = json.load(open(page))
    assert [v["view"] for v in views] == list(pg["photographs"]) and len(pg["roles"]) == len(views), "the poses and the run's photographs (page.json) differ"
    fit = [v for v, r in zip(views, pg["roles"]) if r == TRAINING]; assert fit, f"no photograph of role {TRAINING}"
    def pose(v):
        R = np.array(v["R_world_to_camera"], float).reshape(3, 3); t = np.array(v["t_world_to_camera"], float).ravel()
        return float(np.degrees(np.arccos(min(1.0, abs(R[2, 2]))))), float(np.degrees(np.arctan2(np.hypot(t[0], t[1]), t[2]))), float(v["camera_distance_m"])
    rank = sorted(fit, key=lambda v: pose(v)[0]); best = min(rank[:8], key=lambda v: pose(v)[0] + pose(v)[1]); stem = os.path.splitext(best["view"])[0]
    number, of = fit.index(best) + 1, len(fit)                                                                                     # its number among the training photographs, in the run's (file) order
    raw_name = next(f for f in sorted(os.listdir(frames)) if os.path.splitext(f)[0] == stem); tilt, off, dist = pose(best)
    with rawpy.imread(f"{frames}/{raw_name}") as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16, gamma=(1, 1)).astype(np.float32) / 65535   # linear
    photo = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32); PH, PW = photo.shape
    # where the band lies in the photograph: a homography design -> photograph, used for the cut's place only
    p8 = np.clip(255 * (photo / np.percentile(photo, 99.5)) ** (1 / 2.2), 0, 255).astype(np.uint8); d8 = (design * 255).astype(np.uint8)
    sift = cv2.SIFT_create(nfeatures=20000, contrastThreshold=0.01)
    k1, f1 = sift.detectAndCompute(cv2.GaussianBlur(d8, (0, 0), 2.0), None); k2, f2 = sift.detectAndCompute(p8, None)
    good = [a for a, b in cv2.BFMatcher(cv2.NORM_L2).knnMatch(f1, f2, k=2) if a.distance < 0.75 * b.distance]
    src = np.float32([k1[a.queryIdx].pt for a in good]); dst = np.float32([k2[a.trainIdx].pt for a in good])
    Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0); assert Hm is not None and inl.sum() >= 30, "the photograph's registration failed"
    pa, pb = bands[ORDER.index("photograph")] if "photograph" in ORDER else bands[0]                                               # the photograph band's rows of the text block
    rect = np.float32([[[x0 - .5, y0 + pa - .5]], [[x1 - .5, y0 + pa - .5]], [[x1 - .5, y0 + pb - .5]], [[x0 - .5, y0 + pb - .5]]])  # the band's rectangle on the page (pixel-centre coordinates)
    q = cv2.perspectiveTransform(rect, Hm).reshape(4, 2); scale = float(np.linalg.norm(q[1] - q[0]) / (x1 - x0))
    inplane = float(np.degrees(np.arctan2(q[1, 1] - q[0, 1], q[1, 0] - q[0, 0])))                                                 # the text lines' direction in the photograph
    assert abs(inplane) < 45, f"the page is turned by {inplane:.0f} deg in the photograph: a plain cut would show the text on its side"
    xa, ya = (int(np.floor(v + .5)) for v in q.min(axis=0)); xb, yb = (int(np.ceil(v + .5)) for v in q.max(axis=0))                   # the bounding box, whole pixels (first, past-last)
    assert 0 <= xa < xb <= PW and 0 <= ya < yb <= PH, ("the band lies partly outside the photograph", (xa, ya, xb, yb), (PW, PH))
    pbox = (xa, ya, xb, yb)                                                                                                        # the band's bounding box in the photograph (the record's photo_box)
    if level:   # turned in the image plane about the band's centre by the lines' angle, scale 1: the lines level, no perspective correction
        c = q.mean(axis=0); A = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), inplane, 1.0); qr = q @ A[:, :2].T + A[:, 2]
        assert abs(qr[1, 1] - qr[0, 1]) < 0.5, ("the rotation does not level the lines", qr)
        xa, ya = (int(np.floor(v + .5)) for v in qr.min(axis=0)); xb, yb = (int(np.ceil(v + .5)) for v in qr.max(axis=0))
        A[:, 2] -= (xa, ya); cut = cv2.warpAffine(photo, A, (xb - xa, yb - ya), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else: cut = photo[ya:yb, xa:xb]
    T["photograph"] = np.clip(cut * (np.percentile(design[y0 + pa:y0 + pb, x0:x1], 90) / np.percentile(cut, 90)), 0, 1)           # the paper's white as the design's (linear grey); the pixels are the photograph's own
    Image.fromarray(np.round(np.clip(T["photograph"], 0, 1) * 255).astype(np.uint8), "L").save(f"{tdir}/photograph.png", optimize=True)
    M = ["# The training photograph's band of fig:texture-learned (experiments/tools/texture_learned_photograph.py): photograph.png — a cut",
         "# of the decoded photograph at its own scale, not rectified (8-bit grey, design units 0 … 1 as 0 … 255). An input of record of",
         "# scripts/texture_recovery.jl's stage bands, which computes the rest of the figure's tables. Texel coordinates are 0-based, x right,",
         "# y down; a range is first, past-last. [bands.photograph] rows: the band's rows of the region of the bands' text lines; lines: its",
         "# text lines (1-based) of the text block; photo_box: the bounding box of its rectangle in the photograph (its pixels); pixels: the",
         "# cut's width and height; [photograph] turned_deg: the cut is turned in the image plane by this angle about the band's centre so",
         "# that the text lines are level (0: the untouched pixels). design: the run's file of the printed design, the reference of the",
         "# photograph's place and of the cuts.",
         f"tex_mm = {tex:g}", f"page = [{TW}, {TH}]", f"text_block = [{x0}, {y0}, {x1}, {y1}]", f"region = [{', '.join(str(int(v)) for v in region)}]", f'design = "{DESIGN}.png"',
         f"lines_per_band = {PER}", f"read_on = {'true' if read_on else 'false'}", "order = [" + ", ".join(f'"{s[0]}"' for s in SOURCES) + "]"]
    for (a, b), (la, lb), (slug, n) in zip(bands, lines, SOURCES):
        if slug == "photograph":
            M += ["", f"[bands.{slug}]", f'source = "{n + ".png" if n else raw_name}"', f"rows = [{a - top}, {b - top}]", f"lines = [{la}, {lb}]"]
            M += [f"pixels = [{xb - xa}, {yb - ya}]", f"photo_box = [{', '.join(str(v) for v in pbox)}]"]
    M += ["", "[photograph]", f'view = "{raw_name}"', f'role = "{TRAINING}"', f"number = {number}", f"of = {of}", f"of_all_photographs = {len(views)}",
          f"tilt_deg = {tilt:.2f}", f"off_axis_deg = {off:.2f}", f"distance_m = {dist:.3f}", f"in_plane_deg = {inplane:.2f}", f"turned_deg = {inplane if level else 0:.2f}", "rectified = false",
          f"photo_px_per_texel = {scale:.3f}", f"sift_matches = {len(good)}", f"homography_inliers = {int(inl.sum())}"]
    write(f"{tdir}/meta.toml", M)
    print(f"-> {tdir}/  (photograph.png, meta.toml; region {tuple(int(v) for v in region)}: {x1 - x0} x {region[3] - region[1]} texels of {tex:g} mm; {len(gaps) + 1} text lines in the block)")
    print(f"   photograph {raw_name}: training photograph {number} of {of} ({len(views)} photographs in all); tilt {tilt:.1f} deg, {off:.1f} deg off the axis, {dist:.2f} m, "
          f"lines turned {inplane:.1f} deg in the image; {scale:.2f} photo px per texel; cut {xb - xa} x {yb - ya} px, {'turned level' if level else 'untouched pixels'}, band at {pbox}; {len(good)} SIFT matches, {int(inl.sum())} homography inliers")
    print(f"   rawpy {rawpy.__version__} (LibRaw {'.'.join(str(v) for v in rawpy.libraw_version)}), OpenCV {cv2.__version__}, NumPy {np.__version__}, Pillow {Image.__version__}")


if __name__ == "__main__":
    a = sys.argv[1:]; page = None; read_on = False; level = True
    for o in [x for x in a if x.startswith("--")]:
        a.remove(o)
        if o.startswith("--bands="):
            ORDER = tuple(o.split("=", 1)[1].split(",")); assert ORDER and all(s in ALL for s in ORDER) and len(set(ORDER)) == len(ORDER), ORDER
            SOURCES = tuple(ALL[s] for s in ORDER)
        elif o.startswith("--roles="): page = o.split("=", 1)[1]
        elif o == "--read-on": read_on = True
        elif o == "--plain-cut": level = False
        else: sys.exit(f"unknown option {o}\n\n{__doc__}")
    if len(a) != 4: sys.exit(__doc__)
    if page is None: sys.exit("--roles=<page.json> is needed (the refine stage's record of the photographs' roles): the photograph must be a training photograph")
    photograph(a[0], a[1], a[2], a[3], page, read_on, level)
