#!/usr/bin/env python
"""The 1 m Siemens-star board (geometric_accuracy/data/target/siemens_1m/30x30: Schöps's generator create_calibration_pattern.py,
30 × 30 squares of 33.07 mm, 16 segments per star, one AprilTag 36h11 #9 of 4 × 4 squares at the centre, 816 star features) as the inputs
of the schoeps front-end of refine_grid_photos.jl and of the renderer (render_gt_siemens.jl), EXACTLY from the generator's geometry:
  texture.png            the 4 px/mm raster of the print file as RGB (the board texture; with the distance map the albedo comes from the
                         printed-edge model, the texture only sets the texel grid)
  texture_features.bin   the 816 star centres at their EXACT lattice positions (texture px of the 4000 × 4000 raster, colmap: texel i spans
                         [i, i+1)), ids = Schöps's feature ids (the lattice table of the target's texture_features_4ppmm.bin: id ↔ (gx, gy),
                         lattice point i = gx + 1 — checked against the detector's positions on the raster)
  sdf.f32 + sdf.json     the design's signed distance map at --sdf-ppm px/mm (+ on paper, − in ink; the renderer's printed-edge model,
                         RC_EDGE_SDF) — ANALYTIC: the distance from every texel centre to the nearest colour boundary of the generator's
                         drawing (the 16 sector rays of every star, the star-square edges, the pattern border, the tag border, the tag cells);
                         every candidate segment is kept iff the colours on its two sides differ (sampled 1 µm off its midpoint)
  board.json             the layout
  dataset.bin            with --truth <poses.json>: the TRUE projections of every view's visible stars (one set per view, colmap px) — the
                         scoring board (board_true/) of run_refine_siemens.sh; with --detections <dataset.bin>: that file copied
  python siemens_board.py --out <dir> [--target <30x30 dir>] [--sdf] [--sdf-ppm 8] [--sdf-clip-mm 6] [--check]
                          [--truth <poses.json> | --detections <dataset.bin>] [--image-size 5208,3476] [--no-texture]
Geometry (the generator, PDF y up → texture y down): pattern x, y ∈ [4, 996] mm; star centres at 4 + i·s, i = 0..30 (the border ones are
half / quarter stars, not features); each star = the square of side s around its centre, cut into 16 sectors by the rays at θ_k = k·22.5°,
direction (−sin θ_k, −cos θ_k) in the texture frame, sector k ∈ [θ_k, θ_{k+1}) black iff k even; the colours flip across every star-square
edge; the tag (squares 13..16 both ways, 132.3 mm) = the 10 × 10 cells of the tag image (white outer ring) in place of the stars."""
import argparse, json, os, re, struct, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.normpath(os.path.join(HERE, "..", "data", "target", "siemens_1m", "30x30"))   # the design files of the print (Schöps's generator output)
PPM_TEX = 4.0                                             # the raster's resolution [px/mm] (texture_features_4ppmm.bin)


def read_dataset_bin(path):
    """Schöps calib_data v0 (mixed endian): (W, H), [(name, [(x, y, id)])], [(cell_m, {id: (gx, gy)})]."""
    b = open(path, "rb").read(); assert b[:10] == b"calib_data", path
    o = 10; _, ncam = struct.unpack(">II", b[o:o + 8]); o += 8; assert ncam == 1
    W, H = struct.unpack(">II", b[o:o + 8]); o += 8
    nset, = struct.unpack(">I", b[o:o + 4]); o += 4; sets = []
    for _ in range(nset):
        L, = struct.unpack(">I", b[o:o + 4]); o += 4; name = b[o:o + L].decode(); o += L
        nf, = struct.unpack(">I", b[o:o + 4]); o += 4
        f = [struct.unpack("<ff", b[o + 12 * k:o + 12 * k + 8]) + struct.unpack(">i", b[o + 12 * k + 8:o + 12 * k + 12]) for k in range(nf)]; o += 12 * nf
        sets.append((name, f))
    geoms = []
    if o < len(b):
        ng, = struct.unpack(">I", b[o:o + 4]); o += 4
        for _ in range(ng):
            cell, = struct.unpack("<f", b[o:o + 4]); n, = struct.unpack(">I", b[o + 4:o + 8]); o += 8; d = {}
            for _ in range(n):
                i, x, y = struct.unpack(">iii", b[o:o + 12]); o += 12; d[i] = (x, y)
            geoms.append((cell, d))
    return (W, H), sets, geoms


def write_feature_file(path, W, H, sets, geoms=None):
    """Schöps-format feature file (as aprilgrid_board.py): sets = [(name, [(x, y, id), …])]; geoms = [(cell_m, {id: (gx, gy)})] (the
    pattern's lattice table, needed by Schöps's calibration of a dataset file)."""
    with open(path, "wb") as f:
        f.write(b"calib_data"); f.write(struct.pack(">II", 0, 1)); f.write(struct.pack(">II", W, H)); f.write(struct.pack(">I", len(sets)))
        for name, feats in sets:
            b = name.encode(); f.write(struct.pack(">I", len(b))); f.write(b); f.write(struct.pack(">I", len(feats)))
            for x, y, i in feats:
                f.write(struct.pack("<ff", x, y)); f.write(struct.pack(">i", i))
        if geoms:
            f.write(struct.pack(">I", len(geoms)))
            for cell, d in geoms:
                f.write(struct.pack("<f", cell)); f.write(struct.pack(">I", len(d)))
                for i in sorted(d): f.write(struct.pack(">iii", i, d[i][0], d[i][1]))


class Pattern:
    """The generator's drawing in the texture frame (mm, x right, y down from the page's top-left)."""
    def __init__(self, target):
        y = open(os.path.join(target, [f for f in os.listdir(target) if f.endswith(".yaml")][0])).read()
        g = lambda k: float(re.search(rf"{k}:\s*([-0-9.e]+)", y).group(1))
        self.nseg = int(g("num_star_segments")); self.nx = int(g("squares_x")); self.ny = int(g("squares_y")); self.s = 1e3 * g("square_length_in_meters")
        self.page = (g("width_mm"), g("height_mm")); self.x0, self.y0, self.xe, self.ye = g("pattern_start_x_mm"), g("pattern_start_y_mm"), g("pattern_end_x_mm"), g("pattern_end_y_mm")
        self.tag = (int(g("tag_x")), int(g("tag_y")), int(g("width")), int(g("index")))
        tag_png = os.path.join(target, "..", "tag36_11_%05d.png" % self.tag[3])
        im = cv2.imread(tag_png, cv2.IMREAD_UNCHANGED); self.tagbits = (im[..., 0] if im.ndim == 3 else im) < 128     # [row (y), col (x)]: True = black
        assert self.tagbits.shape == (10, 10) and not self.tagbits[0].any() and not self.tagbits[-1].any(), "tag image: 10 × 10 with a white ring expected"
        self.tx0 = self.x0 + self.tag[0] * self.s; self.ty0 = self.y0 + self.tag[1] * self.s; self.tw = self.tag[2] * self.s; self.cell = self.tw / 10
        assert self.nseg == 16 and abs(self.page[0] - 1000) < 1e-6 and abs(self.page[1] - 1000) < 1e-6

    def colour(self, px, py):
        """Ink (True) at the texture-frame points (mm); white outside the pattern rectangle."""
        px = np.asarray(px, float); py = np.asarray(py, float); s = self.s
        inside = (px >= self.x0) & (px <= self.xe) & (py >= self.y0) & (py <= self.ye)
        intag = (px >= self.tx0) & (px < self.tx0 + self.tw) & (py >= self.ty0) & (py < self.ty0 + self.tw)
        i = np.rint((px - self.x0) / s); j = np.rint((py - self.y0) / s)
        u = px - (self.x0 + i * s); v = py - (self.y0 + j * s)
        th = np.mod(np.arctan2(-u, -v), 2 * np.pi); k = np.floor(th / (2 * np.pi / self.nseg)).astype(int) % self.nseg
        star = (k % 2 == 0)
        a = np.clip(np.floor((px - self.tx0) / self.cell).astype(int), 0, 9); b = np.clip(np.floor((py - self.ty0) / self.cell).astype(int), 0, 9)
        return np.where(intag, self.tagbits[b, a], star) & inside

    def lattice_mm(self, gx, gy):
        """Schöps lattice (gx, gy) → star centre (mm): lattice point i = gx + 1."""
        return self.x0 + (gx + 1) * self.s, self.y0 + (gy + 1) * self.s

    def segments(self, eps=1e-3):
        """Every colour boundary of the drawing as segments (x0, y0, x1, y1) [mm]: the candidates below, kept iff the colours 1 µm off
        the two sides of the midpoint differ."""
        s = self.s; S = []
        # (a) the sector rays of every star (0..30 both ways), clipped to the pattern rectangle
        for i in range(self.nx + 1):
            for j in range(self.ny + 1):
                cx, cy = self.x0 + i * s, self.y0 + j * s
                for k in range(self.nseg):
                    th = 2 * np.pi * k / self.nseg; dx, dy = -np.sin(th), -np.cos(th); t1 = 0.5 * s / max(abs(dx), abs(dy))
                    t0 = 0.0                                                       # clip t ∈ [0, t1] to the rectangle (Liang–Barsky)
                    for d, lo, hi, c in ((dx, self.x0, self.xe, cx), (dy, self.y0, self.ye, cy)):
                        if abs(d) < 1e-12: continue
                        ta, tb = sorted(((lo - c) / d, (hi - c) / d)); t0 = max(t0, ta); t1 = min(t1, tb)
                    if t1 - t0 > 1e-9: S.append((cx + t0 * dx, cy + t0 * dy, cx + t1 * dx, cy + t1 * dy))
        # (b) the star-square edges (the lines half a square off the lattice), in pieces of s/2 (the adjacent sectors change at the lattice lines)
        half = np.arange(0, 2 * self.ny + 1) * s / 2
        for i in range(self.nx):
            x = self.x0 + (i + 0.5) * s
            for m in range(2 * self.ny): S.append((x, self.y0 + half[m], x, self.y0 + half[m + 1]))
        for j in range(self.ny):
            y = self.y0 + (j + 0.5) * s
            for m in range(2 * self.nx): S.append((self.x0 + half[m], y, self.x0 + half[m + 1], y))
        # (c) the pattern border and (d) the tag border, in pieces of s/2
        for (xa, ya, xb, yb) in ((self.x0, self.y0, self.xe, self.ye), (self.tx0, self.ty0, self.tx0 + self.tw, self.ty0 + self.tw)):
            n = int(round((xb - xa) / (s / 2)))
            for m in range(n):
                S.append((xa + m * s / 2, ya, xa + (m + 1) * s / 2, ya)); S.append((xa + m * s / 2, yb, xa + (m + 1) * s / 2, yb))
                S.append((xa, ya + m * s / 2, xa, ya + (m + 1) * s / 2)); S.append((xb, ya + m * s / 2, xb, ya + (m + 1) * s / 2))
        # (e) the tag cells: the edges between cells of different colour
        c = self.cell; T = self.tagbits
        for b in range(10):
            for a in range(10):
                if a < 9 and T[b, a] != T[b, a + 1]: S.append((self.tx0 + (a + 1) * c, self.ty0 + b * c, self.tx0 + (a + 1) * c, self.ty0 + (b + 1) * c))
                if b < 9 and T[b, a] != T[b + 1, a]: S.append((self.tx0 + a * c, self.ty0 + (b + 1) * c, self.tx0 + (a + 1) * c, self.ty0 + (b + 1) * c))
        S = np.array(S, float); n = len(S)
        mx, my = 0.5 * (S[:, 0] + S[:, 2]), 0.5 * (S[:, 1] + S[:, 3]); tx, ty = S[:, 2] - S[:, 0], S[:, 3] - S[:, 1]; L = np.hypot(tx, ty)
        nx, ny = -ty / L, tx / L
        keep = self.colour(mx + eps * nx, my + eps * ny) != self.colour(mx - eps * nx, my - eps * ny)
        return S[keep], n


def distance_map(pat, ppm, clip_mm):
    """Signed distance [mm] from every texel centre ((u + ½)/ppm) of the page at ppm px/mm to the nearest colour boundary, + on paper,
    clipped to ±clip_mm. Exact (to float32) for |d| < clip: per star square (the texel block of lattice point (i, j), the margin and the
    tag included in the border / central blocks) the minimum over every boundary segment whose bounding box comes within clip of the block."""
    S, ncand = pat.segments(); s = pat.s
    N = int(round(pat.page[0] * ppm)); assert abs(N - pat.page[0] * ppm) < 1e-9 and abs(pat.page[1] - pat.page[0]) < 1e-9
    D = np.empty((N, N), np.float32)
    bb = np.c_[np.minimum(S[:, 0], S[:, 2]), np.minimum(S[:, 1], S[:, 3]), np.maximum(S[:, 0], S[:, 2]), np.maximum(S[:, 1], S[:, 3])]
    ax, ay = S[:, 0].astype(np.float32), S[:, 1].astype(np.float32); tx, ty = (S[:, 2] - S[:, 0]).astype(np.float32), (S[:, 3] - S[:, 1]).astype(np.float32)
    L2 = tx * tx + ty * ty
    edges = lambda n, o: [0] + [int(round((o + (m + 0.5) * s) * ppm)) for m in range(n)] + [N]
    ux, uy = edges(pat.nx, pat.x0), edges(pat.ny, pat.y0)                           # texel block edges: half a square off the lattice
    for j in range(pat.ny + 1):
        for i in range(pat.nx + 1):
            u0, u1, v0, v1 = ux[i], ux[i + 1], uy[j], uy[j + 1]
            xa, xb, ya, yb = u0 / ppm, u1 / ppm, v0 / ppm, v1 / ppm
            c = np.nonzero((bb[:, 0] < xb + clip_mm) & (bb[:, 2] > xa - clip_mm) & (bb[:, 1] < yb + clip_mm) & (bb[:, 3] > ya - clip_mm))[0]
            X, Y = np.meshgrid((np.arange(u0, u1, dtype=np.float32) + 0.5) / ppm, (np.arange(v0, v1, dtype=np.float32) + 0.5) / ppm)
            px, py = X.ravel()[:, None], Y.ravel()[:, None]
            wx, wy = px - ax[c], py - ay[c]
            t = np.clip((wx * tx[c] + wy * ty[c]) / L2[c], 0, 1)
            d = np.sqrt((wx - t * tx[c]) ** 2 + (wy - t * ty[c]) ** 2).min(axis=1)
            d = np.minimum(d, clip_mm); ink = pat.colour(X.ravel(), Y.ravel())
            D[v0:v1, u0:u1] = np.where(ink, -d, d).reshape(v1 - v0, u1 - u0)
    return D, len(S), ncand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--target", default=TARGET)
    ap.add_argument("--sdf", action="store_true"); ap.add_argument("--sdf-ppm", type=float, default=8.0); ap.add_argument("--sdf-clip-mm", type=float, default=6.0)
    ap.add_argument("--check", action="store_true", help="the analytic drawing against the target's 4 px/mm raster and its detected features")
    ap.add_argument("--truth", default=None, help="gt/poses.json → dataset.bin of the true projections (the scoring board)")
    ap.add_argument("--detections", default=None, help="a Schöps dataset.bin → copied as this board's dataset.bin")
    ap.add_argument("--merge", default=None, help="a directory of per-view detections (<V>/dataset.bin) → one dataset.bin (sorted by view, the lattice table kept)")
    ap.add_argument("--image-size", default="5208,3476"); ap.add_argument("--no-texture", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    pat = Pattern(a.target); s = pat.s
    (TW, TH), tsets, geoms = read_dataset_bin(os.path.join(a.target, "texture_features_4ppmm.bin"))
    assert (TW, TH) == (4000, 4000) and len(geoms) == 1, (TW, TH, len(geoms))
    lat = geoms[0][1]; det = {i: (x, y) for x, y, i in tsets[0][1]}
    feats = []
    for i in sorted(lat):
        x, y = pat.lattice_mm(*lat[i]); feats.append((x * PPM_TEX, y * PPM_TEX, i))
    dev = np.array([np.hypot(det[i][0] - x, det[i][1] - y) for x, y, i in feats])
    print(f"{len(feats)} star features at the exact lattice (s = {s:.4f} mm, pattern from {pat.x0:g} mm, tag squares {pat.tag[0]}..{pat.tag[0] + pat.tag[2] - 1}); "
          f"Schöps's detector on the raster deviates by median {np.median(dev):.4f}, max {dev.max():.4f} texture px ({1e3 * np.median(dev) / PPM_TEX:.1f} µm)")
    assert dev.max() < 0.5, "the lattice ↔ id mapping does not match the detector's positions"
    write_feature_file(os.path.join(a.out, "texture_features.bin"), TW, TH, [("texture.png", feats)])
    raster = os.path.join(a.target, [f for f in os.listdir(a.target) if f.endswith("_4ppmm.png")][0])
    tex = cv2.imread(raster, cv2.IMREAD_GRAYSCALE); assert tex.shape == (TH, TW)
    if not a.no_texture: cv2.imwrite(os.path.join(a.out, "texture.png"), cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR))
    layout = {"target": os.path.abspath(a.target), "page_mm": list(pat.page), "square_mm": s, "squares": [pat.nx, pat.ny], "pattern_mm": [pat.x0, pat.y0, pat.xe, pat.ye],
              "segments": pat.nseg, "tag": {"squares": [pat.tag[0], pat.tag[1]], "size_squares": pat.tag[2], "index": pat.tag[3], "mm": [pat.tx0, pat.ty0, pat.tw]},
              "texture_px": [TW, TH], "px_per_mm": PPM_TEX, "n_features": len(feats), "feature_id": "Schöps's (the target's lattice table); lattice point i = gx + 1",
              "features": "exact lattice positions (4 + i s mm) in texture px, colmap (texel i spans [i, i+1))",
              "frame": "texture x right, y down (the print seen from the front)", "detector_on_raster_px": {"median": float(np.median(dev)), "max": float(dev.max())},
              "env": {"RC_PAGE_MM": "1000,1000", "RC_TEXFEAT_SIZE": f"{TW},{TH}"}}
    if a.check:
        u = (np.arange(TW) + 0.5) / PPM_TEX; X, Y = np.meshgrid(u, u)
        ink = pat.colour(X, Y); mism = ink != (tex < 128); grey = (tex > 16) & (tex < 240)
        print(f"check vs the raster: {mism.sum()} of {tex.size} texels differ ({100 * mism.mean():.4f} %), {(mism & grey).sum()} of them anti-aliased edge texels, "
              f"{(mism & ~grey).sum()} pure black/white texels → {'OK' if (mism & ~grey).sum() < 200 else 'MISMATCH'}")
        layout["check"] = {"raster_mismatch_texels": int(mism.sum()), "raster_mismatch_solid_texels": int((mism & ~grey).sum())}
        (mism & ~grey).sum() < 200 or sys.exit("the analytic drawing does not match the raster")
    if a.sdf:
        assert a.sdf_ppm == int(a.sdf_ppm) and a.sdf_ppm >= 8
        D, nseg, ncand = distance_map(pat, a.sdf_ppm, a.sdf_clip_mm)
        D.tofile(os.path.join(a.out, "sdf.f32"))
        json.dump({"height": int(D.shape[0]), "width": int(D.shape[1]), "page_mm": list(pat.page), "px_per_mm": a.sdf_ppm, "clip_mm": a.sdf_clip_mm, "dtype": "float32",
                   "order": "row-major, row = texture y (down), column = texture x", "sign": "+ on paper, − in ink", "unit": "mm",
                   "method": f"analytic: distance to the nearest of {nseg} colour-boundary segments of the generator's drawing ({ncand} candidates)"},
                  open(os.path.join(a.out, "sdf.json"), "w"), indent=1)
        g = np.hypot(*np.gradient(D[::4, ::4].astype(np.float64), 4 / a.sdf_ppm)); inner = np.abs(D[::4, ::4]) < a.sdf_clip_mm - 1
        print(f"distance map {D.shape[1]} × {D.shape[0]} at {a.sdf_ppm:g} px/mm (clip ±{a.sdf_clip_mm:g} mm): {nseg} boundary segments ({ncand} candidates); ink texels "
              f"{100 * (D < 0).mean():.2f} %, min {D.min():.3f} max {D.max():.3f} mm; |∇d| median {np.median(g[inner]):.4f} → {a.out}/sdf.f32")
        if a.check:
            r = (tex < 128); d4 = D[1::2, 1::2] if a.sdf_ppm == 8 else None       # the 8 px/mm texel (2u+1, 2v+1) centres sit at (u + ¾)/4 mm: ¼ texel off
            if d4 is not None:
                sgn = (d4 < 0) != r; far = np.abs(d4) > 0.5 / PPM_TEX
                print(f"check: sdf sign vs the raster: {sgn.sum()} texels differ, {(sgn & far).sum()} of them farther than half a raster texel from an edge → {'OK' if (sgn & far).sum() < 200 else 'MISMATCH'}")
                layout["check"]["sdf_sign_mismatch_far"] = int((sgn & far).sum())
    W, H = (int(v) for v in a.image_size.split(","))
    if a.truth:
        P = json.load(open(a.truth)); sets = []
        for v in P["views"]:
            sets.append((v["name"], [(float(c["px_colmap"][0]), float(c["px_colmap"][1]), int(c["id"])) for c in v["corners"]]))
        write_feature_file(os.path.join(a.out, "dataset.bin"), W, H, sets)
        layout["dataset"] = {"source": os.path.abspath(a.truth), "convention": "colmap (the true projections)", "images": len(sets), "features": sum(len(f) for _, f in sets), "image_size": [W, H]}
        print(f"dataset.bin: {len(sets)} views, {sum(len(f) for _, f in sets)} true star projections (from {a.truth})")
    if a.detections or a.merge:
        if a.merge:
            dsets, dg = [], None
            for d in sorted(os.listdir(a.merge)):
                f = os.path.join(a.merge, d, "dataset.bin")
                if not os.path.isfile(f) or "_w" in d: continue                  # (the retries with another window sit in <V>_w<N>/)
                (DW, DH), s, g = read_dataset_bin(f); assert (DW, DH) == (W, H), (DW, DH); dsets += s; dg = dg or g
            src = a.merge
        else:
            (DW, DH), dsets, dg = read_dataset_bin(a.detections); assert (DW, DH) == (W, H), (DW, DH); src = a.detections
        write_feature_file(os.path.join(a.out, "dataset.bin"), W, H, dsets, dg)
        layout["dataset"] = {"source": os.path.abspath(src), "convention": "colmap (Schöps's detector)", "images": len(dsets), "features": sum(len(f) for _, f in dsets), "image_size": [W, H]}
        print(f"dataset.bin: {len(dsets)} views, {sum(len(f) for _, f in dsets)} detected stars (from {src}; lattice table {'kept' if dg else 'MISSING'})")
    json.dump(layout, open(os.path.join(a.out, "board.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
