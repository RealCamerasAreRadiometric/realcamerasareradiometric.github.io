# Compare the JSON outputs of a pipeline case with its golden files. Exit code 1 on any difference beyond the mode's tolerance.
#   julia --project=RadiometricCamera experiments/test/pipeline/compare_outputs.jl <golden dir> <output dir>
#
# Two modes, chosen by the recording environment in experiments/test/pipeline/golden/RECORDED.toml:
#   exact      the running Julia has the recorded major.minor AND the active Manifest has the recorded sha256 AND it runs on the recorded
#              device (PIPELINE_DEVICE, gpu | cpu): every numeric leaf
#              must be bit-identical (timings excluded);
#   tolerance  any other environment: the determined quantities within fixed tolerances —
#                poses (poses_*.json, trajectory   rotation ≤ 1e-6° (the angle of R_goldenᵀ R), translation ≤ 1e-6 relative to |t|
#                records)
#                grid-lens nodes (grid.json, the   ≤ 1e-9 relative, ONLY nodes the golden LM moved (grid.json ≠ grid_init.json): a node
#                cameras record's lens.spline.
#                controls.direction)
#                                                  without observations keeps its start and carries no information
#                every other number (lens / intrinsics, residual statistics, costs, metrics, material, heights, self-check
#                vectors)                          ≤ 1e-9 relative
#                residual-vector components        ≤ 1e-9 ABSOLUTE in their unit: selfcheck_vectors.json `views[].dx[]`, `dy[]` (a
#                                                  detection minus the model's reprojection, px) and `du[]`, `dv[]` (the same on the
#                                                  board, board-chart units). Each is a difference of O(1e3) px (O(1)) coordinates,
#                                                  so its rounding is absolute in that unit and a relative tolerance at 0 is meaningless
#              relative = |Δ| / max(|golden|, 1e-12); integer-valued counts must be equal. The report lists the max differences.
# Strings (paths, labels), timings (`seconds`, `time`, `stopped`) and the execution setting `window_width` are not compared.
# `compare_case(gdir, odir; exact)` is the comparison: the package's unit tests include this file and run it in tolerance mode on every
# golden case. Run as a script, it reads the mode from RECORDED.toml.
import Printf
import VisualGeometryToolkitIO as VgtkIO
const SKIP = Set(["seconds", "time", "stopped", "window_width"])   # window_width: the LM's concurrent blocks, an execution setting
function leaves!(out, x, key = "")
    if x isa AbstractDict
        for (k, v) in pairs(x); String(k) in SKIP || leaves!(out, v, key * "." * String(k)); end
    elseif x isa AbstractVector
        for (i, v) in enumerate(x); leaves!(out, v, key * "[" * string(i) * "]"); end
    elseif x isa Number && !(x isa Bool)
        out[key] = Float64(x)
    end
    return out
end
readjson(p) = VgtkIO.read_record(p)
# The fields that are differences of coordinates (absolute tolerance), by file and key path (from the writer, _write_selfcheck in
# src/diagnostics/selfcheck.jl: `dx`, `dy` = residual_vectors in px; `du`, `dv` = board_hits in board-chart units).
const ABSOLUTE_FIELDS = Dict("selfcheck_vectors.json" => r"^\.views\[\d+\]\.(dx|dy|du|dv)\[\d+\]$")
const ABSOLUTE_TOL = 1e-9
relerr(h, g) = abs(h - g) / max(abs(g), 1e-12)
# The rotation of a unit quaternion (w, x, y, z), Hamilton: the trajectory records' `quaternion_wxyz_hamilton`.
function quat_rotation(q)
    w, x, y, z = Float64.(q)
    return [1 - 2(y^2 + z^2) 2(x * y - w * z) 2(x * z + w * y); 2(x * y + w * z) 1 - 2(x^2 + z^2) 2(y * z - w * x); 2(x * z - w * y) 2(y * z + w * x) 1 - 2(x^2 + y^2)]
end

# A grid-lens camera record's control node (vgio/cameras/v6.0, IO 0.10's one lattice record: cameras[c].lens.spline.controls.direction[i],
# a 3-vector; an older record reads as this one, `read_record` runs IO's version steps): a leaf's node key.
const CONTROL_NODE = r"^(\.cameras\[\d+\]\.lens\.spline\.controls\.direction\[\d+\])\[\d+\]$"
node_key(k) = (m = match(CONTROL_NODE, k)) === nothing ? nothing : String(m.captures[1])
# The control nodes the golden LM moved: those whose components differ between the golden grid.json and grid_init.json.
function moved_nodes(G, Gi)
    g = leaves!(Dict{String, Float64}(), G); gi = leaves!(Dict{String, Float64}(), Gi)
    all_nodes = Set(filter(!isnothing, node_key.(collect(keys(g)))))
    isempty(all_nodes) && error("grid.json: no control nodes (cameras[].lens.spline.controls.direction): the camera record's shape changed")
    return Set(node_key(k) for (k, v) in g if node_key(k) !== nothing && !isequal(get(gi, k, NaN), v)), length(all_nodes)
end

function compare_case(gdir, odir; exact::Bool, io::IO = stdout)
    bad = 0; n = 0
    report(msg) = (bad += 1; bad <= 20 && println(io, "    ", msg))
    for f in sort(filter(endswith(".json"), readdir(gdir)))
        o = joinpath(odir, f)
        isfile(o) || (report("MISSING output $f"); continue)
        G = readjson(joinpath(gdir, f)); H = readjson(o)
        if exact
            g = leaves!(Dict{String, Float64}(), G); h = leaves!(Dict{String, Float64}(), H)
            keys(g) == keys(h) || report("$f: different fields")
            for (k, gv) in g
                n += 1
                haskey(h, k) && isequal(h[k], gv) || report("$f$k: $(get(h, k, missing)) ≠ golden $gv (bit-exact)")
            end
            continue
        end
        if startswith(f, "poses_")                                               # a trajectory record: world → camera per frame
            rot = 0.0; trel = 0.0
            length(G.frames) == length(H.frames) || (report("$f: pose count"); continue)
            for (a, b) in zip(G.frames, H.frames)
                n += 1
                Ra = quat_rotation(a.pose.rotation); Rb = quat_rotation(b.pose.rotation)
                D = Ra' * Rb                                                     # the angle of D by atan2(‖vee(skew D)‖, (tr D − 1)/2): acos
                w = [D[3, 2] - D[2, 3], D[1, 3] - D[3, 1], D[2, 1] - D[1, 2]] ./ 2  # of the trace alone resolves nothing below ≈ 1.2e-6°
                θ = rad2deg(atan(sqrt(sum(abs2, w)), (D[1, 1] + D[2, 2] + D[3, 3] - 1) / 2))
                ta = collect(Float64, a.pose.translation); tb = collect(Float64, b.pose.translation)
                dt = sqrt(sum(abs2, tb .- ta)) / max(sqrt(sum(abs2, ta)), 1e-12)
                rot = max(rot, θ); trel = max(trel, dt)
                θ <= 1e-6 || report("$f $(a.image): rotation differs by $(θ)°")
                dt <= 1e-6 || report("$f $(a.image): translation differs by $(dt) relative")
            end
            Printf.@printf(io, "    %-22s max rotation %.3g°, max translation %.3g relative\n", f, rot, trel)
            continue
        end
        g = leaves!(Dict{String, Float64}(), G); h = leaves!(Dict{String, Float64}(), H)
        moved = nothing; nnodes = 0
        if f == "grid.json" && isfile(joinpath(gdir, "grid_init.json"))           # a grid-lens camera record: its control nodes
            moved, nnodes = moved_nodes(G, readjson(joinpath(gdir, "grid_init.json")))
        end
        mrel = 0.0; mabs = 0.0; worst = ""; nabs = 0; mabs_abs = 0.0
        absfield = get(ABSOLUTE_FIELDS, f, nothing)
        for (k, gv) in g
            moved !== nothing && (nk = node_key(k)) !== nothing && !(nk in moved) && continue
            n += 1
            haskey(h, k) || (report("$f$k: missing"); continue)
            hv = h[k]
            if isinteger(gv) && abs(gv) > 1 && isinteger(hv)                     # counts (pixels, iterations, sizes)
                hv == gv || report("$f$k: count $hv ≠ golden $gv")
                continue
            end
            isnan(hv) == isnan(gv) || (report("$f$k: NaN mismatch"); continue)
            isnan(gv) && continue
            if absfield !== nothing && occursin(absfield, k)                    # a coordinate difference: absolute tolerance
                d = abs(hv - gv); nabs += 1; mabs_abs = max(mabs_abs, d)
                d <= ABSOLUTE_TOL || report("$f$k: $hv ≠ golden $gv (absolute $(Printf.@sprintf("%.3g", d)))")
                continue
            end
            r = relerr(hv, gv); r > mrel && (mrel = r; worst = k); mabs = max(mabs, abs(hv - gv))
            r <= 1e-9 || report("$f$k: $hv ≠ golden $gv (relative $(Printf.@sprintf("%.3g", r)))")
        end
        Printf.@printf(io, "    %-22s max relative %.3g (%s), max absolute %.3g%s\n", f, mrel, isempty(worst) ? "-" : worst, mabs,
                       (moved === nothing ? "" : "; $(length(moved)) moved nodes compared, $(nnodes - length(moved)) unmoved skipped") *
                       (nabs == 0 ? "" : Printf.@sprintf("; %d coordinate differences, max absolute %.3g", nabs, mabs_abs)))
    end
    println(io, "    compared $n quantities ($(exact ? "exact" : "tolerance")): ", bad == 0 ? "all match" : "$bad differ")
    return (; bad, n)
end

# Run as a script: the mode from RECORDED.toml. TOML and SHA are imported here only, in their own statement (SHA is not in the package's
# test target, whose unit test includes this file for `compare_case`).
const RUN_AS_SCRIPT = abspath(PROGRAM_FILE) == @__FILE__
RUN_AS_SCRIPT && @eval import TOML, SHA
if RUN_AS_SCRIPT
    gdir, odir = ARGS
    rec = TOML.parsefile(joinpath(dirname(normpath(gdir)), "RECORDED.toml"))
    manifest = Base.project_file_manifest_path(Base.active_project())
    msha = manifest === nothing ? "" : bytes2hex(SHA.sha256(read(manifest)))
    jv = "$(VERSION.major).$(VERSION.minor)"
    dev = get(ENV, "PIPELINE_DEVICE", "cpu"); rdev = get(rec, "device", "cpu")     # the device the run ran on, the goldens' (RECORDED.toml)
    exact = jv == rec["julia_minor"] && msha == rec["manifest_sha256"] && dev == rdev
    println("    mode: ", exact ? "exact (Julia $jv, Manifest sha256 and device $dev as recorded)" :
            "tolerance (recorded Julia $(rec["julia_minor"]), Manifest $(first(rec["manifest_sha256"], 12)), device $rdev; running Julia $jv, Manifest $(first(msha, 12)), device $dev)")
    exit(compare_case(gdir, odir; exact).bad == 0 ? 0 : 1)
end
