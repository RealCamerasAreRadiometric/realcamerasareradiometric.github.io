# =============================================================================
# src/data.jl — module DataTiers: the tiers of the project's published data (configs/data_artifacts.toml), VisualGeometryToolkitEval
# DataArtifacts `tree` entries with a `filter`: every file below `dir` (relative to its mount root, data_root()) whose path relative
# to the mount root matches one of the `filter` globs and none of the `!`-prefixed ones (Eval's glob rules). An entry is one tarball,
# `<tarball>_<yyyymmdd>`; its `tier` is the tier name the configs' inputs use (`rc_raw_photos`, `rc_baselines`, …: an input's tier
# keys its runs), so one tier is held by several entries. The retired entries (`retired = true`) hold no tier of record. And the
# privacy rule: the photographs no tier and no record may ever hold.
# =============================================================================
module DataTiers

using VisualGeometryToolkitEval.DataArtifacts: DataCatalog
using TOML: TOML

export tier_of, tier_entries, unpacked_entries, is_unpacked, is_cache_dir, PRIVATE_PHOTOGRAPHS, private_photographs

"The tier an entry's files' inputs name (its `tier`); a retired entry's is its name without the pack date."
tier_of(cat::DataCatalog, entry::AbstractString) =
    String(get(cat.artifacts[entry], "tier", replace(entry, r"_[0-9]{8}$" => "")))

"The tiers of record: `tier => [entry, …]` (sorted), over the entries that are not retired."
function tier_entries(cat::DataCatalog)
    out = Dict{String,Vector{String}}()
    for (n, a) in cat.artifacts
        get(a, "retired", false) === true && continue
        push!(get!(out, tier_of(cat, n), String[]), n)
    end
    foreach(sort!, values(out))
    return out
end

"""The photographs that are never published (owner 2026-11-21: the 14 mm Siemens-star session of 2026-11-03, DSC06045 and
DSC06052–DSC06055, show another person): no tier, tarball or record may hold one, under any directory, prefix or extension."""
const PRIVATE_PHOTOGRAPHS = ("DSC06045", "DSC06052", "DSC06053", "DSC06054", "DSC06055")

"The paths of `paths` whose file name names a photograph of [`PRIVATE_PHOTOGRAPHS`](@ref) (case-insensitive, any prefix or extension)."
private_photographs(paths) = [String(p) for p in paths if any(x -> occursin(x, uppercase(basename(String(p)))), PRIVATE_PHOTOGRAPHS)]

"""The catalog's `[[unpacked.<tier>]]` entries (files of the author's folders that no config reads and no tier holds; the catalog's
header): `tier => [(; paths, configs, where, provenance)]`."""
function unpacked_entries(catalog_toml::AbstractString)
    u = get(TOML.parsefile(catalog_toml), "unpacked", Dict{String,Any}())
    return Dict(t => [(; paths = String.(e["paths"]), configs = String.(e["configs"]), where = String(e["where"]),
                       provenance = String(e["provenance"])) for e in es] for (t, es) in u)
end

"""Whether the data_root()-relative path `p` lies at or below a path of `unpacked` (`unpacked_entries`): of `tier`'s entries, or
of any tier's when `tier === nothing`."""
function is_unpacked(unpacked, tier, p::AbstractString)
    under(q) = p == q || startswith(p, q * "/")
    tiers = tier === nothing ? keys(unpacked) : (tier,)
    return any(t -> haskey(unpacked, t) && any(e -> any(under, e.paths), unpacked[t]), tiers)
end

"""Whether the directory `dir` is a cache by the Cache Directory Tagging Standard (https://bford.info/cachedir/): it holds a
`CACHEDIR.TAG` that begins with the standard's signature — as the Hugging Face CLI's `.cache/huggingface/` of a `--local-dir`
download does. A cache is the downloading tool's state, never data, and lies in no tier."""
is_cache_dir(dir::AbstractString) = (f = joinpath(dir, "CACHEDIR.TAG"); isfile(f) &&
    startswith(read(f, String), "Signature: 8a477f597d28d172789f06886806bc55"))

end # module DataTiers
