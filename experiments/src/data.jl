# =============================================================================
# src/data.jl — module DataTiers: the tiers of the project's published data (configs/data_artifacts.toml), VisualGeometryToolkitEval
# DataArtifacts `tree` entries with a `filter`: every file below `dir` (relative to its mount root, data_root()) whose path relative
# to the mount root matches one of the `filter` globs and none of the `!`-prefixed ones (Eval's glob rules). The tiers of a catalog:
# `rc_raw_photos_<date>`, `rc_synthetic_<date>`, `rc_baselines_<date>`, `rc_calibration_<date>`; the configs' inputs name the tier
# without its date (`tier_name`), the catalog's entry is that tier's current version.
# =============================================================================
module DataTiers

using VisualGeometryToolkitEval.DataArtifacts: DataCatalog
using TOML: TOML

export tier_name, tier_entries, unpacked_entries, is_unpacked, is_cache_dir

"The tier an artifact name versions: `rc_synthetic_20261029` → `rc_synthetic`."
tier_name(artifact::AbstractString) = replace(artifact, r"_[0-9]{8}$" => "")

"The catalog's tiers: `tier name => artifact name` (one current version each; two versions of one tier are an error)."
function tier_entries(cat::DataCatalog)
    out = Dict{String,String}()
    for n in keys(cat.artifacts)
        t = tier_name(n)
        haskey(out, t) && error("the catalog holds two versions of tier $t: $(out[t]) and $n")
        out[t] = n
    end
    return out
end

"""The catalog's `[[unpacked.<tier>]]` entries (files of data_root() that no packed version of the tier holds yet; the header of
configs/data_artifacts.toml): `tier => [(; paths, configs, where, provenance)]`. `configs` is empty for files that no config of
record reads yet."""
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
