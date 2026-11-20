# ---------------------------------------------------------------------------
# Precision conversion of the pipeline containers (Float64 host state → Float32
# device kernels and back): VisualGeometryToolkitCore's `convert_eltype(T, x)` on the package's types — element types only,
# no re-solves, no re-sampling; the TPS coefficients, knots etc. are converted
# verbatim. The lens methods live next to their lens types.
# ---------------------------------------------------------------------------

# (a focus surface's, the quadrature rules' and a material's conversions are VisualRadiometryToolkitCore's, with the types)
VgtkCore.convert_eltype(::Type{T}, v::View) where {T} =
    View(_pose_eltype(T, v.pose), VgtkCore.convert_eltype(T, v.geometric), VgtkCore.convert_eltype(T, v.focus), T(v.r))

VgtkCore.convert_eltype(::Type{T}, st::RenderSettings) where {T} =
    RenderSettings(VgtkCore.convert_eltype(T, st.sensor), VgtkCore.convert_eltype(T, st.aperture), st.sampler)

function VgtkCore.convert_eltype(::Type{T}, b::PrintedTarget) where {T}
    sd = b.sdf === nothing ? nothing : T.(b.sdf)
    return PrintedTarget(b.target, VgtkCore.Quad{T}(b.quad), SVector{2, T}(b.size), T(b.facing), T.(b.texture), sd, SVector{2, T}(b.page_mm),
                         VgtkCore.convert_eltype(T, b.warping))
end
VgtkCore.convert_eltype(::Type{T}, s::Scene) where {T} = Scene(VgtkCore.convert_eltype(T, s.board), s.W, s.H, VgtkCore.convert_eltype(T, s.surface))
