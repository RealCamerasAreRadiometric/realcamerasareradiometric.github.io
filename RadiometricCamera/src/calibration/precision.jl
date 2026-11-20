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

# (a textured target's conversion, with its state's, is VisualRadiometryToolkitCore's, with the type)
