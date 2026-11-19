# ---------------------------------------------------------------------------
# Extrinsics Θ_e. A view's pose is VisualGeometryToolkitCore's camera → world `EuclideanMap` (R = R_cw, t = the camera centre 𝐜; the
# unlabeled plain form, isbits, so it travels into the kernels as it is); the world → camera extrinsics a `PosedCamera` holds are its
# `inv`. The LM variable of a pose is Core's left-perturbation chart `LeftRigidManifold6` on that map (an NLLSsolver variable through
# Core's manifold adapter): R ← exp([ω]ₓ) R, t ← exp([ω]ₓ) t + ρ, re-anchored after every accepted step (Core's `recenter`, which
# re-projects R onto SO(3)). The rotation error is `Rotations.rotation_angle`.
# ---------------------------------------------------------------------------


"""The pose in the element type `T` (Float32 for a device kernel, a dual type for a Jacobian evaluation), with unlabeled frames — the isbits
form the kernels take: Core's `convert_eltype` (any `T <: Real`: a `RotMatrix` rotation and a plain translation, the camera centre in
metres) and `unlabel`."""
@inline _pose_eltype(::Type{T}, p::EuclideanMap) where {T} = VgtkCore.unlabel(VgtkCore.convert_eltype(T, p))

"""
    _seeded_pose(pose, cst, ω, ρ) -> EuclideanMap

The pose at the chart coordinates `(ω, ρ)` (dual seeds of a Jacobian evaluation) of Core's `LeftRigidManifold6` anchored at `pose`, the
anchor converted by `cst` (a constant dual of the evaluation's type, so a Float32 evaluation stays Float32).
"""
@inline function _seeded_pose(pose::EuclideanMap, cst, ω::StaticVector{3}, ρ::StaticVector{3})
    ch = VgtkCore.LeftRigidManifold6(SMatrix{3, 3}(map(cst, SMatrix{3, 3}(pose.R))), SVector{3}(map(cst, SVector{3}(pose.t))))
    return VgtkCore.retract(ch, vcat(SVector{3}(ω), SVector{3}(ρ)))
end
