@testitem "DoubleSphere (Core lens contract)" setup = [RCFixtures] begin
    using RadiometricCamera: in_domain, lm_box, _chief_ray
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3, project

    import VisualGeometryToolkitCore as VgtkCore
    _kc(l) = VgtkCore.parameter_vector(VgtkCore.lens(l))      # the lens coefficients (Core's parameter_vector)
    W = H = 100                                               # the image (pixels)
    lens = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 128.8, (51.0, 48.0), W, H)
    for 𝐩 in (SVector(50.0, 50.0), SVector(80.0, 30.0), SVector(5.0, 95.0), SVector(99.0, 50.0))
        d = _chief_ray(lens, 𝐩)
        @test norm(d) ≈ 1
        @test project(lens, d) ≈ 𝐩 atol = 1e-8                     # π ∘ π⁻¹ = id
        @test project(lens, 3.7 * d) ≈ 𝐩 atol = 1e-8                # scale invariance
        @test in_domain(lens, 𝐩)
    end
    pin = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.0, 0.0)), 100.0, (50.0, 50.0), W, H)
    @test _chief_ray(pin, SVector(80.0, 70.0)) ≈ normalize(SVector(0.3, 0.2, 1.0))      # pinhole limit
    # parity with Core's double-sphere lens on the normalised point m = K⁻¹𝐩 (Core's `DoubleSphere(ξ, α)`)
    m_ds = SVector((80.0 - 51.0) / 128.8, (30.0 - 48.0) / 128.8)
    @test VgtkCore.backproject(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), m_ds) ≈ _chief_ray(lens, SVector(80.0, 30.0)) atol = 1e-12
    # Core's camera: the posed camera of the model back-projects the same ray
    ext = VgtkCore.EuclideanMap(VgtkCore.Rotations.RotMatrix{3, Float64}(I), SVector(0.0, 0.0, 0.0))
    pc = VgtkCore.PosedCamera(VgtkCore.model(lens), ext)
    @test VgtkCore.backproject(pc.model, SVector(80.0, 30.0)) ≈ _chief_ray(lens, SVector(80.0, 30.0))
    # Dual numbers flow through the parameters (K, ξ, α), which lie in the LM's box
    ld = camera_of(VgtkCore.Lens(VgtkCore.DoubleSphere(0.0, 0.0)), ForwardDiff.Dual.(vcat(pinhole_θ(lens), _kc(lens)), 1.0), W, H)
    @test _chief_ray(ld, SVector(60.0, 60.0)) isa SVector{3, <:ForwardDiff.Dual}
    m = VgtkCore.model(lens)
    lo, hi = VgtkCore.parameter_box(VgtkCore.ParameterManifold(m; box = lm_box(m, W, H))); @test all(lo .<= VgtkCore.parameter_vector(m) .<= hi)

    # The domain is the fold of the radial profile (Core's double sphere). Reference, independent of Core and of RC: the image radius of
    # the unit ray at view angle θ, r(θ) = sin θ / (α d₂ + (1 − α)(ξ + cos θ)), d₂ = √(sin²θ + (ξ + cos θ)²), scanned densely from the axis
    # to its first maximum or pole: a normalised point is imaged iff its radius is below that maximum (the profile is one-to-one up to it,
    # and past it a second ray would share the pixel). Lattice points within 1e-3 (relative) of the edge are not classified. The lifted
    # ray of an imaged point is the closed form of eq. 46-51 written out here, and projecting it returns the point.
    @testset "DoubleSphere domain = the fold of the radial profile (brute scan)" begin
        function r_end(ξ, α; n = 400_000)
            best = 0.0
            for i in 1:n
                θ = π * i / n; zs = ξ + cos(θ); den = α * sqrt(sin(θ)^2 + zs^2) + (1 - α) * zs
                den > 0 || return best
                r = sin(θ) / den
                r > best || return best
                best = r
            end
            return best
        end
        gs(x) = sqrt(max(x, zero(x)))
        function lift_ref(l, m)
            r² = sum(abs2, m)
            m_z = (1 - _kc(l)[2]^2 * r²) / (_kc(l)[2] * gs(1 - (2_kc(l)[2] - 1) * r²) + 1 - _kc(l)[2])
            fr = (m_z * _kc(l)[1] + gs(m_z^2 + (1 - _kc(l)[1]^2) * r²)) / (m_z^2 + r²)
            return SVector(fr * m[1], fr * m[2], fr * m_z - _kc(l)[1])
        end
        lenses = [cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 128.8, (51.0, 48.0), W, H), cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.15, 0.5)), 125.0, (50.0, 50.0), W, H),
                  cam(VgtkCore.Lens(VgtkCore.DoubleSphere(-0.6, 0.6)), 50.0, (50.0, 50.0), W, H), cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.5, 0.7)), 30.0, (50.0, 50.0), W, H),
                  cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.9, 0.6)), 30.0, (50.0, 50.0), W, H), cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.0, 0.0)), 30.0, (50.0, 50.0), W, H)]
        for l in lenses
            R = r_end(_kc(l)[1], _kc(l)[2])
            ms = [SVector(x, y) for x in range(-3.0, 3.0, length = 81), y in range(-3.0, 3.0, length = 81)]
            ms = filter(m -> abs(norm(m) - R) > 1e-3 * R, ms)
            θl = pinhole_θ(l)
            𝐩s = [SVector(θl[1] * m[1] + θl[3], θl[2] * m[2] + θl[4]) for m in ms]
            @test all(in_domain(l, 𝐩) == (norm(m) < R) for (m, 𝐩) in zip(ms, 𝐩s))
            inside = [(m, 𝐩) for (m, 𝐩) in zip(ms, 𝐩s) if norm(m) < R]
            @test maximum(norm(_chief_ray(l, 𝐩) - normalize(lift_ref(l, m))) for (m, 𝐩) in inside) < 1e-12
            @test maximum(norm(project(l, _chief_ray(l, 𝐩)) - 𝐩) for (m, 𝐩) in inside) < 1e-8
        end
        l32 = VgtkCore.convert_eltype(Float32, lenses[1])
        @test _chief_ray(l32, SVector{2, Float32}(80.0, 30.0)) isa SVector{3, Float32}
        @test norm(_chief_ray(l32, SVector{2, Float32}(80.0, 30.0)) - _chief_ray(lenses[1], SVector(80.0, 30.0))) < 1e-6
    end

    # The double sphere is omnidirectional: inside the fold it images rays past 90° (z < 0), and they round-trip through `project`.
    let l = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.9, 0.6)), 30.0, (50.0, 50.0), W, H),
        S = [SVector(x, y) for x in range(0.0, 100.0, length = 101), y in range(0.0, 100.0, length = 101)]
        back = [𝐩 for 𝐩 in S if in_domain(l, 𝐩) && _chief_ray(l, 𝐩)[3] < 0]
        @test length(back) > 100
        @test maximum(norm(project(l, _chief_ray(l, 𝐩)) - 𝐩) for 𝐩 in back) < 1e-10
    end
end
