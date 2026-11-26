# The no-global-state rule (owner, 2026-11-14: "These global constants should not exist"): no module-level mutable container configures
# or records a run. A run's state — the LM's final-iteration mark, the scrubbed Jacobian entries, the device-call counts — lives on the
# calibration's data (`CalibrationData.run`), so two calibrations in one process never see each other's. The allow-list holds the
# process resources that are no run's state, each with its reason; it only ever shrinks.

@testitem "quality: no module-level run state" begin
    using Test, RadiometricCamera
    ALLOWED = Dict(
        "_DEVICE_POOL" => "the device worker tasks (one device queue each), a process resource reused by every run",
        "_DEVICE_POOL_LOCK" => "guards _DEVICE_POOL",
        "_DEVICE_COPIES" => "a weak-keyed cache of device copies of host arrays, keyed by the array object; no run reads another's entry",
        "_DEVICE_CACHE_LOCK" => "guards the device caches' lazy entries",
        "_CAPACITY" => "the device's measured kernel capacity per functor type, a property of the device, not of a run")
    root = joinpath(pkgdir(RadiometricCamera), "src")
    re = r"^const ([A-Za-z_0-9]+)\s*(?:::[^=]+)?=\s*(?:Dict|Base\.IdSet|IdSet|Set|Ref|WeakKeyDict|IdDict|ReentrantLock|Channel|Vector|[A-Za-z_][A-Za-z_0-9]*\[\])"
    found = [(relpath(f, root), i, m[1]) for (d, _, fs) in walkdir(root) for n in fs if endswith(n, ".jl")
             for f in (joinpath(d, n),) for (i, l) in enumerate(eachline(f)) for m in (match(re, l),) if m !== nothing]
    bad = filter(h -> !haskey(ALLOWED, h[3]), found)
    @test isempty(bad)
    isempty(bad) || foreach(h -> println("module-level mutable state: ", h[1], ":", h[2], " ", h[3]), bad)
    @test all(n -> any(h -> h[3] == n, found), keys(ALLOWED))        # an allow-list entry that matches nothing is stale
end

@testitem "a state's run record: its own final-iteration mark, scrubbed counts and device calls" begin
    using Test, RadiometricCamera
    const RC = RadiometricCamera
    @test fieldtype(RC.CalibrationData, :run) === RC._LmRun
    a, b = RC._LmRun(), RC._LmRun()
    a.final_iteration = true; a.scrubbed[:pose] = 3; a.device_calls[:flux] += 1
    @test !b.final_iteration && isempty(b.scrubbed) && b.device_calls == Dict(:flux => 0, :jac => 0)   # never shared between states
    RC._reset_scrubbed!(a); @test isempty(a.scrubbed)
    J = [1.0 NaN; Inf 2.0]
    RC._scrub_entries!(J, 1, :lens, b)
    @test J == [1.0 0.0; 0.0 2.0] && b.scrubbed == Dict(:lens => 2) && isempty(a.scrubbed)
end
