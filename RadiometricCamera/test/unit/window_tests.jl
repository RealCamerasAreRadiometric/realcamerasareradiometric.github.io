# The LM's blocks concurrently (`_windowed`, per_view.jl): a request for block i starts the blocks after it at the same trial on pooled
# workers, as many as `_window_width` derives from the device and the problem (owner ruling 2026-11-13: an execution setting, no
# constant, no config key). Its values must not depend on the width: the Gauss–Newton blocks and the costs of every tile of a grid
# problem, at two widths and without the window, bit for bit.

@testitem "the concurrent blocks: the same Gauss–Newton blocks at any window width" setup = [RCFixtures, EdgeScene] begin
    using RadiometricCamera: build_problem_grid
    s = edge_state(edge_s, targets_e)
    prob, idx = build_problem_grid(s; patch = (5, 5), height_degree = 1)
    tiles = RC.tile_costs(prob)
    blocks() = [RC.NLLSsolver.computecostgradhess(RC.static(1023), t, RC.NLLSsolver.getvars(t, prob.variables)...) for t in tiles]
    function at_width(w)
        RC._register_grid_problem!(s, prob, idx)
        w === nothing || RC._register_window!(s, prob, T -> T <: RC.PerTileResidual; width = w)
        try
            return blocks()
        finally
            RC._unregister_window!(s)
            RC._unregister_grid_problem!(s)
        end
    end
    b0 = at_width(nothing)                                   # no window: one tile at a time
    for w in (1, 3)
        bw = at_width(w)
        @test all(((x, y),) -> x[1] == y[1] && x[2] == y[2] && x[3] == y[3], zip(b0, bw))
    end
    @test RC._device_entry(s, :window_width) == 3            # the width a window ran at stays for the run's provenance
    # the CPU backend's width: the thread count, within 1 … the number of blocks
    @test RC._window_width(s, tiles) == clamp(Threads.nthreads(), 1, length(tiles))
    # the solve's state is the registered problem's, released with it: its blocks' device Jacobian buffers (one pair per task) and the
    # bytes of its blocks' jobs (the window's memory bound) — no process-global, nothing held by the pooled workers after a solve
    w = RC._with_registered(s, prob, idx) do
        blocks()
        RC._device_entry(s, :window)
    end
    @test w.block_bytes[] > 0 && isempty(w.buffers) && RC._device_entry(s, :window) === nothing && RC._device_entry(s, :grid) === nothing
    @test !isdefined(RC, :_BLOCK_BYTES)
    @test_throws ErrorException RC._with_registered(() -> error("a failing solve"), s, prob, idx)
    @test RC._device_entry(s, :window) === nothing && RC._device_entry(s, :grid) === nothing      # unregistered whatever the solve does
end
