@testitem "the page's ±y choice of a rendered or resumed state holds (page_flip)" begin
    const RC = RadiometricCamera
    opts(; render_state = "", resume = "") = RC.PipelineOptions(; io = RC.IOOptions(; render_state), resume = RC.ResumeOptions(; state = resume))
    never(path) = error("no state should be read for $path")
    # no state to render or resume: nothing is read, both directions are scored
    @test RC._state_page_flip(opts(), never) === nothing
    # the state's choice is returned as a Float64 sign; io.render_state before resume.state
    @test RC._state_page_flip(opts(; render_state = "r.json"), p -> (; page_flip = -1.0)) === -1.0
    @test RC._state_page_flip(opts(; resume = "s.json"), p -> (; page_flip = 1)) === 1.0
    read = String[]
    @test RC._state_page_flip(opts(; render_state = "r.json", resume = "s.json"), p -> (push!(read, p); (; page_flip = -1.0))) === -1.0
    @test read == ["r.json"]
    # a state that records none (written before the field existed): as before (both directions scored), with a warning. The round
    # trip of the field through a state file is records_tests.jl's.
    @test (@test_logs (:warn, r"carries no page_flip") RC._state_page_flip(opts(; resume = "old.json"), p -> (; page_flip = nothing))) === nothing
end
