# The Schöps front end's detections (`dataset_features`): a run's import `detections.json` (VisualGeometryToolkitIO's observations
# kind, `TargetObservations`, in COLMAP px) read back per image, in file order, with the reference's 0-based feature ids.

@testitem "Schöps detections: dataset_features reads a detections.json (IO's observations) per image, 0-based ids" setup = [RCFixtures] begin
    using StaticArrays
    F = VgtkCore.PixelFrame{:colmap}
    xy1 = [SVector(10.5, 20.25), SVector(30.0, 40.75), SVector(55.5, 5.5)]
    xy2 = [SVector(1.0, 2.0), SVector(3.5, 4.5)]
    frames = Dict(1 => Dict(:board => VgtkCore.correspondences([1, 4, 7], xy1)), 2 => Dict(:board => VgtkCore.correspondences([2, 3], xy2)))
    obs = VgtkCore.TargetObservations{F}(frames; images = Dict(1 => "V01.png", 2 => "V02.png"), sensors = "cam", targets = [repeat("ab", 32)])   # the board's identity (a SHA-256 digest; any)
    mktempdir() do dir
        path = joinpath(dir, "detections.json")
        VgtkIO.save(path, obs)
        sets = RC.dataset_features(path)
        @test [s[1] for s in sets] == ["V01.png", "V02.png"]                      # by frame index, in order
        @test sets[1][2] == [(10.5, 20.25, 0), (30.0, 40.75, 3), (55.5, 5.5, 6)]   # (x, y, 0-based id): the point index − 1
        @test sets[2][2] == [(1.0, 2.0, 1), (3.5, 4.5, 2)]
        # two targets' detections: their 0-based ids would collide (the texture lookup `tex[id]`), so the file is refused, naming both
        # (review finding 6), never flattened in the Dict's order
        two = Dict(1 => Dict(:board => VgtkCore.correspondences([1, 4], xy1[1:2]), :other => VgtkCore.correspondences([1], xy1[3:3])))
        obs2 = VgtkCore.TargetObservations{F}(two; images = Dict(1 => "V01.png"), sensors = "cam", targets = [repeat("ab", 32), repeat("cd", 32)])
        p2 = joinpath(dir, "two", "detections.json"); mkpath(dirname(p2))
        VgtkIO.save(p2, obs2)
        @test_throws "the detections of 2 targets (board, other)" RC.dataset_features(p2)
    end
end

@testitem "Schöps detections: a views list naming a featureless photograph is refused by name (require_featured_views)" setup = [RCFixtures] begin
    # a view the front end gave no feature has no start pose (it was a MethodError in _fit_page_pose on an empty ray list): the case's
    # views list must not name it — refused with the photograph and the reason, never dropped inside the solver
    counts = ["DSC06033.png" => 41, "DSC06034.png" => 0, "DSC06129.png" => 0, "DSC06130.png" => 12]
    run = "schoeps/si30"
    @test RC.require_featured_views(counts, ["DSC06033.png", "DSC06130.png"], run) === nothing
    e = try RC.require_featured_views(counts, ["DSC06033.png", "DSC06034.png"], run); nothing catch err; err end
    @test e isa ArgumentError && occursin("DSC06034.png", e.msg) && occursin("0 front-end features", e.msg) && occursin(run, e.msg) &&
          occursin("no start pose", e.msg) && !occursin("DSC06129", e.msg)
    e = try RC.require_featured_views(counts, String[], run); nothing catch err; err end   # every view of the run: both named
    @test e isa ArgumentError && occursin("DSC06034.png, DSC06129.png", e.msg)
    e = try RC.require_featured_views(counts, ["DSC06033.png", "DSC09999.png"], run); nothing catch err; err end   # not in the run at all
    @test e isa ArgumentError && occursin("DSC09999.png", e.msg) && occursin("no detections", e.msg)
end
