# The manuscript beside the experiments (../paper, main.tex at its top, built there by its Makefile): its layout and the data
# policy (no data under the manuscript's tree).
@testitem "the manuscript" default_imports = false begin
    using Test
    using DrWatson

    paper = projectdir("..", "paper")

    @testset "main.tex and the generated macro file" begin
        main = joinpath(paper, "main.tex")
        @test isfile(main)
        @test occursin("\\input{latex/numbers}", read(main, String))   # built from ../paper: every path relative to it
        @test isfile(joinpath(paper, "latex", "numbers.tex"))
    end

    @testset "no data under the manuscript" begin
        banned = [r"(^|/)data/", r"(^|/)results/", r"\.jld2$"i, r"\.ARW$"i, r"\.CR2$"i]
        hits = String[]
        for (root, _, files) in walkdir(paper), f in files
            rel = relpath(joinpath(root, f), paper)
            startswith(rel, "build") && continue                      # the local LaTeX build (ignored by git)
            any(re -> occursin(re, rel), banned) && push!(hits, rel)
        end
        foreach(h -> @info("data-policy hit under ../paper/: " * h), hits)
        @test isempty(hits)
    end
end
