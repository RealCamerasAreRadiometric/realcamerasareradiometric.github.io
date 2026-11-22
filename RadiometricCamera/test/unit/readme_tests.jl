# readme_tests.jl — the README's example runs verbatim, in order, in one module, in a child process whose environment is built from
# the install line alone (this package by path, the rest from the registries), and the install line names exactly the non-stdlib
# packages the example loads; the landing page (docs/src/index.md) shows the same install line and the same example. The stack's
# the stack's README-test template (VisualGeometryToolkitCore's test/readme_tests.jl is its pilot instance).
#
# Conventions the README follows for this to work: the install line is a ```julia-repl block with ONE `pkg> add A B C` line; a
# block that cannot run (the reader's own data, a GPU) starts with `# Illustrative` and is neither run nor compared; no placeholder is left
# undefined.

@testitem "README: the install line names what the examples load; the examples run in a fresh environment" begin
    using Test, RadiometricCamera
    using VisualGeometryToolkitCore: julia_blocks         # Core's README helper (its Test extension): the fenced julia blocks of a page
    root = pkgdir(RadiometricCamera)
    illustrative(block) = startswith(block, "# Illustrative")
    readme = read(joinpath(root, "README.md"), String)
    index = joinpath(root, "docs", "src", "index.md")
    example = filter(!illustrative, julia_blocks(joinpath(root, "README.md")))
    landing = filter(!illustrative, julia_blocks(index))
    @test length(example) == 1                       # pin the counts
    @test landing == example                         # one example, shown twice
    # the project page (site/src/index.md of the repository, beside the package; absent from an installed package) shows it too
    site = joinpath(dirname(root), "site", "src", "index.md")
    isfile(site) && @test filter(!illustrative, julia_blocks(site)) == example

    install = [m[1] for m in eachmatch(r"^pkg> add (.+)$"m, readme)]
    @test length(install) == 1
    # a package added from its repository's subdirectory is spelled `<url>:<subdirectory>` (Pkg's REPL), its name the subdirectory's
    pkgname(spec) = occursin("://", spec) ? String(last(split(spec, ':'))) : String(spec)
    packages = pkgname.(split(only(install)))
    loaded = Set{String}()
    for code in example, m in eachmatch(r"^(?:using|import)\s+([^:\n]+)"m, code)
        for part in split(m[1], ',')
            push!(loaded, first(split(strip(part), r"[.\s]")))
        end
    end
    stdlib(name) = isdir(joinpath(Sys.STDLIB, name))
    @test Set(filter(!stdlib, loaded)) == Set(packages)
    @test occursin("pkg> add $(only(install))", read(index, String))

    env = mktempdir()
    write(joinpath(env, "readme_example.jl"), join(example, "\n"))
    registered = filter(!=("RadiometricCamera"), String.(packages))
    script = joinpath(env, "run.jl")
    write(script, """
    using Pkg
    Pkg.develop(path = $(repr(root)); io = devnull)
    isempty($(repr(registered))) || Pkg.add($(repr(registered)); io = devnull)
    readme = Module(:ReadmeSandbox)
    Base.include(readme, $(repr(joinpath(env, "readme_example.jl"))))
    using Test
    @testset "README example" begin
        @test count(readme.mask) == prod(Tuple(readme.sz)) == 76800  # "76800 = 320 × 240: the board covers every pixel"
        @test isapprox(readme.Φ, readme.img[120, 160, :]; rtol = 1e-4)   # "true: the render is Φ_c(𝐮) of each pixel" (row v, column u), to Float32 on a GPU
        @test all(>(0), readme.img[:, :, 1][readme.mask])
    end
    """)
    # Base.julia_cmd() carries the test run's flags (--check-bounds, coverage). The load path is the project and the standard library
    # only: no default environment can supply a package the install line does not add. JULIA_PKG_OFFLINE from the child's start (the
    # stack's pattern, VisualGeometryToolkitEval test/paper_project_template_tests.jl:66): the registries are the depot's and the resolve
    # takes the versions installed there, so the suite makes no network call and updates no registry, Pkg's first load included.
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$env $script`
    proc = withenv("JULIA_LOAD_PATH" => "@:@stdlib", "JULIA_PKG_OFFLINE" => "true", "JULIA_PROJECT" => nothing) do
        run(ignorestatus(cmd))
    end
    @test proc.exitcode == 0
end
