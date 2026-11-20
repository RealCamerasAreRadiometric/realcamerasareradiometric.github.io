using Documenter
using RadiometricCamera

# Built locally, by the CI workflow's docs job (.github/workflows/CI.yml of the repository) as a check, and by the Site workflow
# (.github/workflows/site.yml through site/build.sh), which publishes it at https://realcamerasareradiometric.github.io/docs/
# inside the project page's one deploy. This file never deploys: a deploydocs here would overwrite the project page on gh-pages.
# Build with
#     julia +1.13 --project=docs --startup-file=no docs/make.jl [build directory]
# (--startup-file=no: a startup.jl that activates another environment overrides --project); the site lands in docs/build/, or
# in the directory given (site/build.sh gives <site>/docs).
makedocs(;
    build = isempty(ARGS) ? "build" : abspath(ARGS[1]),
    # The repository root, which holds the package in its subdirectory RadiometricCamera/ (Documenter takes the source paths
    # relative to the git root).
    repo = Documenter.Remotes.GitHub("RealCamerasAreRadiometric", "realcamerasareradiometric.github.io"),
    modules = [RadiometricCamera],
    authors = "Ties Harder <TiesHarder@icloud.com>",
    sitename = "RadiometricCamera.jl",
    # api/pipeline.html measures 149.6 KiB (2026-11-19), above the default warning size of 100 KiB and below the hard 200 KiB.
    # Served under the project page at /docs/: canonical names that URL; plain .html page URLs (prettyurls = false) resolve the
    # same from a file:// preview, any static server and GitHub Pages; the edit links point at main's RadiometricCamera/docs/src
    # (Documenter takes the source paths relative to the git root).
    format = Documenter.HTML(;
        canonical = "https://realcamerasareradiometric.github.io/docs/",
        prettyurls = false,
        edit_link = "main",
        size_threshold_warn = 160 * 2^10,
    ),
    # The docs gates: every public docstring is in the manual, the doctests and the @example blocks run, every external link
    # resolves, and nothing is warnonly — a warning fails the build. That every public name HAS a docstring is
    # test/unit/docs_tests.jl's job (the missing-docs check only sees docstrings that exist).
    checkdocs = :public,
    doctest = true,
    linkcheck = true,
    # github.com rate-limits crawlers with 429 (and answers 404 for a repository not yet public).
    linkcheck_ignore = [r"^https://github\.com/"],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "guide/getting-started.md",
        "How to" => [
            "Calibrate a camera" => "guide/calibrating.md",
            "Set up the data and the hardware" => "guide/setup.md",
            "Run the tests and check a change" => "guide/testing.md",
        ],
        "Explanation" => [
            "The radiometric camera model" => "guide/model.md",
        ],
        "API Reference" => [
            "Camera" => "api/camera.md",
            "Target and lighting" => "api/target.md",
            "Calibration" => "api/calibration.md",
            "Pipeline and options" => "api/pipeline.md",
            "API surface" => "api/surface.md",
        ],
    ],
)
