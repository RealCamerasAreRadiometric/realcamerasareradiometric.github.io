using Documenter
using DocumenterLandingPage   # LandingPage(): the VitePress-style home page (index.md frontmatter)

# Builds the landing page only, into ARGS[1] (default site/build/). The whole site — this page, the package
# docs at docs/ and the paper at paper.pdf — is assembled by site/build.sh and deployed by site/deploy.jl
# (.github/workflows/site.yml runs both, so CI and a local preview are one code path).
build = isempty(ARGS) ? "build" : abspath(ARGS[1])

# The icon is drawn from the measured image circle and vignetting of the
# fisheye (site/diagrams.jl): regenerated on every build, never committed.
include(joinpath(@__DIR__, "diagrams.jl"))
write_icons(joinpath(@__DIR__, "src", "assets", "landing"))

makedocs(;
    build,
    authors="Ties Harder, Felix Seegräber, David Nakath, Hendrik Sauer, Kevin Köser, James Pritts",
    sitename="Real Cameras are Radiometric",
    repo=Remotes.GitHub("RealCamerasAreRadiometric", "realcamerasareradiometric.github.io"),
    format=Documenter.HTML(;
        assets=[
            "assets/landing_tagline.css",
            "assets/landing_tagline.js",
            asset("assets/landing/favicon.png"; class=:ico, islocal=true),
        ],
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://realcamerasareradiometric.github.io",
        edit_link="main",
        inventory_version="",   # an unversioned site: no package version to read
    ),
    plugins=[LandingPage()],
    linkcheck=true,
    pages=["Home" => "index.md"],
)
