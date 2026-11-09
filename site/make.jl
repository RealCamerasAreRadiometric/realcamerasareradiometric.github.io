using Documenter
using DocumenterLandingPage   # LandingPage(): the VitePress-style home page (index.md frontmatter)

# The icon is drawn from the measured image circle and vignetting of the
# fisheye (site/diagrams.jl): regenerated on every build, never committed.
include(joinpath(@__DIR__, "diagrams.jl"))
write_icons(joinpath(@__DIR__, "src", "assets", "landing"))

makedocs(;
    authors="Authors to be announced",
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

# The site is the RealCamerasAreRadiometric organization page
# (https://realcamerasareradiometric.github.io): unversioned, on the gh-pages
# branch of this same repository, deployed by CI (.github/workflows/site.yml)
# with the workflow's GITHUB_TOKEN. Off CI, deploydocs does nothing.
deploydocs(;
    repo="github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io.git",
    branch="gh-pages",
    devbranch="main",
    versions=nothing,
)
