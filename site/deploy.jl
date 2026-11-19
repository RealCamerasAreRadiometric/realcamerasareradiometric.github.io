using Documenter

# The one deploy of https://realcamerasareradiometric.github.io: the tree site/build.sh assembled (the landing page,
# the package docs at docs/, the paper at paper.pdf), from ARGS[1] (default site/build/), to the ROOT of the gh-pages
# branch of this same repository, unversioned (versions = nothing replaces the whole branch content on every deploy:
# nothing else may deploy to gh-pages). Run by CI (.github/workflows/site.yml) with the workflow's GITHUB_TOKEN; off CI,
# deploydocs does nothing.
deploydocs(;
    root=@__DIR__,
    target=isempty(ARGS) ? "build" : abspath(ARGS[1]),
    repo="github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io.git",
    branch="gh-pages",
    devbranch="main",
    versions=nothing,
)
