# Real Cameras are Radiometric — project page

The landing page of the CVPR 2027 paper *Real Cameras are Radiometric*, served at
<https://realcamerasareradiometric.github.io>. It is a Documenter site with the
DocumenterLandingPage plugin, in `site/`:

- `site/src/index.md` — the page (hero frontmatter, abstract, BibTeX);
- `site/diagrams.jl` — draws the icon (`icon.svg`, `icon_dark.svg`, `favicon.png`,
  `avatar.png`) into `site/src/assets/landing/` on every build, from the measured
  image circle and vignetting of the fisheye; the files are not committed;
- `site/make.jl` — builds the site and, on CI only, deploys it.

## Build locally

```sh
julia --project=site -e 'using Pkg; Pkg.instantiate()'
julia --project=site site/make.jl
```

Open `site/build/index.html`. `deploydocs` does nothing off CI.

## Deploy

`.github/workflows/site.yml` builds on every push to `main` and deploys to the
`gh-pages` branch of this repository with the workflow's `GITHUB_TOKEN`. In the
repository settings, set Pages to serve the `gh-pages` branch.
