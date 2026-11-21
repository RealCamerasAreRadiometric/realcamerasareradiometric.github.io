# Real Cameras are Radiometric

The code, the experiments and the manuscript of the paper *Real Cameras are Radiometric*, and its project page, served at
<https://realcamerasareradiometric.github.io>:

- **Code**: this repository, <https://github.com/RealCamerasAreRadiometric/realcamerasareradiometric.github.io>.
- **Docs**: the package documentation, <https://realcamerasareradiometric.github.io/docs/>.
- **Paper**: the paper and its supplement, <https://realcamerasareradiometric.github.io/paper.pdf>; arXiv: [2611.25265](https://arxiv.org/abs/2611.25265).

## Repository layout

| Path | Holds |
|---|---|
| `RadiometricCamera/` | the Julia package RadiometricCamera.jl: radiometric camera calibration by inverse ray tracing ([README](RadiometricCamera/README.md)) |
| `experiments/` | the experiments that reproduce the paper: staged drivers, configs of record, the asset index ([README](experiments/README.md), [REPRODUCE.md](experiments/REPRODUCE.md)); its own Julia project, using the package from `../RadiometricCamera` |
| `paper/` | the manuscript (`main.tex`, paper and supplement); `make` builds `build/main.pdf`, `build/paper.pdf` and `build/supp.pdf` |
| `site/` | the project page (below) |

Install the package (Julia 1.13 or later):

```julia
using Pkg
Pkg.Registry.add([Pkg.RegistrySpec(name = "General"), Pkg.RegistrySpec(url = "https://github.com/prittjam/CauRegistry.jl")])   # once: the package's registry, beside General
Pkg.add("RadiometricCamera")
```

The package and the Visual Radiometry Toolkit packages it builds on are registered in
[CauRegistry](https://github.com/prittjam/CauRegistry.jl); registration in the General registry is planned. Only `RadiometricCamera/` is installed; the experiments and the manuscript are not downloaded with it. To run the experiments,
clone the repository and instantiate both environments:

```sh
julia --project=RadiometricCamera -e 'using Pkg; Pkg.instantiate()'
julia --project=experiments -e 'using Pkg; Pkg.instantiate()'
julia --project=experiments experiments/test/runtests.jl
```

## Project page

The project page is one site, deployed as one tree to the root of the `gh-pages` branch:

| URL | Built from |
|---|---|
| `/` | the landing page, `site/` (Documenter with the DocumenterLandingPage plugin) |
| `/docs/` | the package documentation, `RadiometricCamera/docs/` |
| `/paper.pdf` | the paper and its supplement, `paper/main.tex` (`paper/build/main.pdf`) |

- `site/src/index.md` — the landing page (hero frontmatter, abstract, method, results, BibTeX);
- `site/src/assets/landing/logo.svg`, `logo_dark.svg` — the hero logo (committed);
- `site/diagrams.jl` — draws the icon (`icon.svg`, `icon_dark.svg`, `favicon.png`,
  `avatar.png`) into `site/src/assets/landing/` on every build, from the measured
  image circle and vignetting of the fisheye; the files are not committed;
- `site/make.jl` — builds the landing page;
- `site/build.sh` — assembles the site: the landing page, the package docs into `docs/`,
  the paper PDF as `paper.pdf`, and checks the links between them;
- `site/deploy.jl` — deploys the assembled tree (on CI only).

### Build locally

```sh
julia --project=site -e 'using Pkg; Pkg.instantiate()'
julia --project=RadiometricCamera/docs -e 'using Pkg; Pkg.instantiate()'
make -C paper pdf
site/build.sh /tmp/site-preview
```

Open `/tmp/site-preview/index.html` (without an argument, the site lands in `site/build/`).
Nothing is committed: the built HTML and PDF are produced by the build.

### Deploy

`.github/workflows/site.yml` runs on every push to `main`: one job compiles `paper/main.tex`
(TeX Live, `xu-cheng/latex-action`), the next runs `site/build.sh` and then `site/deploy.jl`,
which deploys to the `gh-pages` branch of this repository with the workflow's `GITHUB_TOKEN`.
It is the only deploy to `gh-pages`: it replaces the whole branch content (`versions = nothing`),
so the package docs never deploy on their own. In the repository settings, set Pages to serve
the `gh-pages` branch.

## Package workflows

`.github/workflows/CI.yml` runs the package's tests (CPU) and builds its documentation; `CompatHelper.yml` and `TagBot.yml`
work on the package's subdirectory `RadiometricCamera/`.
