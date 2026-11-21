#!/bin/bash
# =============================================================================
# run_of_record.sh — the run of record of the paper, in ONE command on a Slurm site, the moment RadiometricCamera <version> is registered:
#
#   VGTK_SITE=<site> bash experiments/scripts/slurm/run_of_record.sh <version> [--results-base=<dir>] --plan   # nothing submitted
#   bash experiments/scripts/slurm/run_of_record.sh <version>               # verify, make the tree, submit every stage
#   bash experiments/scripts/slurm/run_of_record.sh <version> --ref=<rev>   # the tree at <rev> (default v<version>): the registered tag
#                                                                         # plus tooling commits, its package code the tag's
#   bash experiments/scripts/slurm/run_of_record.sh <version> --dry-run=<rev> --from=<git dir | bundle> --scratch-sources=<file>
#        a SCRATCH dry run before the release (the plan only): the tree at <rev> with temporary [sources] (lines `<Package> = <path>`)
#        as one scratch commit, labelled SCRATCH everywhere and never a run of record
#
# 1. the registry: RadiometricCamera <version> is in the registry of record (configs/run_of_record.toml [launch] registry) after a
#    registry update, and resolves in a FRESH environment from the registries alone (no [sources], no dev path, no repo-url) — a CPU
#    job (srun --gres=none). Anything else FAILS LOUDLY (exit 10): no experiment runs on an unregistered version.
# 2. the tree: one clean clone of the repository of record (<results base>/[launch] tree_dir/tree), at v<version> (or --ref, whose package code —
#    RadiometricCamera/{src, bin, ext, Project.toml} — must be the tag's), the tag's RadiometricCamera/ tree the registry's git-tree-sha1, every [sources] a path inside
#    the tree; instantiated from the registries in a CPU job (no precompile: the graph's setup stage compiles inside the scheduler).
# 3. the launch: scripts/run_of_record.jl from a bare julia in the tree (VisualGeometryToolkitEval's runner on configs/reproduce.toml,
#    the stages of configs/run_of_record.toml): every stage an sbatch job chained upstream → downstream (afterok, aftercorr per item),
#    named ror-<driver>-<stage>[-<phase>], into the NEW results root <results base>/[launch] results_dir; each driver's produce_or_load keys reuse
#    every finished product, so nothing runs twice. A GPU-bound refinement gets one GPU; every CPU phase runs with no GPU
#    (configs/reproduce.toml phases); 8480M per CPU (the node ratio); the GPU queue in longest-first order (src/run_of_record.jl).
# The machine is the site profile's (VGTK_SITE; VisualGeometryToolkitEval configs/sites/): its setup puts julia on PATH
# (or JULIA) and sets JULIA_DEPOT_PATH; the results base is the project's results root on the machine: --results-base, else
# VGTK_RESULTS_ROOT (the site profile's [storage] results_root, the value to give: run_of_record.jl ror_results_base resolves the same). Run where sbatch is on PATH, from any checkout of the tooling; release jobs are never touched.
# =============================================================================
set -euo pipefail
V=${1:?usage: run_of_record.sh <version> [--plan] [--ref=<rev>] | <version> --dry-run=<rev> --from=<git> --scratch-sources=<file>}
shift
PLAN=""; REF=""; DRY=""; FROM=""; SCRATCH_SOURCES=""; BASE_ROOT=${VGTK_RESULTS_ROOT:-}
for a in "$@"; do
  case $a in
    --plan) PLAN=--plan ;;
    --ref=*) REF=${a#--ref=} ;;
    --dry-run=*) DRY=${a#--dry-run=}; PLAN=--plan ;;
    --from=*) FROM=${a#--from=} ;;
    --scratch-sources=*) SCRATCH_SOURCES=${a#--scratch-sources=} ;;
    --results-base=*) BASE_ROOT=${a#--results-base=} ;;
    *) echo "run_of_record.sh: unknown option $a"; exit 2 ;;
  esac
done
[[ $V =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "run_of_record.sh: version $V is not X.Y.Z"; exit 2; }
[ -n "$DRY" ] && { [ -n "$FROM" ] && [ -f "$SCRATCH_SOURCES" ] || { echo "--dry-run needs --from and --scratch-sources"; exit 2; }; }
command -v sbatch > /dev/null || { echo "run_of_record.sh: no sbatch: run it on a Slurm site"; exit 2; }
[ -n "$BASE_ROOT" ] || { echo "run_of_record.sh: no results base: give --results-base=<dir> or VGTK_RESULTS_ROOT (the site profile's [storage] results_root)"; exit 2; }
TOOL=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
CFG=$TOOL/experiments/configs/run_of_record.toml
conf() { python3 -I -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["launch"][sys.argv[2]].replace("{version}", sys.argv[3]))' "$CFG" "$1" "$V"; }
JL=${JULIA:-$(command -v julia)}; REPO=$(conf repository); REGISTRY=$(conf registry); BASE=$BASE_ROOT/$(conf tree_dir); RESULTS=$BASE_ROOT/$(conf results_dir)
[ -x "$JL" ] || { echo "run_of_record.sh: no julia (the site's setup puts it on PATH, or set JULIA)"; exit 2; }
export JULIA=$JL JULIA_PKG_USE_CLI_GIT=true JULIA_PKG_SERVER="" JULIA_PKG_PRECOMPILE_AUTO=0
mkdir -p "$BASE"
CPUJOB=(srun --gres=none --cpus-per-task=4 --mem-per-cpu=8480M --time=01:00:00)
say() { echo "[$(date +%T)] $*"; }
die() { echo; echo "!! run of record REFUSED: $*"; exit "${CODE:-10}"; }

# ---- 1. the registry ----------------------------------------------------------------------------------------------------------
say "RadiometricCamera $V in $REGISTRY: registry update and a fresh environment (CPU job)"
OUT=$("${CPUJOB[@]}" --job-name="ror-env-verify-$V" "$JL" --startup-file=no -e '
    using Pkg, TOML
    v, regname = ARGS[1], ARGS[2]
    Pkg.Registry.update()
    regs = filter(r -> r.name == regname, Pkg.Registry.reachable_registries())
    isempty(regs) && (println("NOREG no registry $regname in the depot"); exit(1))
    reg = only(regs)
    pkgs = TOML.parsefile(joinpath(reg.path, "Registry.toml"))["packages"]
    e = filter(p -> p.second["name"] == "RadiometricCamera", collect(pkgs))
    isempty(e) && (println("NOPKG RadiometricCamera not in $regname"); exit(1))
    vt = TOML.parsefile(joinpath(reg.path, only(e).second["path"], "Versions.toml"))
    haskey(vt, v) || (println("NOVERSION RadiometricCamera $v not registered in $regname (latest ", maximum(VersionNumber.(keys(vt))), ")"); exit(1))
    println("TREE ", vt[v]["git-tree-sha1"])
    Pkg.activate(mktempdir())
    Pkg.add(Pkg.PackageSpec(name = "RadiometricCamera", version = v))
    deps = Pkg.dependencies()
    rc = only(d for d in values(deps) if d.name == "RadiometricCamera")
    string(rc.version) == v || (println("NOTRESOLVED RadiometricCamera resolved to ", rc.version); exit(1))
    bad = [d.name for d in values(deps) if d.is_tracking_path || d.is_tracking_repo]
    isempty(bad) || (println("UNREGISTERED ", join(bad, ", ")); exit(1))
    println("RESOLVED RadiometricCamera ", rc.version, " with ", length(deps), " packages, every one from the registries")' "$V" "$REGISTRY" 2>&1) && OK=1 || OK=0
echo "$OUT" | grep -E '^(TREE|RESOLVED|NOREG|NOPKG|NOVERSION|NOTRESOLVED|UNREGISTERED)' | sed 's/^/  /' || true
TREE_SHA=$(echo "$OUT" | sed -n 's/^TREE //p')
if [ "$OK" != 1 ]; then
  if [ -n "$DRY" ]; then say "SCRATCH dry run: RadiometricCamera $V is not (yet) registered and resolvable — the launch proper would stop here"
  else echo "$OUT" | tail -20; die "RadiometricCamera $V is not registered in $REGISTRY or does not resolve in a fresh environment"; fi
fi

# ---- 2. the tree --------------------------------------------------------------------------------------------------------------
if [ -n "$DRY" ]; then T=$BASE/dry-tree; RESULTS=$BASE/dry-results    # a scratch tree and root of their own: never the run of record's
else T=$BASE/tree; fi
if [ ! -d "$T/.git" ]; then
  say "cloning ${FROM:-$REPO} into $T"
  git clone -q --no-checkout "${FROM:-$REPO}" "$T"
fi
git -C "$T" fetch -q --tags "${FROM:-$REPO}" "+refs/heads/*:refs/remotes/src/*" 2>/dev/null || git -C "$T" fetch -q --tags "${FROM:-$REPO}"
if [ -n "$DRY" ]; then
  REV=$(git -C "$T" rev-parse --verify "src/$DRY^{commit}" 2>/dev/null || git -C "$T" rev-parse --verify "$DRY^{commit}")
else
  REV=$(git -C "$T" rev-parse --verify "${REF:-v$V}^{commit}") || die "no tag v$V (or ref $REF) in $REPO"
  # the package is registered from its subdirectory (subdir = "RadiometricCamera"): the registry's git-tree-sha1 is that directory's tree
  [ "$(git -C "$T" rev-parse "v$V:RadiometricCamera")" = "$TREE_SHA" ] || die "the tag v$V's RadiometricCamera tree $(git -C "$T" rev-parse "v$V:RadiometricCamera") is not the registered git-tree-sha1 $TREE_SHA"
  git -C "$T" diff --quiet "v$V" "$REV" -- RadiometricCamera/src RadiometricCamera/bin RadiometricCamera/ext RadiometricCamera/Project.toml ||
    die "$REV's package code (RadiometricCamera/: src bin ext Project.toml) differs from the registered v$V"
fi
HEAD_NOW=$(git -C "$T" rev-parse HEAD 2>/dev/null || echo none)
if [ -n "$DRY" ]; then
  # the scratch commit on <rev>: the temporary [sources] in every tracked Project.toml that lists the package (never pushed)
  if [ "$(git -C "$T" rev-parse HEAD~1 2>/dev/null || echo none)" != "$REV" ] || ! git -C "$T" log -1 --format=%s | grep -q '^SCRATCH'; then
    git -C "$T" checkout -q --detach "$REV"
    python3 -I - "$T" "$SCRATCH_SOURCES" <<'PY'
import sys, re, subprocess, tomllib
tree, src = sys.argv[1], sys.argv[2]
pairs = [tuple(x.strip() for x in l.split("=", 1)) for l in open(src) if l.strip() and not l.startswith("#")]
for f in subprocess.check_output(["git", "-C", tree, "ls-files", "*Project.toml"], text=True).split():
    p = f"{tree}/{f}"; d = tomllib.load(open(p, "rb")); text = open(p).read()
    names = set(d.get("deps", {})) | set(d.get("weakdeps", {})) | set(d.get("extras", {}))
    add = [f'{n} = {{path = "{path}"}}   # SCRATCH (run_of_record.sh --dry-run only; never pushed)' for n, path in pairs if n in names and n not in d.get("sources", {})]
    if not add: continue
    text = re.sub(r"^\[sources\]\n", "[sources]\n" + "\n".join(add) + "\n", text, count=1, flags=re.M) if "sources" in d else text.rstrip("\n") + "\n\n[sources]\n" + "\n".join(add) + "\n"
    open(p, "w").write(text); print("  SCRATCH [sources] ->", f)
PY
    git -C "$T" -c user.name=ror-dry -c user.email=ror-dry@localhost commit -qam "SCRATCH (run_of_record.sh --dry-run only, never pushed): temporary [sources] $(paste -sd' ' "$SCRATCH_SOURCES" | cut -c1-200)"
  fi
elif [ "$HEAD_NOW" != "$REV" ]; then
  git -C "$T" checkout -q --detach "$REV"
fi
[ -z "$(git -C "$T" status --porcelain --untracked-files=no)" ] || die "the tree $T is not clean"
git -C "$T" ls-files | grep -qE '(^|/)Manifest(-v[0-9.]+)?\.toml$' && die "a tracked Manifest in $REV"
CODE=6
python3 -I - "$T" "${DRY:+scratch}" <<'PY' || die "[sources] outside the tree (an unregistered package)"
import sys, os, subprocess, tomllib
tree, scratch = os.path.realpath(sys.argv[1]), sys.argv[2] == "scratch"
bad = 0
for f in subprocess.check_output(["git", "-C", tree, "ls-files", "*Project.toml"], text=True).split():
    for n, s in tomllib.load(open(f"{tree}/{f}", "rb")).get("sources", {}).items():
        p = os.path.realpath(os.path.join(tree, os.path.dirname(f), s.get("path", "/nonexistent-url-source")))
        inside = p == tree or p.startswith(tree + "/")
        if not inside:
            print(f"  [sources] {f}: {n} => {s}" + ("  SCRATCH (dry run)" if scratch else "  OUTSIDE THE TREE")); bad += not scratch
sys.exit(1 if bad else 0)
PY
CODE=10
say "tree $T at $(git -C "$T" rev-parse --short HEAD)$([ -n "$DRY" ] && echo " (SCRATCH on $REV)" || echo " (registered v$V$([ "$REV" != "$(git -C "$T" rev-parse "v$V^{commit}")" ] && echo " + tooling, package code identical"))")"
say "instantiate the tree from the registries (CPU job; the setup stage precompiles inside the scheduler)"
(cd "$T" && "${CPUJOB[@]}" --job-name="ror-env-instantiate-$V" bash -c "'$JL' --startup-file=no --project=experiments -e 'using Pkg; Pkg.instantiate()' && '$JL' --startup-file=no --project=RadiometricCamera -e 'using Pkg; Pkg.instantiate()'") ||
  die "the tree does not instantiate"

# ---- 3. the launch ------------------------------------------------------------------------------------------------------------
cd "$T"
mkdir -p logs
say "${PLAN:+plan of }the run of record: results $RESULTS"
[ -n "$DRY" ] && export VGTK_SCRATCH=1                  # a scratch launch: never a run of record's key (the runner)
exec "$JL" --startup-file=no --project=experiments experiments/scripts/run_of_record.jl --version="$V" --results-root="$RESULTS" $PLAN ${DRY:+--scratch}
