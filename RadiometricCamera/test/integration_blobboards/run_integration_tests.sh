#!/usr/bin/env bash
# MIGRATION TESTS for the BlobBoards stack and VisualGeometryCore (README.md next to this file). Stages, in order:
#   C  test_session_outputs.jl     (harness env)       RadiometricCamera on the session RECORDS (VisualGeometryToolkitIO; written once by
#                                                      CaliBlobs' scripts/migrate_rc_session.jl into test/fixtures/lenssim_fisheye3/session_io),
#                                                      held-out truth
#   D  test_vgc_contract.jl        (harness env)       the VisualGeometryCore contract of the ray-tracing code (no BlobBoards)
#   E  Pkg.test()                  (package env)       the full package suite, only with --with-pkg-tests
# Usage: bash test/integration_blobboards/run_integration_tests.sh [--record] [--with-pkg-tests]
#   --record          write golden/session_fisheye3.json + golden/recorded_stack.txt from the CURRENT stack (once, on the
#                     reference stack, BEFORE migrating; both files are versioned)
#   --with-pkg-tests  also run the package suite (stage E) — needed when VisualGeometryCore is bumped in the package environment
# GPU 1 unless CUDA_VISIBLE_DEVICES is set. Every invocation writes into a fresh test/integration_blobboards/work/run-<time>/
# (git-ignored; work/latest points at it; nothing is deleted). Exit code 0 = every stage succeeded.
set -u
RC=$(cd "$(dirname "$0")/../.." && pwd); cd "$RC"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
M=test/integration_blobboards; HARNESS=test/integration_blobboards   # stages C, D: its Project.toml
FIX=test/fixtures/lenssim_fisheye3; VIEWS="v1,v2,v3"
RECORD=0; PKGTESTS=0
while [ $# -gt 0 ]; do
  case $1 in
    --record) RECORD=1 ;;
    --with-pkg-tests) PKGTESTS=1 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
RUN=$M/work/run-$(date +%Y%m%d-%H%M%S); mkdir -p "$RUN/logs"; ln -sfn "$(basename "$RUN")" "$M/work/latest"
export RC_MIG_WORK="$RC/$RUN"
status=0; report=()
log(){ echo "[$(date +%H:%M:%S)] $*"; }
stage(){ # name command...
  name=$1; shift; log "=== $name"
  if "$@" > "$RUN/logs/$name.log" 2>&1; then report+=("OK    $name"); else report+=("FAIL  $name   (see $RUN/logs/$name.log)"); status=1; fi
  grep -aE "Test Summary|^\s*(Pa""ss|Fail|Error|Broken|Total)|SESSION DONE|no board accepted|ERROR|frames:" "$RUN/logs/$name.log" | tail -n 8 | sed 's/^/    /'
}
for need in "$FIX/manifest.json" "$FIX/lsc_v1_ref.png" "$FIX/gtrays/board_v1_uv.exr" "$FIX/gtrays/board_val1_uv.exr" "$FIX/board_291/geometry.json" "$FIX/session_io/camera.json"; do
  [ -e "$need" ] || { echo "missing fixture input $need"; exit 2; }
done
log "run folder $RUN"

stage C_consumers env RC_MIG_FIX="$RC/$FIX" RC_MIG_VIEWS=$VIEWS RC_MIG_RECORD=$RECORD julia --project=$HARNESS $M/test_session_outputs.jl

stage D_vgc_contract julia --project=$HARNESS $M/test_vgc_contract.jl

if [ $PKGTESTS = 1 ]; then
  stage E_pkg_tests julia --project=. -e 'using Pkg; Pkg.test()'
fi

if [ $RECORD = 1 ]; then
  { cat "$RUN/stack_versions.txt" "$RUN/rc_stack_versions.txt" 2>/dev/null
    for r in BlobBoards.jl RobustVisualGeometry.jl VisualGeometryFeatures.jl VisualGeometryIo.jl VisualGeometryCore.jl; do
      d=$RC/../../$r; [ -d "$d" ] && echo "$r $(git -C "$d" rev-parse --short HEAD 2>/dev/null) $(grep -E '^version' "$d/Project.toml" | head -n 1)"
    done
    echo "recorded $(date '+%Y-%m-%d %H:%M:%S') from $RUN"; } > "$M/golden/recorded_stack.txt"
  log "golden values recorded from: $(head -n 1 "$M/golden/recorded_stack.txt")"
fi
echo; echo "=== migration test summary ($RUN) ==="; printf '%s\n' "${report[@]}"
exit $status
