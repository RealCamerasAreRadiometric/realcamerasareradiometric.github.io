#!/bin/bash
# Every result and figure of the paper on the desktop (one RTX 4090, no Slurm; device 0): reproduce.sh with the site desktop.
exec bash "$(dirname "${BASH_SOURCE[0]}")/reproduce.sh" --site=desktop "$@"
