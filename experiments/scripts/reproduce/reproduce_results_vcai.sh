#!/bin/bash
# Every result and figure of the paper on the vcai Slurm cluster: reproduce.sh with the site vcai. --dry-run, --only=…, --no-deps, status.
exec bash "$(dirname "${BASH_SOURCE[0]}")/reproduce.sh" --site=vcai "$@"
