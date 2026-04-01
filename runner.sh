#!/usr/bin/env bash
# HPC execution needs explicit interpreter control because `conda activate`
# can be unreliable and `python` may resolve to the wrong environment.
# Centralize Python execution here so environment switching is one variable.
ENV_NAME=${ENV_NAME:-unified_tsf}
PYTHON_RUNNER=(conda run -n "$ENV_NAME" python)
