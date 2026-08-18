#!/usr/bin/env bash
# Install what the `reachy2_ppo` extension needs on top of the shared Isaac Lab base
# environment (../isaaclab_setup.sh's `env_isaaclab`): isaaclab_tasks (env
# configs, including the nut-pour task), isaaclab_rl (the skrl glue
# train.py/play.py import), and isaaclab_assets (robot configs). None of
# these are published as standalone PyPI packages -- isaaclab_setup.sh only
# installs the core `isaaclab` package from PyPI by design -- so this clones
# the IsaacLab repo at the git tag matching whatever `isaaclab` version is
# actually installed, and editable-installs those three subpackages from it.
# Also installs skrl and the `reachy2_ppo` extension itself.
#
# Idempotent: rerunning preserves the existing checkout/installs.

set -Eeuo pipefail
# Without this, a `die`/`exit` inside a nested $(...) command substitution
# (e.g. resolve_isaaclab_tasks_ref calling installed_isaaclab_version) only
# kills that inner subshell instead of aborting the script, so a single
# failure can cascade into a second, confusing error instead of stopping
# cleanly at the first one.
shopt -s inherit_errexit

SCRIPT_NAME="$(basename "$0")"
SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || printf '%s' "$SCRIPT_DIR")"

# Same base environment ../isaaclab_setup.sh creates; this script only adds
# to it, and never creates it.
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
CONDA_ROOT="${CONDA_ROOT:-$PROJECT_ROOT/conda}"

ISAACLAB_TASKS_DIR="${ISAACLAB_TASKS_DIR:-$SCRIPT_DIR/../../nutpouring/ppo/IsaacLab}"
ISAACLAB_TASKS_REPO="${ISAACLAB_TASKS_REPO:-https://github.com/isaac-sim/IsaacLab.git}"
# Overrides auto-detection (the git tag matching the installed `isaaclab`
# package version) -- only needed if that exact tag doesn't exist upstream.
ISAACLAB_TASKS_REF="${ISAACLAB_TASKS_REF:-}"

SKRL_MIN_VERSION="${SKRL_MIN_VERSION:-1.4.3}"

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME COMMAND

  bootstrap          Run every step below, in order
  install-tasks      Clone IsaacLab at the tag matching your installed
                      isaaclab version and editable-install isaaclab_tasks,
                      isaaclab_rl, and isaaclab_assets into $CONDA_ENV
  install-skrl       Install skrl>=$SKRL_MIN_VERSION into $CONDA_ENV
  install-extension  pip install -e source/reachy2_ppo into $CONDA_ENV

Verify:
  doctor        Check installs and versions (fast; does not boot Isaac Sim)
  self-test     Check this Bash file without requiring Isaac Lab
  show-config   Print resolved paths and versions

Prerequisite (not run by this script):
  ../isaaclab_setup.sh setup-software   # creates the $CONDA_ENV environment

Quick start:
  ./$SCRIPT_NAME bootstrap
  ./$SCRIPT_NAME doctor
  python scripts/list_envs.py   # real end-to-end check: boots Isaac Sim and
                                 # confirms Template-Nut-Pour-Ppo-v0 registers

Useful overrides:
  CONDA_ENV=another-env CONDA_ROOT=/abs/path
  ISAACLAB_TASKS_DIR=/abs/path    (default: the nutpouring baseline's IsaacLab checkout)
  ISAACLAB_TASKS_REF=v2.3.2       (default: auto-detected from the installed
                                    isaaclab package version)
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

info() {
  printf '\n[%s] %s\n' "$SCRIPT_NAME" "$*"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_file() {
  [[ -f "$1" ]] || die "File not found: $1"
}

find_conda_base() {
  [[ -x "$CONDA_ROOT/bin/conda" ]] && printf '%s\n' "$CONDA_ROOT"
}

activate_conda() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    return
  fi
  local conda_base=""
  conda_base="$(find_conda_base 2>/dev/null || true)"
  [[ -n "$conda_base" ]] || die "Conda was not found at $CONDA_ROOT. Run '../isaaclab_setup.sh setup-software' first."
  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"
  conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$CONDA_ENV" ||
    die "Conda environment '$CONDA_ENV' not found. Run '../isaaclab_setup.sh setup-software' first."
  conda activate "$CONDA_ENV"
  # Isaac Sim's bundled extensions need a newer libstdc++ (CXXABI_1.3.15+)
  # than Ubuntu 22.04 ships. Conda's own copy has it; put it ahead of the
  # system one so the linker finds it first.
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

installed_isaaclab_version() {
  # Read installed package metadata rather than `import isaaclab; ...
  # __version__` -- the isaaclab package doesn't expose `__version__` as a
  # module attribute. Reading metadata also avoids the `omni.*`/`pxr` import
  # chain that isaaclab_tasks/isaaclab_rl pull in (those only exist once
  # Isaac Sim's Kit app has launched, see AppLauncher in every scripts/*.py
  # entry point), so this stays a fast, plain `python -c`.
  python -c "import importlib.metadata as m; print(m.version('isaaclab'))" 2>/dev/null ||
    die "isaaclab is not installed in '$CONDA_ENV'. Run '../isaaclab_setup.sh setup-software' first."
}

resolve_isaaclab_tasks_ref() {
  if [[ -n "$ISAACLAB_TASKS_REF" ]]; then
    printf '%s\n' "$ISAACLAB_TASKS_REF"
    return
  fi
  local installed_version
  installed_version="$(installed_isaaclab_version)"
  # Strip any post/dev/local suffix (e.g. "2.3.2.post1" -> "2.3.2"); IsaacLab
  # tags releases as "v<major>.<minor>.<patch>" with no such suffix.
  local core_version
  core_version="$(printf '%s' "$installed_version" | grep -oE '^[0-9]+\.[0-9]+\.[0-9]+')"
  [[ -n "$core_version" ]] || die "Could not parse an X.Y.Z version out of installed isaaclab version '$installed_version'."
  printf 'v%s\n' "$core_version"
}

clone_isaaclab_tasks_repo() {
  require_command git
  activate_conda
  local ref
  ref="$(resolve_isaaclab_tasks_ref)"

  if [[ -d "$ISAACLAB_TASKS_DIR/.git" ]]; then
    local current_ref
    current_ref="$(git -C "$ISAACLAB_TASKS_DIR" describe --tags --exact-match 2>/dev/null || true)"
    if [[ "$current_ref" == "$ref" ]]; then
      printf 'IsaacLab checkout already at %s: %s\n' "$ref" "$ISAACLAB_TASKS_DIR"
      return
    fi
    die "$ISAACLAB_TASKS_DIR exists but is at '${current_ref:-an unrecognized ref}', not '$ref'" \
      "(the tag matching your installed isaaclab version). Move it aside, or set ISAACLAB_TASKS_DIR to a different path."
  fi

  info "Cloning IsaacLab $ref (only used for isaaclab_tasks/isaaclab_rl/isaaclab_assets) into $ISAACLAB_TASKS_DIR"
  git clone --branch "$ref" --depth 1 "$ISAACLAB_TASKS_REPO" "$ISAACLAB_TASKS_DIR"
}

install_tasks() {
  clone_isaaclab_tasks_repo
  activate_conda
  info "Editable-installing isaaclab_tasks, isaaclab_rl, isaaclab_assets from $ISAACLAB_TASKS_DIR"
  python -m pip install \
    -e "$ISAACLAB_TASKS_DIR/source/isaaclab_tasks" \
    -e "$ISAACLAB_TASKS_DIR/source/isaaclab_rl" \
    -e "$ISAACLAB_TASKS_DIR/source/isaaclab_assets"
}

install_skrl() {
  activate_conda
  if python -c "
import sys
from packaging import version
import skrl
sys.exit(0 if version.parse(skrl.__version__) >= version.parse('$SKRL_MIN_VERSION') else 1)
" 2>/dev/null; then
    printf 'skrl>=%s already installed.\n' "$SKRL_MIN_VERSION"
    return
  fi
  info "Installing skrl>=$SKRL_MIN_VERSION"
  python -m pip install "skrl>=$SKRL_MIN_VERSION"
}

install_extension() {
  activate_conda
  require_file "$SCRIPT_DIR/source/reachy2_ppo/setup.py"
  info "Installing the 'reachy2_ppo' extension (editable) from $SCRIPT_DIR/source/reachy2_ppo"
  python -m pip install -e "$SCRIPT_DIR/source/reachy2_ppo"
}

bootstrap() {
  install_tasks
  install_skrl
  install_extension
  doctor
  info "reachy2_ppo setup completed"
}

show_config() {
  cat <<EOF
PROJECT_ROOT=$PROJECT_ROOT
CONDA_ENV=$CONDA_ENV
CONDA_ROOT=$CONDA_ROOT
ISAACLAB_TASKS_DIR=$ISAACLAB_TASKS_DIR
ISAACLAB_TASKS_REPO=$ISAACLAB_TASKS_REPO
ISAACLAB_TASKS_REF=${ISAACLAB_TASKS_REF:-<auto-detected from installed isaaclab version>}
SKRL_MIN_VERSION=$SKRL_MIN_VERSION
EOF
}

doctor() {
  local failures=0
  activate_conda

  printf '\nBare-importable packages (no Isaac Sim boot required):\n'
  if ! python - <<PY
import importlib
import sys

required = ("torch", "isaaclab", "isaacsim", "skrl")
failed = []
for name in required:
    try:
        importlib.import_module(name)
        print(f"OK import: {name}")
    except Exception as exc:
        failed.append(name)
        print(f"FAILED import: {name}: {exc}", file=sys.stderr)
raise SystemExit(bool(failed))
PY
  then
    failures=$((failures + 1))
  fi

  # isaaclab_tasks / isaaclab_rl / reachy2_ppo all transitively pull in `omni.*`,
  # which only exists once Isaac Sim's Kit app has launched -- so this
  # checks install *metadata* rather than actually importing them. The real
  # import-and-register check is `python scripts/list_envs.py`, which boots
  # Kit the same way train.py does.
  printf '\nInstalled packages (metadata only; not imported):\n'
  local pkg
  for pkg in isaaclab_tasks isaaclab_rl isaaclab_assets reachy2_ppo; do
    if python -m pip show "$pkg" >/dev/null 2>&1; then
      printf 'OK installed: %s\n' "$pkg"
    else
      printf 'FAILED: %s is not installed in %s\n' "$pkg" "$CONDA_ENV" >&2
      failures=$((failures + 1))
    fi
  done

  if (( failures > 0 )); then
    die "$failures required check(s) failed."
  fi
  printf '\nAll required checks passed. Run '\''python scripts/list_envs.py'\'' next for the real end-to-end check.\n'
}

self_test() {
  bash -n "$SCRIPT_PATH"
  [[ "$SKRL_MIN_VERSION" == "1.4.3" ]]
  printf 'Bash syntax and pinned-configuration checks passed.\n'
  if command -v shellcheck >/dev/null 2>&1; then
    shellcheck "$SCRIPT_PATH"
    printf 'ShellCheck passed.\n'
  else
    printf 'ShellCheck is not installed; static lint was skipped.\n'
  fi
}

main() {
  case "${1:-}" in
    bootstrap) bootstrap ;;
    install-tasks) install_tasks ;;
    install-skrl) install_skrl ;;
    install-extension) install_extension ;;
    doctor) doctor ;;
    show-config) show_config ;;
    self-test) self_test ;;
    help|-h|--help|'') usage ;;
    *) die "Unknown command '$1'. Run '$SCRIPT_NAME help'." ;;
  esac
}

main "$@"
