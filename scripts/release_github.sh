#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "scripts/release_github.sh <py|jl> [version]"
  exit 0
fi

target="${1:-}"
if [[ -z "${target}" ]]; then
  echo "ERROR: missing release target (expected py or jl)" >&2
  exit 1
fi

case "${target}" in
  py|python)
    mode="py"
    ;;
  jl|julia)
    mode="jl"
    ;;
  *)
    echo "ERROR: unknown release target '${target}' (expected py or jl)" >&2
    exit 1
    ;;
esac

if [[ -n "${2:-}" ]]; then
  raw_version="$2"
else
  if [[ ! -f VERSION ]]; then
    echo "ERROR: VERSION file not found" >&2
    exit 1
  fi
  raw_version="$(tr -d '[:space:]' < VERSION)"
fi

if [[ "${raw_version}" == python-v* ]]; then
  version="${raw_version#python-v}"
elif [[ "${raw_version}" == julia-v* ]]; then
  version="${raw_version#julia-v}"
else
  version="${raw_version#v}"
fi

if [[ -z "${version}" ]]; then
  echo "ERROR: empty version" >&2
  exit 1
fi

git update-index -q --refresh
if ! git diff-index --quiet HEAD --; then
  echo "ERROR: tracked files have uncommitted changes; commit before releasing" >&2
  exit 1
fi

if [[ "${mode}" == "py" ]]; then
  tag="python-v${version}"
  file_version="$(tr -d '[:space:]' < VERSION)"

  if [[ "${file_version}" != "${version}" ]]; then
    echo "ERROR: requested Python version (${version}) does not match VERSION file (${file_version})" >&2
    exit 1
  fi

  if git rev-parse --verify --quiet "refs/tags/${tag}" >/dev/null; then
    echo "ERROR: tag '${tag}' already exists locally" >&2
    exit 1
  fi

  if git ls-remote --exit-code --tags origin "refs/tags/${tag}" >/dev/null 2>&1; then
    echo "ERROR: tag '${tag}' already exists on origin" >&2
    exit 1
  fi

  echo "Creating tag ${tag} on current HEAD..."
  git tag "${tag}"

  echo "Pushing tag ${tag} to origin..."
  git push origin "${tag}"

  if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: GitHub CLI 'gh' is required for Python releases" >&2
    exit 1
  fi

  notes_file="$(mktemp)"
  trap 'rm -f "${notes_file:-}"' EXIT
  python scripts/release_notes.py "${version}" > "${notes_file}"

  echo "Creating GitHub release ${tag}..."
  gh release create "${tag}" --notes-file "${notes_file}"
  rm -f "${notes_file}"
  trap - EXIT
  echo "Release ${tag} published. This triggers .github/workflows/python-publish.yml."
else
  if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: GitHub CLI 'gh' is required for Julia releases" >&2
    exit 1
  fi

  julia_toml="julia/Scorio.jl/Project.toml"
  if [[ ! -f "${julia_toml}" ]]; then
    echo "ERROR: ${julia_toml} not found" >&2
    exit 1
  fi

  julia_version="$(sed -nE 's/^version = "([^"]+)"/\1/p' "${julia_toml}" | head -n1)"
  if [[ -z "${julia_version}" ]]; then
    echo "ERROR: could not read version from ${julia_toml}" >&2
    exit 1
  fi

  if [[ "${julia_version}" != "${version}" ]]; then
    echo "ERROR: VERSION (${version}) does not match ${julia_toml} version (${julia_version})" >&2
    exit 1
  fi

  sha="$(git rev-parse HEAD)"
  ref="$(git rev-parse --abbrev-ref HEAD)"
  if [[ "${ref}" == "HEAD" ]]; then
    echo "ERROR: detached HEAD is not supported for Julia release dispatch" >&2
    exit 1
  fi

  if ! git ls-remote --exit-code --heads origin "${ref}" >/dev/null 2>&1; then
    echo "ERROR: branch '${ref}' does not exist on origin; push it before release-jl" >&2
    exit 1
  fi

  git fetch --quiet origin "${ref}"
  if ! git merge-base --is-ancestor "${sha}" "origin/${ref}"; then
    echo "ERROR: current commit ${sha} is not pushed to origin/${ref}; push before release-jl" >&2
    exit 1
  fi

  echo "Dispatching Julia registration workflow for version ${version} on ${sha}..."
  gh workflow run julia-register.yml --ref "${ref}" -f sha="${sha}" -f version="${version}" -f subdir="julia/Scorio.jl"
  echo "Julia registration workflow dispatched."
  echo "Track with: gh run list --workflow julia-register.yml --limit 1"
fi
