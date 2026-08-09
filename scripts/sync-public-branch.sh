#!/usr/bin/env bash
# Mirrors master's tracked changes onto main (the GitHub-facing branch) and
# pushes. Skips .planning/ and the local-only demo runtime files
# (llmsec.config.yaml, demo-app/backend/dvla_config.db) — those never leave
# master. Run from master; the script returns you there when done.
#
# Usage: scripts/sync-public-branch.sh ["commit message"]

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

EXCLUDE_PATHSPECS=(':!.planning' ':!llmsec.config.yaml' ':!demo-app/backend/dvla_config.db')
COMMIT_MSG="${1:-sync: mirror master changes to main}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: working tree not clean — commit or stash first." >&2
  exit 1
fi

CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != "master" ]]; then
  echo "error: run this from master (currently on '$CURRENT_BRANCH')." >&2
  exit 1
fi

if ! git show-ref --verify --quiet refs/heads/main; then
  echo "error: no local 'main' branch found." >&2
  exit 1
fi

git checkout main

# Files that differ between main and master, excluding local-only paths.
# Status letters: A (only in master -> add to main), M (modified -> copy from
# master), D (only in main, gone from master -> remove from main).
DIFF_OUTPUT=$(git diff --name-status main master -- . "${EXCLUDE_PATHSPECS[@]}")

if [[ -z "$DIFF_OUTPUT" ]]; then
  echo "Nothing to sync — main already matches master (outside excluded paths)."
  git checkout "$CURRENT_BRANCH"
  exit 0
fi

echo "Changes to sync:"
echo "$DIFF_OUTPUT"
echo

while IFS=$'\t' read -r status file rest; do
  case "$status" in
    A|M)
      git checkout master -- "$file"
      ;;
    D)
      git rm -q -- "$file"
      ;;
    R*)
      # Rename: $file is the old path, $rest is the new path.
      git rm -q -- "$file" 2>/dev/null || true
      git checkout master -- "$rest"
      ;;
    *)
      echo "warning: unhandled status '$status' for '$file' — skipping." >&2
      ;;
  esac
done <<< "$DIFF_OUTPUT"

git add -A -- . "${EXCLUDE_PATHSPECS[@]}"

if git diff --cached --quiet; then
  echo "Nothing staged after sync — nothing to commit."
  git checkout "$CURRENT_BRANCH"
  exit 0
fi

git commit -m "$COMMIT_MSG"
git push origin main
git checkout "$CURRENT_BRANCH"

echo "main synced and pushed."
