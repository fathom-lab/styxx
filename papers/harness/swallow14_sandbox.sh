#!/bin/bash
# SWALLOW-14's second run: the interpreter the frozen instrument was given as sys.executable, so that
# each repository's process -- the one that runs the workflows' steps' shell -- ran in a throwaway
# copy of the machine. Not part of the instrument, which is unchanged; this is the environment it ran
# in (RESULT_swallow14 §0). Installed as /home/claude/sbx/python3.v2 and started as
#
#   /home/claude/sbx/python3.v2 -m benchmarks.harness_mutation.empty_list --population ... --work ... --out ... --workers 3
#
# (bash's `exec -a` below keeps the wrapper's path in the parent's sys.executable, so the instrument's
# `[sys.executable, "-m", ..., "--one", REPO, ...]` for each repository comes back here). It needs
# root (mount, bubblewrap 0.9) and a table of an id per repository (uids.tsv: repo<TAB>id).
# The interpreter the frozen SWALLOW-14 instrument calls for each repository (its sys.executable).
# Each repository's process runs as root, as before, but in a throwaway copy of the machine: the
# root filesystem under an overlay whose writes land in a tmpfs of its own, an empty /tmp, a minimal
# /dev, its own process tree, no capabilities beyond a container's file-owner set, and none of the
# session's secrets in its environment. What a simulated step writes or deletes happens in that copy
# and is gone when the process ends; only the result file is copied out. Anything else runs the same
# python unchanged.
REAL=/usr/bin/python3
SBX=/home/claude/sbx
if [ "$1" = "-m" ] && [ "$3" = "--one" ]; then
  json=""; args=("$@")
  for ((k = 0; k < ${#args[@]}; k++)); do [ "${args[$k]}" = "--json" ] && json="${args[$((k + 1))]}"; done
  [ -n "$json" ] || { echo "sbx: no --json" >&2; exit 96; }
  id=$(awk -F'\t' -v r="$4" '$1 == r {print $2; exit}' "$SBX/uids.tsv")
  [ -n "$id" ] || { echo "sbx: no id for $4" >&2; exit 97; }
  d="$SBX/ov/$id"
  if mountpoint -q "$d/root"; then umount -l "$d/root"; fi
  if mountpoint -q "$d"; then umount -l "$d"; fi
  mkdir -p "$d" && mount -t tmpfs -o size=2g,mode=700 tmpfs "$d" && mkdir -p "$d/up" "$d/wk" "$d/root" "$d/out" \
    && mount -t overlay overlay -o "lowerdir=/,upperdir=$d/up,workdir=$d/wk" "$d/root" || { echo "sbx: overlay failed" >&2; exit 98; }
  ulimit -f 4194304
  bwrap --bind "$d/root" / --dev /dev --proc /proc --tmpfs /tmp --bind "$d/out" "$(dirname "$json")" \
    --unshare-pid --unshare-ipc --unshare-uts --die-with-parent --new-session --chdir "$PWD" \
    --cap-drop ALL --cap-add CAP_CHOWN --cap-add CAP_DAC_OVERRIDE --cap-add CAP_DAC_READ_SEARCH --cap-add CAP_FOWNER \
    --cap-add CAP_FSETID --cap-add CAP_KILL --cap-add CAP_SETGID --cap-add CAP_SETUID --cap-add CAP_SETPCAP --cap-add CAP_SETFCAP \
    --cap-add CAP_NET_BIND_SERVICE --cap-add CAP_SYS_CHROOT --cap-add CAP_AUDIT_WRITE \
    --clearenv --setenv PATH "$PATH" --setenv HOME /root \
    --setenv HTTPS_PROXY "$HTTPS_PROXY" --setenv https_proxy "$HTTPS_PROXY" --setenv NO_PROXY "$NO_PROXY" --setenv no_proxy "$NO_PROXY" \
    --setenv GIT_SSL_CAINFO /root/.ccr/ca-bundle.crt --setenv SSL_CERT_FILE /root/.ccr/ca-bundle.crt \
    --setenv REQUESTS_CA_BUNDLE /root/.ccr/ca-bundle.crt --setenv CURL_CA_BUNDLE /root/.ccr/ca-bundle.crt --setenv GIT_TERMINAL_PROMPT 0 \
    "$REAL" "$@"
  rc=$?
  f="$d/out/$(basename "$json")"
  if [ -f "$f" ]; then cp "$f" "$json.part" && mv -f "$json.part" "$json"; fi
  umount -l "$d/root"; umount -l "$d"; rmdir "$d" 2>/dev/null
  exit $rc
fi
exec -a "$0" "$REAL" "$@"
