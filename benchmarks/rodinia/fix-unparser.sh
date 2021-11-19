#!/usr/bin/env bash

if (( $# < 1 )); then
  echo "fix-unparser.sh target [sed-script-file]"
  exit 1
fi

target="$1"

if (( $# > 2 )); then
  script="$2"
else
  script=$(dirname "${target}")
  script="$script/fix-unparser.sh"
fi

echo "$script"

if test -f "$script"; then
    exec "$script" "$target"
fi

