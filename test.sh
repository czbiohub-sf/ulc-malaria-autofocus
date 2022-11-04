#! /usr/bin/sh

if [[ $# -eq 0 ]]; then
  v="python3 train.py"
else
  v="$@"
fi

echo $v
