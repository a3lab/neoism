#!/bin/bash
for filename in "$@"
do
  base=$(basename $filename)
  path=$(dirname $filename)
  echo "$base $path"
  python3 preprocess_text_file.py "$filename" "${path}/preproc_${base}"
done
