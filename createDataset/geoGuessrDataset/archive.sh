#!/bin/bash

# Use find to list jpg files and split into chunks of 10,000
find . -maxdepth 1 -name "*.jpg" -print | sed 's|^\./||' | split -l 10000 - files_chunk_

# Loop over each chunk file and create a tar.gz archive
for f in files_chunk_*; do
  tar_name="archive_${f##*_}.tar.gz"
  tar -czf "$tar_name" -T "$f"
done

# Remove the temporary chunk files
rm files_chunk_*

