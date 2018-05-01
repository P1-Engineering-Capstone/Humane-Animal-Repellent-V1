#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/Squirrels
DATA=data/Squirrels
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/Squirrels_train_lmdb \
  $DATA/Squirrels_mean.binaryproto

echo "Done."
