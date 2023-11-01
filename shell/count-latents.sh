#!/usr/bin/env bash
echo /p/scratch/ccstdl/birch1/dataset-out/imagenet-latents/wds/0000{0,1}.tar \
| xargs -n 1 tar -tf \
| awk '/\.latent\.pth$/ {
  distinct_latent_filenames[$0] = 1
}
END {
  count = 0;
  for (key in distinct_latent_filenames) {
    count++;
  }
  print "distinct latent filenames:", count
}'