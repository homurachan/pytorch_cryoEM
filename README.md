# pytorch_cryoEM

pytorch implementation of many cryo-EM softwares. Right now all softwares come from the Grigorieff Lab.

## Required libraries

pytorch, numpy, scipy, mrcfile, tifffile, imagecodecs, matplotlib

# Contents

1. The pytorch-implementation of Unbend ([Lingli Kong](https://linglikong.github.io/) et al., https://elifesciences.org/reviewed-preprints/109119)

2. The pytorch-implementation of [ctffind 4.1.8](https://grigoriefflab.umassmed.edu/ctffind4) and [ctftilt](https://grigoriefflab.umassmed.edu/ctf_estimation_ctffind_ctftilt)

# Advantage

Using pytorch means the software runs on GPUs. When running on single RTX 4090, unbend is equal to about 30 CPU cores (half of the time is spent on decompress the tiff files),
and ctftilt is equal to about 120 CPU cores.

The pytorch-ctffind is roughly the same speed as [GCTF](https://www.sciencedirect.com/science/article/pii/S1047847715301003), and can output one relion 3.1 star-file, just like running the `relion_ctffind_runner`

# Disadvantage

The pytorch-unbend excludes less patches than the old version. But the final results don't have much differences.

The pytorch-ctftilt is not stable as the original version. To be fixed soon.
