# pytorch_cryoEM

pytorch implementation (by ChatGPT 5.5 and 5.6) of many cryo-EM softwares. Right now all softwares come from the Grigorieff Lab.

## Required libraries

pytorch, numpy, scipy, mrcfile, tifffile, imagecodecs, matplotlib

# Contents

1. The pytorch-implementation of Unbend ([Lingli Kong](https://linglikong.github.io/) et al., https://elifesciences.org/reviewed-preprints/109119)

2. The pytorch-implementation of [ctffind 4.1.8](https://grigoriefflab.umassmed.edu/ctffind4) and [ctftilt](https://grigoriefflab.umassmed.edu/ctf_estimation_ctffind_ctftilt)

3. The pytorch-implementation of cisTEM-simulate is here: https://github.com/homurachan/cisTEM_simulate_pytorch
# Advantage

Using pytorch means the software runs on GPUs. When running on single RTX 4090, unbend is equal to about 30 CPU cores (half of the time is spent on decompress the tiff files),
and ctftilt is equal to about 120 CPU cores.

The pytorch-ctffind is roughly the same speed as [GCTF](https://www.sciencedirect.com/science/article/pii/S1047847715301003), and can output one relion 3.1 star-file, just like running the `relion_ctffind_runner`

# Disadvantage

The pytorch-unbend excludes less patches than the old version. But the final results don't have much differences.

The pytorch-ctftilt is not stable as the original version. To be fixed soon.

# Usage

## Unbend-pytorch

`python unbend_pytorch.py input.tif input_sum_doseweighted.mrc --pixel-size $pixel_size --output-binning 2 --device cuda --exposure-per-frame $dose_per_frame  --voltage 300 --gain gain.mrc`

The gain file must be converte into mrc format. Because reading dm4 in python requires some weird libs. All the patch splitting and spline fitting are done automatically. You can check more options by running `python unbend_pytorch.py --help`

## ctffind_ctftilt-pytorch

`python ctffind_ctftilt_pytorch.py "Your_*_sum_doseweighted.mrc" --output micrographs_ctf.star  --pixel-size $pixel_size --voltage 300 --cs 2.7  --box-size 512 --amplitude-contrast 0.07 --min-resolution 30 --max-resolution 5 --min-defocus 3000 --max-defocus 50000 --defocus-step 100 --no-diagnostic-output --preprocess-batch-size 4 --fit-batch-size 64 --optimizer-check-interval 8`

If you want the .ctf outputs, remove --no-diagnostic-output

For ctftilt:

`python ctffind_ctftilt_pytorch.py tilt_micrograph.mrc --output tilt_micrograph_ctf.star  --pixel-size $pixel_size --voltage 300 --cs 1.6 --box-size 512 --amplitude-contrast 0.07 --min-resolution 30 --max-resolution 5 --min-defocus 3000 --max-defocus 50000 --defocus-step 100 --no-diagnostic-output --preprocess-batch-size 4 --fit-batch-size 64 --optimizer-check-interval 8 --fit-tilt --tilt-tile-size 128 --tilt-min-global-cc 0.0 --tilt-angle $tilt_angle --tilt-angle-uncertainty 10 --tilt-candidate-batch-size 256`

There are also many options. Run `python ctffind_ctftilt_pytorch.py --help` for more imformation
