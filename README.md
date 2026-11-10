# pytorch_cryoEM

pytorch implementation (by ChatGPT 5.5, 5.6, and 6) of many cryo-EM softwares. Right now all softwares come from the Grigorieff Lab.

## Required libraries

pytorch, numpy, mrcfile, tifffile, imagecodecs, matplotlib

For unbend_pytorch_v2, nvidia-nvimgcodec-cu12[nvtiff], and nvidia-nvjpeg2k-cu12 is needed.

# Contents

1. The pytorch-implementation of Unbend: ([Lingli Kong](https://linglikong.github.io/) et al., https://elifesciences.org/reviewed-preprints/109119v2)

   [unbend_pytorch_v2.py](https://github.com/homurachan/pytorch_cryoEM/blob/main/unbend_pytorch_v2.py) 

2. The pytorch-implementation of [ctffind 4.1.8](https://grigoriefflab.umassmed.edu/ctffind4) and [ctftilt](https://grigoriefflab.umassmed.edu/ctf_estimation_ctffind_ctftilt)

   [ctffind_ctftilt_pytorch.py](https://github.com/homurachan/pytorch_cryoEM/blob/main/ctffind_ctftilt_pytorch.py)
   
3. The pytorch-implementation of [ctffind 5](https://elifesciences.org/reviewed-preprints/97227v2)

   [ctffind5_pytorch.py](https://github.com/homurachan/pytorch_cryoEM/blob/main/ctffind5_pytorch.py)

6. The pytorch-implementation of cisTEM-simulate is here: https://github.com/homurachan/cisTEM_simulate_pytorch

7. The pytorch-implementation of [SIRM-RELION](https://github.com/homurachan/SIRM_RELION)

   [relion_torch_v0.7.2rc6.zip](https://github.com/homurachan/pytorch_cryoEM/blob/main/relion_torch_v0.7.2rc6.zip)

## Update 20261106

Upload the relion_torch: [relion_torch_v0.7.2rc6.zip](https://github.com/homurachan/pytorch_cryoEM/blob/main/relion_torch_v0.7.2rc6.zip).

Requirements: pytorch, numpy, mrcfile, starfile, pandas, scipy.

This version has the following features:

1. Autopicking using LoG filters
2. 2D classification, with/skip alignment
3. 3D classification, with/skip alignment
4. 3D auto refine, does not require multiple GPUs
5. 3D reconstruction using GPUs
6. SIRM refinement and reconstruction
7. CTF refine
8. Joint orientation, translation, defocus refine. See the supplemental in [this paper](https://www.science.org/doi/10.1126/science.aan0327)

This version does NOT have the following features:

1. Running Motion Correction or CTFFIND (Replaced by [unbend_pytorch_v2.py](https://github.com/homurachan/pytorch_cryoEM/blob/main/unbend_pytorch_v2.py) and [ctffind4_pytorch_no_cudnn.py](https://github.com/homurachan/pytorch_cryoEM/blob/main/ctffind4_pytorch_no_cudnn.py))
2. relion_display (Replaced by [v2lite](https://github.com/homurachan/pytorch_cryoEM/blob/main/v2lite.py) and [show_star_particles.py](https://github.com/homurachan/pytorch_cryoEM/blob/main/show_star_particles.py))
3. Autopicking using 3D models (Replaced by [template matching](https://github.com/homurachan/cistem2dtm_pytorch))
4. Particle Extraction (Replaced by [read_star_shift_crop_and_generate_new_star_v5.py](https://github.com/homurachan/Block-based-recontruction/blob/master/read_star_shift_crop_and_generate_new_star_v5_no_starfile.py))
5. Subset Selection (will be replaced by standalone program)
6. 3D Initial Model (No plan)
7. 3D Multibody Refinement (No plan)
8. Bayesian Polishing (in progress)
9. Mask Creation ([relion_mask_create.py](https://github.com/homurachan/pytorch_cryoEM/blob/main/relion_mask_create.py))
10. Particle Subtraction (will be replaced by standalone program)
11. Post-processing ([relion_postprocess.py](https://github.com/homurachan/pytorch_cryoEM/blob/main/relion_postprocess.py))
12. Local Resolution (Replaced by [resmap_pytorch_py310_complete_2.0.0.zip](https://github.com/homurachan/pytorch_cryoEM/blob/main/resmap_pytorch_py310_complete_2.0.0.zip))
13. relion_star_handler ([relion_star_handler.py](https://github.com/homurachan/pytorch_cryoEM/blob/main/relion_star_handler.py))
14. relion_image_handler ([relion_image_handler.py](https://github.com/homurachan/pytorch_cryoEM/blob/main/relion_image_handler.py))

Usage: Basically the same as the original RELION. However, the continue run is done by --continue Class3D/job001/run_it025_state.pt

Speed: Class3D: --healpix_order 2 is slow (7.5 deg with --oversampling 1), but --healpix_order 3 (3.75 deg with --oversampling 1) and above is much faster.

A GUI will be uploaded soon.

## Update 20261013

Update the [ctffind_ctftilt_pytorch.py](https://github.com/homurachan/pytorch_cryoEM/blob/main/ctffind_ctftilt_pytorch.py) , so it can read the RELION corrected_micrographs.star
from Motion Correction

Usage: `python ctffind_ctftilt_pytorch.py --input_star corrected_micrographs.star --output micrographs_ctf.star --no-diagnostic-output --preprocess-batch-size 32 --fit-batch-size 64 --optimizer-check-interval 8`

Remove `--no-diagnostic-output` if you want the .ctf output. Reduce `--preprocess-batch-size` and `--fit-batch-size` if it runs out of memory.

Upload the [ctffind4_pytorch_no_cudnn.py](https://github.com/homurachan/pytorch_cryoEM/blob/main/ctffind4_pytorch_no_cudnn.py) . This version removes the ctftilt feature, the need for scipy, and the requirement for cudnn. Usage is the same as the ctffind_ctftilt_pytorch.py.

## Update 20260915

Upload [resmap_pytorch_py310_complete_2.0.0.zip](https://github.com/homurachan/pytorch_cryoEM/blob/main/resmap_pytorch_py310_complete_2.0.0.zip) . This one is the pytorch version of [resmap](https://resmap.sourceforge.net/) and [monores](https://www.sciencedirect.com/science/article/pii/S0969212617304434).

Upload [blocfilt_resmap_pytorch.py](https://github.com/homurachan/pytorch_cryoEM/blob/main/blocfilt_resmap_pytorch.py) . This program is the pytorch version of blocfilt from [bsoft](https://cbiit.github.io/Bsoft), which reads the output of resmap/monores and does localized filtering.

Upload [v2lite.py](https://github.com/homurachan/pytorch_cryoEM/blob/main/v2lite.py) . This is a simple version of [EMAN](https://www.sciencedirect.com/science/article/pii/S1047847799941746?via%3Dihub) package v2. It requires python3-tk to be installed.

Upload [show_star_particles.py](https://github.com/homurachan/pytorch_cryoEM/blob/main/show_star_particles.py) . This is a simple version of relion_display. It reads coordinates from relion star file or 2dtm tsv files, and draw green circles accordingly.

## Update 20260826

Upload ctffind5_pytorch.py. It is a pytorch implement of ctffind 5. The --fit-tilt is much more stable than the ctftilt implementation. However, the untilted defocus estimation is not as fast as the ctffind 4.1.8 as the EPA and ice-thickness need more optimization.

Example usage:
`python ctffind5_pytorch.py "Your_*_sum_doseweighted.mrc" --output micrographs_ctf.star  --pixel-size $pixel_size --voltage 300 --cs 2.7  --box-size 512 --amplitude-contrast 0.07 --min-resolution 30 --max-resolution 5 --min-defocus 3000 --max-defocus 50000 --defocus-step 100 --output-star micrographs_ctf.star  --output-tsv micrographs_ctf.tsv --output-ctffind micrographs_ctf.txt  --output-avrot micrographs_avrot.txt --fit-tilt --estimate-thickness --gpu-ids all`

If you don't need to fit the tilted micrographs, remove --fit-tilt.

Upload the multi-GPU supported unbend_pytorch_v2.py.

Example usage:

`python unbend_pytorch_v2.py "*.tif" corr_folder --pixel-size $pixel_size --output-binning 2 --gpu-ids 0,1,2,3,4,5,6,7 --exposure-per-frame $dose_per_frame  --voltage 300 --gain gain.mrc`

A folder named corr_folder will be created, and the corrected micrographs and corresponding png files will be stored there.

## Update 20260825

Upload unbend_pytorch_v2.py, which utilize nvtiff and nvimgcodec to decompress the LZW tiff files.

# Advantage

Using pytorch means the software runs on GPUs. When running on single RTX 4090, unbend is equal to about 30 CPU cores (half of the time is spent on decompress the tiff files),
and ctftilt is equal to about 120 CPU cores.

The pytorch-ctffind is roughly the same speed as [GCTF](https://www.sciencedirect.com/science/article/pii/S1047847715301003), and can output one relion 3.1 star-file, just like running the `relion_ctffind_runner`

# Disadvantage

The pytorch-unbend excludes less patches than the old version. But the final results don't have much differences.

The pytorch-ctftilt is not stable as the original version. Don't use --fit-tilt for ctffind_ctftilt_pytorch.py.

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
