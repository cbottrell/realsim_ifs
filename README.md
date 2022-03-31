![Logo](Figures/IFS/Logo.png)
RealSim-IFS is a Python tool for generating survey-realistic integral-field spectroscopy (IFS) obvervations of galaxies from numerical simulations of galaxy formation. The tool is designed primarily to emulate current and experimental observing strategies for IFS galaxy surveys in astronomy. RealSim-IFS has built-in functions supporting SAMI and MaNGA IFU footprints, but supports any fiber-based IFU design, in general.

Full documentation (and justification) for the code is provided in Chapter 4 of my [PhD Thesis](https://dspace.library.uvic.ca/bitstream/handle/1828/11975/Bottrell_Connor_PhD_2020.pdf?sequence=5&isAllowed=y). 

## Installation

RealSim-IFS is not a pypy package. The github repository but be cloned to your Python environment's path. By convention, it should be placed in the `site-packages` directory of your Python environment. For example, with a conda environment active:
```
conda activate my_env
cd $CONDA_PREFIX/lib/pythonX.x/site-packages
git clone https://github.com/cbottrell/realsim_ifs.git
```
Where X.x is the Python version for the environment. That's it!

## Tutorial

A Jupyter notebook [tutorial](https://github.com/cbottrell/realsim_ifs/blob/master/Tutorials/IFS/Tutorial.ipynb) using RealSim-IFS to make [SDSS-IV MaNGA](https://www.sdss.org/instruments/) stellar kinematics for a [TNG50-1](https://www.tng-project.org/) disk galaxy is included with the package (along with the input datacube). It can be found here: `Tutorials/IFS/Tutorial.ipynb`.

## Precision

RealSim-IFS can reproduce both the flux and variance propagation of real galaxy spectra to cubes. The [precision](https://github.com/cbottrell/realsim_ifs/blob/master/Tutorials/IFS/Precision.ipynb) notebook uses real calibrated fiber spectra from the MaNGA survey to generate flux and variance cubes that are identical to those produced by the MaNGA data reduction pipeline.
