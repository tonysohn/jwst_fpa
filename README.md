# JWST Focal Plane Alignment Workflow
This repository contains codes and scripts that can be used for performing the **Geometric Distortion Correction** and the **FGS-SI Alignment Calibration** for JWST imaging detectors. Currently supported detectors are FGS, NIRCam, and NIRISS. Support for NIRSpec and MIRI may be included in a future version.

## Installation
This package has some external dependencies and requires anaconda to be pre-installed. Assuming anaconda is installed in your system, follow the simple two-step instructions below.
- Clone the `jwst_fpa` repository:

  ```git clone https://github.com/tonysohn/jwst_fpa.git```

- Create a dedicated environment name `fpa` with the necessary dependencies:

  ```conda env create -f jwst_fpa/environment.yml -n fpa```

- All necessary packages should have been installed by doing above. In some cases, importing `pysiaf` may result in error. If this happens, uninstall and reinstall `lxml` as follows:

  ```pip uninstall lxml```

  ```pip install lxml```

## Usage

There are two main scripts and several supplemental codes/scripts included in this package. Below are quick descriptions for how to run the two main scripts. 

- `jwst_distortion.py` - This script determines the geometric distortion solution for the JWST imaging detectors. The only essential configuration parameters required would be the data directory and whether the PSF is a nominal one or commissioning version (for OTE programs only, i.e., right after global alignment). Once the `data_dir` points to where the `_cal.fits` files are, open an `ipython` session and run the script: `run jwst_distortion`
- `jwst_fpa.py` - This is the script for the focal plane alignment workflow. This script has a separate configuration file named `jwst_fpa_config.py`. All configurations including the data directory should be specified in this file. Once the configuration file is ready, open an `ipython` session and run the script: `run jwst_fpa`