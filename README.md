# Magnetic-Field Constrained Ensemble Image Segmentation of Coronal Holes

[![arXiv](https://img.shields.io/badge/arXiv-2405.04731-b31b1b.svg)](https://arxiv.org/abs/2405.04731)

## Data

Tested data includes observations from the Kitt Peak Vacuum Telescope (KPVT) spectromagnetograph and Synoptic Optical Long-term Investigations of the Sun (SOLIS) Vector spectromagnetograph (VSM).

KPVT Data: https://nispdata.nso.edu/ftp/kpvt/daily/raw/

SOLIS Data: https://solis.nso.edu/0/vsm/VSMDataSearch.php?stime=1059717600&etime=1701647999&thumbs=0&pagesize=150&obsmode[]=1083i&sobsmode=1&sobstype=&display=1

## Code Structure

1. `STRIDE-CH.ipynb`: Notebook for producing STRIDE-CH data products with instructions included.
2. `settings.py`: Script for declaration of data locations in file system.
3. `prepare_data.py`: Library of functions to prepare observations for CH detection.
4. `detect.py`: Library of functions to detect CHs.
5. `plot_detection.py`: Library of functions to plot observations and data products.
6. `v1_1_LDA_model.pkl`: File from v1.1 of the algorithm of the saved Linear Discriminant Analysis (LDA) model to derive confidence in CH status for candidate segmented regions.
7. `acwe_lib`: Directory with code to read data products from the Active Contours Without Edges (ACWE) algorithm, whose code is available at https://github.com/DuckDuckPig/CH-ACWE, for producing fused ACWE-STRIDE-CH confidence maps.
8.  `Exploration.ipynb`: Notebook for exploratory development of STRIDE-CH. This notebook is not polished, but made visible for the sake of completeness.

## Dependencies

### Conda Environment (Optional)

An environment can help to better manage Python packages and dependencies. An installation of anaconda or miniconda will be needed. Once installed run the following commands:
```
conda create --name stride-ch python=3.11.0
conda activate stride-ch
```

### Python Dependency Installation (Required)

```
cd STRIDE-CH
pip install -r requirements.txt
```

The `moviepy` package makes video data products and requires `ffmpeg`. If it is not already installed on your machine, it may be installed on linux or mac with Homebrew:
```
brew install ffmpeg
```

Homebrew may be installed from https://brew.sh/. Check where `ffmpeg` is installed with:
```
whereis ffmpeg
```
Then copy the output of this into either python notebook as the value of the `os.environ['IMAGEIO_FFMPEG_EXE'] = VALUE` line.

#### For Development

```
pip install -r dev_requirements.txt
```

## Contact

Jaime A. Landeros. This project is associated with a 2021-2024 internship under the NASA Goddard Solar Physics Division.
- Email: See jalanderos Github account
