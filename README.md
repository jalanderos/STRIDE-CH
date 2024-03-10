# STRIDE-CH 1.0

Ensemble Image Segmentation of Coronal Holes in Sub-Transition Region Observations (J. Landeros, in prep)

Online Poster: https://agu23.ipostersessions.com/Default.aspx?s=71-5A-00-D2-CF-A7-05-F8-21-71-63-3B-D7-93-43-6D

## Abstract

Coronal Holes (CH) are large-scale, low-density regions in the solar atmosphere which may expel high-speed solar wind streams that incite hazardous, geomagnetic storms. Segmentation of CH boundaries can aid in validating the predictive performance of coronal and solar wind models, but doing so accurately has proved challenging due to similar appearances as filaments in Extreme Ultraviolet (EUV) imagery, the tendency for dense coronal plasmas to obscure underlying CHs, and the lack of ground truth. We propose a method named Sub-Transition Region Identification of Ensemble Coronal Holes (STRIDE-CH) which revisits ground-based, chromospheric He I 10830 Å line imagery and underlying Fe I photospheric magnetograms as a means to disambiguate CHs from polarity inversion lines, account for coronal loop obscuration, and provide a complementary method to the established community methods using on space-borne, coronal EUV observations. Classical computer vision techniques are applied to imbue STRIDE-CH with design variables based in radiative and magnetic properties of CHs, as well as produce an ensemble of boundaries with quantified intra-algorithm uncertainty. This method is science-enabling towards future studies of CH formation and variability from a mid-atmospheric perspective.

## Data

Tested data includes observations from the Kitt Peak Vacuum Telescope (KPVT) spectromagnetograph and Synoptic Optical Long-term Investigations of the Sun (SOLIS) Vector spectromagnetograph (VSM).

KPVT Data: https://nispdata.nso.edu/ftp/kpvt/daily/raw/

SOLIS Data: https://solis.nso.edu/0/vsm/VSMDataSearch.php?stime=1059717600&etime=1701647999&thumbs=0&pagesize=150&obsmode[]=1083i&sobsmode=1&sobstype=&display=1

## Code Structure

1. `STRIDE-CH.ipynb`: Notebook for producing STRIDE-CH data products.
2. `settings.py`: Script for declaration of data locations in file system.
3. `prepare_data.py`: Library of functions to prepare observations for CH detection.
4. `detect.py`: Library of functions to detect CHs.
5. `plot_detection.py`: Library of functions to plot observations and data products.
6. `Exploration.ipynb`: Notebook for exploratory development of STRIDE-CH. This notebook is not polished, but made visible for the sake of completeness.

## Dependencies

The following dependencies do not include packages required for executing and debugging Jupyter notebooks. Please see the Contact section for the email to write to regarding help in this or any other area. 

If you wish to create an environment to better manage Python packages and dependencies, an installation of anaconda or miniconda will be needed. Once installed run the following commands:
```
conda create --name stride-ch python=3.11.0
conda activate stride-ch
```
The following dependency installation command may be executed with or without having created a conda enviroment with the above steps.
```
pip install -r requirements.txt
```

## Contact

Jaime A. Landeros, Cal Poly Pomona Aerospace Engineering Undergraduate Student as of Spring 2024. This project is associated with a NASA Goddard Solar Physics Internship.
- Email: See jalanderos Github account
