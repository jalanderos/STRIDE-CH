# Release Notes

Notable changes to this project will be documented in this file.

This format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased v1.1] - TBD

### Added

- Confidence derivation from unipolarity and smoothness features described in 3.2. Segmentation and Appendix A.2, along with a proximity to disk center feature, as aggregated by a Linear Discriminant Analysis (LDA) model
- Fusion of confidence maps with those produced by the Active Contours Without Edges (ACWE) algorithm, whose code is available at https://github.com/DuckDuckPig/CH-ACWE

### Changed

- Streamlined `STRIDE_CH.ipynb`

## [1.0] - 2024-05-08

- Described by the publication linked in `README.md`

### Changed
- Confidence redefined to lie $\in [0,1]$ rather than $\in [0,100]\%$

## Past Versions Described in Code

### v0.5.1 - 2023-11-04
- Introduction of data processing and data product development with Sunpy maps, as desribed in 3. Methods.

### v0.5 - 2023-08-02
- Unipolarity-derived confidence, as desribed in 3.2. Segmentation.

### v0.4 - 2023-06-25
- Introduction of He I pre-processing, as desribed in 3.1. Data Preparation.

### v0.3 - 2023-04-30
- Smoothness-derived confidence, as described in Appendix A.2.
### v0.2 - 2023-04-01
- Introduction of an ensemble of masks for CH detection with area-derived confidence, as described in Appendix A.2.

### v0.1 - 2023-01-31
- Baseline single mask CH detection, primarily with a threshold and morphological operations


[Unreleased v1.1]: https://github.com/jalanderos/STRIDE-CH/compare/release-v1.0...main
[1.0]: https://github.com/jalanderos/STRIDE-CH/tree/release-v1.0