# Data notice

The files in this folder are derived from the Building Data Genome Project 2 (BDG2):

Miller, C., Kathirgamanathan, A., Picchetti, B. et al. The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition. Sci Data 7, 368 (2020). https://doi.org/10.1038/s41597-020-00712-x

Source repository: https://github.com/buds-lab/building-data-genome-project-2
Original license: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0), https://creativecommons.org/licenses/by-sa/4.0/

## Changes made

- Kept only the electricity meters of 56 buildings with primary space usage "Education" (up to 4 per site, 14 sites, chosen by random sampling with seed 0).
- Readings that are zero or missing for more than 168 consecutive hours were set to missing.
- The files in `model/` are LightGBM models trained on this modified data, plus evaluation numbers computed from them. They are shared under the same terms to be safe.

These files are shared under CC BY-SA 4.0. The source code in this repository is not covered by this notice.
