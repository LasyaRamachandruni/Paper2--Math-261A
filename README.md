# Predicting County-Level Diabetes Prevalence with CDC PLACES Data

Final paper for MATH 261A (Regression Theory and Methods).

Can five community health indicators predict how common diabetes is in a U.S. county? Using 2,957 counties from the CDC PLACES dataset, this project compares three regression models:

| Model | Idea |
|---|---|
| Ordinary least squares | Linear baseline on all five predictors |
| Cubic B-spline | Lets the effect of obesity bend instead of forcing a straight line |
| Lasso | Shrinks coefficients to check which predictors matter |

**Predictors:** obesity, physical inactivity, high blood pressure, current smoking and routine checkup rates (crude prevalence).

## Findings

- The **spline model predicts best** (RMSE 0.902 on validation, 0.905 on test).
- The relationship between obesity and diabetes prevalence is **U-shaped**, which a linear model misses.
- **Physical inactivity** and **high blood pressure** are the strongest positive predictors; higher **checkup** rates go with lower diabetes prevalence.

The full write-up, including diagnostics and model comparison, is in [`paper/paper.qmd`](paper/paper.qmd).

## Data

[CDC PLACES: Local Data for Better Health](https://www.cdc.gov/places/), county-level estimates, saved as `data/places_local_data_2025.csv`. PLACES is published by the CDC as public data.

`analysis/00_clean-data.R` filters the five predictors and the diabetes outcome and reshapes the data to one row per county.

## Project layout

```
analysis/00_clean-data.R   data cleaning and reshaping
data/                      CDC PLACES extract
paper/paper.qmd            the paper (Quarto), with all modeling code
paper/references.bib       bibliography
```

## Acknowledgments

Repository structure based on the MATH 261A template, adapted from [Rohan Alexander's starter folder](https://github.com/RohanAlexander/starter_folder).

## Rendering the paper

The paper is a [Quarto](https://quarto.org) document that runs all analysis code when rendered.

```r
install.packages(c("tidyverse", "splines", "glmnet", "car", "lmtest", "kableExtra"))
install.packages("tinytex"); tinytex::install_tinytex()   # LaTeX, needed for PDF output
```

```bash
quarto render paper/paper.qmd    # writes paper/paper.pdf
```

Quarto runs the code from inside `paper/`, which is why the data is read from `../data/`.
