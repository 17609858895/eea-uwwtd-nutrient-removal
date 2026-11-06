# EEA UWWTD nutrient-removal analysis

Core analysis code for **Treatment-state decomposition revises nutrient-removal gains in Europe's wastewater registry**.

The code estimates treatment contrasts from the public European treatment-plant registry, decomposes influent pass-through into treatment states, and evaluates sensitivity, heterogeneity, upgrade scenarios and earlier reporting rounds. This repository contains analysis source and lightweight checks. Raw data, fitted models, plant-level predictions, manuscripts and publication-layout tools are not distributed here.

## Environment

The analysis environment used Python 3.12.3. Create a separate Python environment and install the pinned dependencies:

```bash
python -m venv .venv
# Activate the environment using the command appropriate to your operating system.
python -m pip install -r requirements.txt
python run_analysis.py --help
python checks/check_core.py
```

Package versions are pinned to the actual analysis environment. Native numerical libraries, operating systems and parallel execution can affect exact numerical reproducibility. Full nuisance fitting, nested prediction benchmarking and country bootstraps are computationally intensive; the included checks do not fit models.

## Official data

Download the public final treatment-plant spatial release for UWWTD data call 2023 (reference year 2022), release filename `EEA_UWWTD_PUBLIC_FINAL_2022-2023_v10_r00.zip`:

- [Official EEA record and DOI](https://doi.org/10.2909/88b3b08d-3b1b-4a65-9406-365b218e09da)
- [Official download directory](https://sdi.eea.europa.eu/data/88b3b08d-3b1b-4a65-9406-365b218e09da)
- [Reference-2022 attribute dictionary](https://dd.eionet.europa.eu/datasets/3539)

The ZIP used in the study has SHA-256 `d1d75c484c2a3209818c8803d7d3ced4e78c5c2552b79791803a2b6491131017` and contains the `UWWTD_TreatmentPlants` GeoPackage layer (22,554 rows). Use this frozen release rather than a changing latest dictionary or a different reporting round. The source data are attributed to the European Environment Agency under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); data licensing is independent of this code repository.

Read the ZIP in place and create the analysis tables inside this repository:

```bash
python run_analysis.py prepare --input-zip /path/to/EEA_UWWTD_PUBLIC_FINAL_2022-2023_v10_r00.zip
```

For the longitudinal check, download the public earlier releases and place their `UWWTD_TreatmentPlants.gpkg` files at the following paths:

| Reference year | Official catalogue record | Local path |
|---|---|---|
| 2018 | [Data call 2019, public version January 2022](https://sdi.eea.europa.eu/catalogue/srv/api/records/53f1d42c-6dbb-4515-ae88-32888d0ba03e) | `raw/earlier_releases/dc2019_public/UWWTD_TreatmentPlants.gpkg` |
| 2020 | [Data call 2021, final public version September 2024](https://sdi.eea.europa.eu/catalogue/srv/api/records/4972decb-6f56-4bfa-9779-9ffc149686f5) | `raw/earlier_releases/dc2021_final_public/UWWTD_TreatmentPlants.gpkg` |

## Run order

All commands run from the repository root. `--dry-run` prints a stage's commands without reading data or fitting models. Generated tables and predictions are written under `analysis/results/` and excluded from Git.

| Stage | Command | Prerequisites and scientific output |
|---|---|---|
| Data preparation | `python run_analysis.py prepare --input-zip /path/to/release.zip` | Official 2022 ZIP; selected label source and raw/clean analysis tables |
| Reference nuisances | `python run_analysis.py support` | Prepared data; three-repeat intermediate nuisance results used by simulation and sensitivity stages; these are **not the ten-repeat primary estimates** |
| Primary contrasts | `python run_analysis.py main` | Prepared data; ten-repeat LightGBM cross-fitting, median-rule contrasts and averaged unit scores |
| Treatment states | `python run_analysis.py decomposition` | Prepared data; five-repeat pass-through arm scores on the separately defined influent-reporting support |
| Influent checks | `python run_analysis.py influent` | Prepared data; alternative outcomes and influent specifications |
| Selective reporting | `python run_analysis.py reporting` | `support`; one-sided unobserved-outcome shifts |
| Treatment permutation | `python run_analysis.py placebo` | Prepared data; permuted-treatment checks by reporting region |
| Latent confounding | `python run_analysis.py confounding` | `support`; partially linear reporting-sample diagnostic and covariate benchmarks |
| Heterogeneity/scenario | `python run_analysis.py scenario` | `main`; supported/transported CATE, tonnage rankings, capacity targets and 200 country resamples |
| Nuisance alternatives | `python run_analysis.py learner` | `main`; three-repeat super learner and common-population comparison |
| Semi-synthetic | `python run_analysis.py simulation` | `support`; N-covariate simulations, 20 replicates per scenario |
| Prediction benchmark | `python run_analysis.py benchmark` | Prepared data; 21 base learners and 3 hybrid/ensemble configurations, random and leave-country-out validation |
| Earlier rounds | `python run_analysis.py longitudinal` | Earlier GeoPackages and `scenario`; linked-round treatment flags and within-plant comparisons |

The source filenames retain their analysis-stage numbering so that intermediate outputs match their consumers. Some analysis scripts execute work at module level: use the runner rather than importing stage scripts. `common.py` and `causal_core.py` contain shared feature and estimator functions.

## Interpretation

The primary estimand is an overlap-weighted contrast on joint treatment/reporting support, on a source-standardised log10 per-p.e. discharge scale. The influent decomposition uses its own reporting subset and support. Capacity-defined scenarios cover 30 registry countries (EU27, Iceland, Norway and Switzerland); they are not a literal implementation of the recast directive. Source standardisation needs its own model assumptions. Scenario bootstrap intervals condition on nuisance estimates, support and baseline predictions. Relative-effect rankings and load-weighted tonnage rankings answer different questions. Negative scenario predictions are retained in totals.

## Verification performed for this release

All included Python files were syntax-compiled. The stage selector's help and dry-run paths were checked. Algebra checks cover strict support boundaries, missing-outcome score augmentation and country-clustered weighted standard errors. No full analysis was refitted when preparing this release; reproducing the study requires the official inputs and the full commands above. The release does not claim independent revalidation of the observational identification assumptions.
