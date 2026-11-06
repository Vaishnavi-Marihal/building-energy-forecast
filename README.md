# Building Energy Day-Ahead Forecast

Day-ahead hourly electricity forecasts for 56 school buildings, with 90% prediction ranges, evaluated by forward-in-time backtests against "same hour last week" and "same hour yesterday" baselines.

**Live demo:** https://building-energy-forecast.onrender.com
(Free hosting: the first visit after 15 minutes of inactivity can take about a minute to wake up.)

![Dashboard](docs/dashboard.png)

## Results (July–December 2017, never seen in training)

| Measure | Result |
|---|---|
| Median error ratio vs same hour last week (below 1 is better) | 0.68 |
| Buildings where the model beats last week's value | 92.9% (52 of 56) |
| True values inside the 90% range | 90.1% |

The dashboard shows this fixed-origin run: models trained through 31 May 2017, range calibrated on June 2017, scored on July to December 2017.

A separate rolling backtest (6 monthly folds, retraining each month on earlier data, calibrating on the month before) gave:

| Measure per test month | Range |
|---|---|
| Median error ratio vs same hour last week | 0.62 to 0.77 |
| Buildings beating last week's value | 85.7% to 92.9% |
| Median error ratio vs same hour yesterday | 0.61 to 0.78 |
| Buildings beating yesterday's value | 83.9% to 92.9% |
| True values inside the 90% range | 87.5% to 91.3% |

## Unseen sites

How does the model do on a site it never saw in training? For each of the 14 sites, I trained on the other 13 sites (data through 31 May 2017) and forecast the held-out site's buildings for July to December 2017. A control model trained on all 56 buildings is scored on the same hours, so the only difference between the two columns is whether the site was in the training data. Reproduce with `python src/backtest_unseen_sites.py`; results are in `docs/unseen_sites.json`.

| Measure (below 1 beats same hour last week) | Site in training | Unseen site |
|---|---|---|
| Median error ratio, by building | 0.682 | 0.701 |
| Median of the 14 site medians | 0.677 | 0.710 |
| Buildings beating last week's value | 52 of 56 | 48 of 56 |
| Sites beating last week's value (site median below 1) | 13 of 14 | 12 of 14 |

The model gets a little worse on unseen sites and still beats last week's value on most. Two sites do not beat it when unseen: Lamb (1.22, against 0.93 when seen) and Moose (1.11, against 1.00 when seen, so Moose is hard for this model even when its own data is in training). Lamb's four buildings are very small loads (about 3 to 44 kWh), which may matter, but I did not test that.

How far to trust this: each site score is the median of only 4 buildings, and I used one random seed, so site-level differences of about 0.03 or less cannot be separated from noise. An unseen site still supplies its own recent readings as model inputs and its own mean load for scaling; only the model's training excludes it. The buildings were chosen with quality checks over the whole period (see Limitations).

## What it does

Every forecast is issued at 23:00 for the next day's 24 hours. Features use only data available at that moment: the same hour 1, 2 and 7 days earlier, yesterday's mean, minimum and maximum, the last observed reading, the forecast horizon (1 to 24), weekday and month. One global LightGBM model serves all buildings, with each building's load divided by its training-period mean. The 90% range comes from 5th and 95th percentile models, widened by a split-conformal correction computed on the previous month.

## How it is tested

- A leakage test: changing any data after the 23:00 issue time does not change any feature.
- Baseline, interval and conformal tests with hand-checkable answers.
- API tests for success, unknown building (404) and bad or out-of-range dates (422).
- A database test: features built from the 8-day window read from PostgreSQL must equal the features built from the whole history, including days with masked (NULL) hours.
- 40 tests in total. The 22 that need a database run only when `TEST_DATABASE_URL` is set (the database name must end in `_test`, because the tests rebuild its tables); the other 18 always run.
- GitHub Actions starts a PostgreSQL 16 service and runs all 40 tests on Python 3.9 and 3.12, plus the frontend lint and build.

## Database

Readings live in PostgreSQL, not in memory. `sql/schema.sql` defines two tables: `buildings` (id, site, the training-period mean load used as the scale) and `readings` (one row per building per hour; a masked hour is stored as NULL). The key `(building_id, ts)` also serves as the index, so a forecast request reads one building's 192 rows (the 7 days before the target day, plus the target day) instead of all 982,464.

`src/load_db.py` fills both tables from `deploy/` and checks that, for every building, the SQL mean load before June 2017 equals the scale the model was trained with. Forecasts from this version were compared with the previous in-memory version on 280 building and date pairs, plus the error cases, and matched to within floating-point rounding (1e-9).

## Data and cleaning

Source: Building Data Genome Project 2 (1,636 buildings, hourly, 2016 to 2017). 1,578 have an electricity meter. I used the Education category (604 buildings, 15 sites). An hour counts as "dead" if the reading is zero or missing. Dead stretches longer than 168 hours were masked (not scored, not used). A building was kept if at most 5% of July–December 2017 and at most 50% of all hours were masked: 535 buildings. I then sampled up to 4 per site, giving 56 buildings from 14 sites.

## What went wrong, and what I learned

- **My first zero-run check undercounted.** It counted only exact zeros, so a missing hour split a dead stretch in two. Counting "zero or missing" doubled the number of Education buildings with a dead stretch over a week (103 to 206).
- **Healthcare was not an option.** The category exists (27 buildings with electricity, 6 sites), but only 13 passed a simple quality filter, so I chose Education.
- **Dead stretches are mostly a site problem.** Panther, Shrew, Cockatoo, Bobcat, Gator and Crow show site-wide dead periods. For Lamb and Eagle my test found none, but it could not rule out shared events affecting under half of a site's buildings, so that question stays open.
- **Weather did not help.** Observed air temperature, and temperature with simulated forecast error (1, 2 and 3 °C), all stayed within about ±0.2% of the no-weather model. I used one random seed, so differences this small cannot be separated from noise. Weather is not in the final model.
- **The raw quantile ranges under-covered** (85.6% to 90.0% per month). Conformal calibration raised that to 87.5% to 91.3%, still slightly below 90% in some months.
- **Hosting:** as of October 2026, Hugging Face Docker Spaces required a paid plan, so I used Render's free tier.

## Limitations

- This is a replay of a past period, not a live service. Dates are limited to July–December 2017 because the models were trained on earlier data. Earlier dates would be in-sample and flatter the results.
- 56 buildings from 14 sites. Buildings on one site share weather and behaviour, so the effective sample is closer to the number of sites. Held-out sites are tested above (one seed); other building types and other years are not.
- There is no holiday or school-calendar feature, only weekday and month. School breaks are a likely source of larger errors, but I have not measured this.
- The 56 buildings were chosen using data-quality checks computed over the whole period, including the test months, which favours buildings with clean test data.
- Scores use only hours where the actual value and both baseline values exist.
- The 90% guarantee of conformal prediction assumes the data behaves consistently over time, which energy data only roughly does.
- Reproducing the data step needs the BDG2 repository downloaded with git-lfs. Paths in the scripts assume it sits at `~/Documents/building-data-genome-project-2`.

## Run locally

With Docker (starts PostgreSQL, loads the data, then starts the app):

```bash
docker compose up --build
```

Then open http://127.0.0.1:8080. The loader recreates the tables on every start, which is fine for this demo database and not something to do to a real one. The database password in `compose.yaml` (`devpass`) is for local use only; override it with `POSTGRES_PASSWORD`.

Tests (need a PostgreSQL server and an empty database whose name ends in `_test`):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
TEST_DATABASE_URL=postgresql://postgres:devpass@localhost:5432/energy_test python -m pytest tests -q
```

Without `TEST_DATABASE_URL` the 22 database tests are skipped and 18 run.

## Reproduce the results

With BDG2 downloaded, run in this order: `python src/prepare_data.py`, `python src/backtest_baselines.py`, `python src/backtest_model.py`, `python src/backtest_intervals.py`, `python src/backtest_weather.py`, `python src/train_final.py`.

## Layout

`src/` Python code (cleaning, features, baselines, uncertainty, API, database loader) · `sql/` database schema · `tests/` · `notebooks/` data audit scripts · `frontend/` React + TypeScript dashboard · `deploy/` demo data, trained models and data notice · `Dockerfile` · `compose.yaml`

## Data attribution

Miller, C., Kathirgamanathan, A., Picchetti, B. et al. The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition. Sci Data 7, 368 (2020). https://doi.org/10.1038/s41597-020-00712-x
The data in `deploy/` is a modified subset, shared under CC BY-SA 4.0. See `deploy/NOTICE.md`.

## Licence

The source code is under the MIT licence (see `LICENSE`). The data and model files in `deploy/` are under CC BY-SA 4.0, as described above.