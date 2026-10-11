# F1 Race-Winner Predictor: 2026 Bahrain Grand Prix (held at Sepang, Malaysia)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khalilosss11/f1_Bahrain_GP_2026_predictor/blob/main/Bahrain_GP_2026_predictor.ipynb)

A machine-learning model that estimates each driver's chance of winning the 2026 Bahrain Grand Prix from qualifying, recent form, circuit history and championship standing.

**Context:** the race kept the Bahrain Grand Prix name but was moved to the **Sepang International Circuit in Malaysia** (2-4 October 2026) because of the regional situation. Circuit history is therefore built from Sepang's old Malaysian GP results, not from Bahrain.

## Prediction
Made before the race, using the real qualifying result (the starting grid includes Hadjar's 5-place penalty).

| Driver | Grid | Win probability |
|---|---|---|
| Verstappen | 1 | 44.0% |
| Antonelli | 3 | 18.5% |
| Russell | 7 | 8.2% |
| Hamilton | 2 | 6.5% |
| Hadjar | 8 | 3.6% |

**Outcome:** Verstappen won, with Antonelli second and Hamilton third.

**Honest reading:** Verstappen won from pole, which a simple "pole wins" rule also gets right, so the winner alone doesn't prove much. The model did rank Antonelli second. Russell, ranked third most likely to win, did not finish. A retirement can't be predicted from the inputs this model uses. This is one race, so it illustrates the approach and doesn't validate it. The held-out evaluation below is the stronger evidence.

## How it works
- **Data:** the Kaggle F1 history (1950-2024) plus FastF1 data for 2025-2026, joined on driver code, team name and race name.
- **Features:** grid and qualifying position, recent form (last 5 races, driver and team), circuit history, and championship points and position before the race. Every form feature is shifted by one race, so a race never sees its own result.
- **Model:** an XGBoost classifier with probability calibration. It is trained with a class-imbalance weight, since only 1 driver in about 20 wins each race. Probabilities are normalised so each race sums to 100%.
- **Validation:** trained on seasons before 2022 and tested on 2022-2024. The split is by season, never random, to avoid leaking future races into training.

## Evaluation (held-out 2022-2024)
| Method | Top-1 accuracy |
|---|---|
| This model | 52.4% (log loss 0.131) |
| "Always pick the pole-sitter" | 36.5% |
| Improvement | +15.9 points |

**Feature importance:** grid and qualifying position together account for about 71% of the model's decisions (grid 45%, qualifying 26%). The rest is spread across recent form, circuit history and championship standing, so the model mostly reflects the starting order, adjusted by form.

## Reproducibility
Cell 0 holds the two settings that define the race:

- `CUTOFF_DATE = '2026-10-04'`: only races before this date are used, so the prediction can't see its own result or anything after it.
- `CIRCUIT_HISTORY_NAME = 'Malaysian Grand Prix'`: Sepang's history under its old name.

The FastF1 data is stored in `fastf1_2025_2026_snapshot.csv`, so the notebook doesn't depend on the live API and results don't shift as new races are added. Re-running it should reproduce the table above, give or take rounding.

## Running it
1. Open the notebook in Colab with the badge above.
2. Download the Kaggle files `races.csv`, `results.csv`, `drivers.csv`, `constructors.csv` and `qualifying.csv` from the dataset linked below, and upload them to the Colab session.
3. Put `fastf1_2025_2026_snapshot.csv` next to the notebook, then run all cells.

The qualifying result is entered by hand in the qualifying cell (FastF1 hadn't published it yet at the time).

## Limitations
- The model is dominated by grid position, so it mostly follows the starting order.
- It has no inputs for weather, tyre strategy or reliability, and retirements aren't modelled.
- Sepang hadn't hosted a race since 2017, so circuit history is thin. Rookies and new teams (Audi, Cadillac) fall back to team and field averages.
- Older Kaggle drivers have no driver code, so those rows drop out. Training effectively uses a smaller, more recent set of races (about 6,300 rows).
- The Kaggle dataset ends in 2024 and isn't updated, so 2025-2026 come from FastF1.
- Sprint points aren't counted in the championship feature.

## Data and credits
- [Formula 1 World Championship dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) by Rohan Rao (Vopani), CC0
- [FastF1](https://github.com/theOehrly/Fast-F1)

Educational project. Not affiliated with Formula 1, and not betting advice.
