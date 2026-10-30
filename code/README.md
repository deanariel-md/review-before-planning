# Analysis code

Analysis code for the study of the knowledge–action gap of large language models in deprescribing. It reads the vignettes and their reference answers (`../vignettes.json`), the NHANES participant records, and the stored model responses, and writes every number and table in the analyses. Model responses are not distributed; they are available from the corresponding author on reasonable request.

Python 3.10 or later with `numpy`; `pandas` for `nhanes_records.py`.

## Steps

```
python nhanes_records.py --raw nhanes_raw/ --doses ../usual_doses.csv --out nhanes_records/
python assemble.py --vignettes ../vignettes.json --nhanes nhanes_records/ --responses responses/ --out rows.json
python gap.py --rows rows.json --out results
python gap.py --rows rows.json --out results --exclude-dev Anthropic
python gap.py --rows rows.json --out results --unresolved-keep
python free_checks.py --rows rows.json --out results
python free_checks.py --rows rows.json --out results --exclude-dev Anthropic
```

1. `nhanes_records.py` rebuilds the 65 NHANES participant records from the NHANES 2017–2018 public-use files and the table of usual maintenance doses.
2. `assemble.py` reads each plan and each answer and writes one row per drug decision (and per plan and proposed change). The folder layout it expects for the responses is in its docstring.
3. `gap.py` computes the gaps, disagreement, flag rates and rates against the reference answers, with the sensitivity analyses (the 19 vignettes in `realism_exclusions.json`, one developer left out, unresolved adjustments read as unchanged).
4. `free_checks.py` computes discrimination, the matched-threshold comparison and the analysis of proposed changes.

## The `gapstudy` package

- `direction.py`: how a plan's decision on a current drug is read (continue, stop, reduce or increase), with the ordered rules for the direction of a dose adjustment.
- `drugs.py`: drug-name matching between a model's free text and a medication record.
- `parsing.py`: reading the JSON in a response, and normalising option probabilities.
- `key/`: the structured key that decides whether a plan made a proposed change, one rule per proposal.
- `models.py`: the 16 models and their developers.

## Licence

MIT.
