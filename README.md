# Multimorbid patient vignettes with reference answers

Sixty-five synthetic vignettes of older adults with several chronic conditions, each with reference answers for its current drugs and for proposed changes to its regimen. They were written to evaluate how large language models plan and appraise medication changes.

## Files

- `vignettes.json`: one entry per vignette.
  - `id`: the vignette's label.
  - `record`: the patient text exactly as given to the models (patient, function and frailty, active problems, current medications, recent laboratory values and vital signs).
  - `current_drugs`: each current drug as listed in the record (`id` as used in the questions put to the models; `name`, `dose`, `frequency`), with its reference answer: `stop`, `reduce dose`, `increase dose` or `null` (no change marked; the paper calls these unmarked drugs).
  - `proposed_changes`: each proposed change, worded as it was put to the models (`id` as used in the questions), with its reference answer (`yes` or `no`).
- `nhanes_participants.csv`: the NHANES 2017–2018 respondent sequence numbers (SEQN) of the 65 participants used as a second set of patients. The underlying public-use files are available from the National Center for Health Statistics.
- `usual_doses.csv`: the usual adult maintenance doses attached to NHANES medications, which NHANES records without strength (225 drugs).
- `code/`: the analysis code, with its own README. Model responses are not included.

## How the vignettes were made

The vignettes are built around eight comorbidity clusters adapted from published multimorbidity patterns (Barnett et al., Lancet 2012; Violán et al., PLoS ONE 2014; Prados-Torres et al., J Clin Epidemiol 2014). Each regimen combined guideline-standard therapy for the patient's conditions, and the regimens contain prescribing problems by construction. Every reference answer cites a published criterion, such as STOPP/START version 3 or the 2023 Beers Criteria. Six physicians validated the reference answers, and three physicians rated the realism of each vignette.

## Reference answers

- Current drugs: 495 in all; 137 to stop or reduce (80 `stop`, 57 `reduce dose`; either action counts as correct for both), 10 to increase, 348 unmarked.
- Proposed changes: 250 in all; 61 recommended (`yes`) and 189 harmful (`no`).

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0).

## Citation

Please cite the accompanying article.
