"""Build the 65 NHANES participant records from the NHANES 2017-2018 public-use files.

Real: the medication list, the reason recorded for each prescription, the conditions, the measured blood
pressure and body measures, and the laboratory values. Not real: the dose. NHANES records the drug but not
its strength, so strengths come from usual_doses.csv. Goals of care are not recorded by the survey and are
left unspecified, in the same sentence for every record.

    python nhanes_records.py --raw nhanes_raw/ --doses usual_doses.csv --out nhanes_records/
The raw folder holds the .XPT files: DEMO_J, RXQ_RX_J, BPQ_J, DIQ_J, MCQ_J, KIQ_U_J, PFQ_J, BPX_J, BMX_J,
BIOPRO_J, CBC_J, GHB_J, TCHOL_J, HDL_J, TRIGLY_J, ALB_CR_J and FERTIN_J, from the National Center for
Health Statistics. Participants are taken in a seeded random order from those aged 65 or over with five or
more prescription medications, skipping those with fewer than three recorded conditions, until 65 are
included; nhanes_participants.csv lists the 65 SEQNs.
"""
from __future__ import annotations
import argparse, csv, json, math, random, sys
from pathlib import Path
RAW = None

GOALS = ("Goals of care were not recorded in the source survey and are deliberately left "
         "unspecified.")

# questionnaire item -> (condition name, icd10, file)
CONDITIONS = [
    ('BPQ020',  'Hypertension',                     'I10',    'BPQ_J'),
    ('BPQ080',  'Hyperlipidaemia',                  'E78.5',  'BPQ_J'),
    ('DIQ010',  'Type 2 diabetes mellitus',         'E11.9',  'DIQ_J'),
    ('MCQ160B', 'Congestive heart failure',         'I50.9',  'MCQ_J'),
    ('MCQ160C', 'Coronary heart disease',           'I25.10', 'MCQ_J'),
    ('MCQ160D', 'Angina pectoris',                  'I20.9',  'MCQ_J'),
    ('MCQ160E', 'Prior myocardial infarction',      'I25.2',  'MCQ_J'),
    ('MCQ160F', 'Prior stroke',                     'I69.30', 'MCQ_J'),
    ('MCQ160G', 'Emphysema',                        'J43.9',  'MCQ_J'),
    ('MCQ160K', 'Chronic bronchitis',               'J42',    'MCQ_J'),
    ('MCQ160L', 'Liver condition',                  'K76.9',  'MCQ_J'),
    ('MCQ160A', 'Arthritis',                        'M19.90', 'MCQ_J'),
    ('MCQ160M', 'Thyroid problem',                  'E03.9',  'MCQ_J'),
    ('MCQ220',  'Malignancy, any site',             'C80.1',  'MCQ_J'),
    ('KIQ022',  'Weak or failing kidneys',          'N18.9',  'KIQ_U_J'),
]

LABS = [  # (file, variable, label, unit, formatter precision)
    ('BIOPRO_J', 'LBXSCR',  'Serum creatinine',      'mg/dL',        2),
    ('BIOPRO_J', 'LBXSKSI', 'Potassium',             'mmol/L',       1),
    ('BIOPRO_J', 'LBXSAL',  'Albumin',               'g/dL',         1),
    ('BIOPRO_J', 'LBXSATSI','ALT',                   'U/L',          0),
    ('CBC_J',    'LBXHGB',  'Haemoglobin',           'g/dL',         1),
    ('GHB_J',    'LBXGH',   'HbA1c',                 '%',            1),
    ('TCHOL_J',  'LBXTC',   'Total cholesterol',     'mg/dL',        0),
    ('HDL_J',    'LBDHDD',  'HDL cholesterol',       'mg/dL',        0),
    ('TRIGLY_J', 'LBDLDL',  'LDL cholesterol',       'mg/dL',        0),
    ('ALB_CR_J', 'URDACT',  'Urine albumin/creatinine ratio', 'mg/g', 1),
    ('FERTIN_J', 'LBXFER',  'Ferritin',              'ng/mL',        0),
]


def read_xpt(name):
    import pandas as pd
    p = Path(RAW) / f'{name}.XPT'
    if not p.exists():
        p = Path(RAW) / f'{name}.xpt'
    if not p.exists():
        return None
    df = pd.read_sas(p, format='xport')
    if 'SEQN' in df.columns:
        df['SEQN'] = df['SEQN'].astype('int64')
    return df


def s(v):
    """bytes -> str, NaN -> ''."""
    if v is None:
        return ''
    if isinstance(v, bytes):
        return v.decode('utf-8', 'ignore').strip()
    try:
        if isinstance(v, float) and math.isnan(v):
            return ''
    except Exception:
        pass
    return str(v).strip()


def egfr_ckdepi_2021(scr, age, female):
    """CKD-EPI 2021 creatinine equation, without a race coefficient."""
    if not scr or scr <= 0:
        return None
    k = 0.7 if female else 0.9
    a = -0.241 if female else -0.302
    e = (142 * (min(scr / k, 1) ** a) * (max(scr / k, 1) ** -1.200)
         * (0.9938 ** age) * (1.012 if female else 1.0))
    return round(e, 1)


def ckd_stage(e):
    if e is None: return None
    for lim, lab in ((15, 'CKD G5'), (30, 'CKD G4'), (45, 'CKD G3b'),
                     (60, 'CKD G3a'), (90, 'CKD G2')):
        if e < lim:
            return lab
    return 'CKD G1 / normal'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=65)
    ap.add_argument('--seed', type=int, default=20260821)
    ap.add_argument('--min-drugs', type=int, default=5)
    ap.add_argument('--min-age', type=int, default=65)
    ap.add_argument('--min-conditions', type=int, default=3)
    ap.add_argument('--raw', default='nhanes_raw'); ap.add_argument('--doses', default='usual_doses.csv')
    ap.add_argument('--out', default='nhanes_records')
    a = ap.parse_args()
    global RAW
    RAW = Path(a.raw); OUT = Path(a.out)
    if not RAW.exists() or not any(RAW.iterdir()):
        print(f"! No NHANES files in {RAW}.", file=sys.stderr)
        return 1
    try:
        import pandas  # noqa: F401
    except ImportError:
        print("! pandas is required to read XPT files.", file=sys.stderr)
        return 1


    with open(a.doses, newline='', encoding='utf-8') as f:   # table order matters: the first matching prefix is used
        doses = {r['drug'].lower(): {'dose': r['dose'], 'frequency': r['frequency']} for r in csv.DictReader(f)}

    demo = read_xpt('DEMO_J')
    rx = read_xpt('RXQ_RX_J')
    if demo is None or rx is None:
        print("! DEMO_J and RXQ_RX_J are required.", file=sys.stderr)
        return 1

    frames = {n: read_xpt(n) for n in
              {f for _, _, _, f in CONDITIONS} | {f for f, _, _, _, _ in LABS} | {'BPX_J', 'BMX_J', 'PFQ_J'}}

    def lookup(fname, seqn, var):
        df = frames.get(fname)
        if df is None or var not in df.columns:
            return None
        row = df.loc[df['SEQN'] == seqn, var]
        if row.empty:
            return None
        v = row.iloc[0]
        try:
            return None if (isinstance(v, float) and math.isnan(v)) else float(v)
        except Exception:
            return None

    # candidate participants
    elig = demo[(demo['RIDAGEYR'] >= a.min_age)][['SEQN', 'RIDAGEYR', 'RIAGENDR']]
    rxu = rx[rx['RXDUSE'] == 1] if 'RXDUSE' in rx.columns else rx
    counts = rxu.groupby('SEQN').size()
    keep = [int(x) for x in elig['SEQN'] if counts.get(int(x), 0) >= a.min_drugs]

    rng = random.Random(a.seed)
    rng.shuffle(keep)

    OUT.mkdir(parents=True, exist_ok=True)
    made, skipped_conditions, no_dose, all_drugs = [], 0, 0, 0
    for seqn in keep:
        if len(made) >= a.n:
            break
        d = demo.loc[demo['SEQN'] == seqn].iloc[0]
        age, female = int(d['RIDAGEYR']), int(d['RIAGENDR']) == 2

        conds = []
        for var, name, icd, fname in CONDITIONS:
            v = lookup(fname, seqn, var)
            if v == 1:
                conds.append({'name': name, 'icd10': icd, 'severity_or_stage': 'self-reported'})
            elif var == 'DIQ010' and v == 3:
                conds.append({'name': 'Prediabetes', 'icd10': 'R73.03',
                              'severity_or_stage': 'self-reported borderline'})
        scr = lookup('BIOPRO_J', seqn, 'LBXSCR')
        e = egfr_ckdepi_2021(scr, age, female) if scr else None
        if e is not None and e < 60 and not any(c['icd10'].startswith('N18') for c in conds):
            conds.append({'name': 'Chronic kidney disease', 'icd10': 'N18.9',
                          'severity_or_stage': f'{ckd_stage(e)} by CKD-EPI 2021'})
        if len(conds) < a.min_conditions:
            skipped_conditions += 1
            continue

        meds = []
        for _, r in rxu.loc[rxu['SEQN'] == seqn].iterrows():
            raw_name = s(r.get('RXDDRUG'))
            # NHANES writes its missing-value sentinels into RXDDRUG itself: 55555 / 77777 /
            # 99999 ("don't know" / "refused"). They are not drugs and must not become medications.
            if (not raw_name or raw_name.upper() in {'REFUSED', "DON'T KNOW", 'UNKNOWN'}
                    or raw_name.strip().isdigit()):
                continue
            reasons = [s(r.get(f'RXDRSD{i}')) for i in (1, 2, 3)]
            indication = '; '.join(x for x in reasons if x) or 'indication not recorded'
            # NHANES writes a combination product as "SACUBITRIL; VALSARTAN" — ONE prescription,
            # one container, one indication. Splitting it into two medications with independent
            # monotherapy doses fabricates a therapeutic duplication (sacubitril/valsartan PLUS
            # standalone valsartan) that a model correctly flags as a prescribing error which the
            # patient does not actually have. Kept as a single medication.
            parts = [p.strip().lower() for p in raw_name.split(';') if p.strip()]
            key = '/'.join(parts)
            all_drugs += 1
            hit = doses.get(key)
            if hit is None and len(parts) == 1:
                # Prefix fallback ("insulin glargine u-300" -> "insulin glargine") is for single
                # ingredients only. Allowing it on a combination would silently award the first
                # component's monotherapy dose to the whole product.
                hit = next((v for k, v in doses.items() if key.startswith(k)), None)
            if hit:
                meds.append({'name': key, 'dose': hit['dose'],
                             'frequency': hit['frequency'], 'indication': indication,
                             'dose_source': 'standardised'})
            else:
                no_dose += 1
                meds.append({'name': key, 'dose': 'strength not recorded',
                             'frequency': 'as prescribed', 'indication': indication,
                             'dose_source': 'unavailable'})
        if len(meds) < a.min_drugs:
            continue

        labs = []
        sbp = [lookup('BPX_J', seqn, f'BPXSY{i}') for i in (1, 2, 3, 4)]
        dbp = [lookup('BPX_J', seqn, f'BPXDI{i}') for i in (1, 2, 3, 4)]
        # NHANES codes an unobtainable reading as a denormal float (~5.4e-79), not 0 or NaN;
        # it survives a truthiness filter and drags the mean down. Drop sub-physiological values.
        sbp = [x for x in sbp if x and x > 1]; dbp = [x for x in dbp if x and x > 1]
        if sbp and dbp:
            labs.append({'name': 'Blood pressure',
                         'value': f'{round(sum(sbp)/len(sbp))}/{round(sum(dbp)/len(dbp))} mmHg',
                         'reference_or_flag': 'mean of available measured readings'})
        bmi = lookup('BMX_J', seqn, 'BMXBMI')
        if bmi:
            labs.append({'name': 'BMI', 'value': f'{bmi:.1f} kg/m2', 'reference_or_flag': 'measured'})
        if e is not None:
            labs.append({'name': 'eGFR', 'value': f'{e} mL/min/1.73m2',
                         'reference_or_flag': f'{ckd_stage(e)} (CKD-EPI 2021, creatinine)'})
        for fname, var, label, unit, prec in LABS:
            v = lookup(fname, seqn, var)
            if v is None:
                continue
            labs.append({'name': label, 'value': f'{v:.{prec}f} {unit}'.strip(),
                         'reference_or_flag': 'measured'})

        walk = lookup('PFQ_J', seqn, 'PFQ061B')      # difficulty walking a quarter mile
        equip = lookup('PFQ_J', seqn, 'PFQ054')      # needs special equipment to walk
        bits = []
        if walk and walk >= 2: bits.append('reports difficulty walking a quarter of a mile')
        if equip == 1:         bits.append('uses walking equipment')
        function = ('; '.join(bits) if bits
                    else 'no walking limitation reported') + ' (self-reported, survey instrument)'

        case = {
            'case_id': f'NHANES-{int(seqn)}',
            'demographics': f"{age}-year-old {'woman' if female else 'man'}, "
                            f"community-dwelling survey participant",
            'conditions': conds,
            'current_medications': meds,
            'labs_vitals': labs,
            # Empty on purpose. The survey does not mark drugs to stop, reduce or
            # increase, and none are added here.
            'restraint_indicated_items': [],
            'frailty_or_function': function,
            'goals_of_care': GOALS,
            'provenance': {
                'source': 'NHANES 2017-2018 (public use)',
                'seqn': int(seqn),
                'doses_real': False,
                'notes': 'Drug names and reasons for use are as recorded by NHANES. Strengths come from '
                         'usual_doses.csv and are not measured. Goals of care are not recorded by the survey.',
            },
        }
        (OUT / f"{case['case_id']}.json").write_text(json.dumps(case, indent=1), encoding='utf-8')
        made.append(case)

    print(f"{len(made)} records written to {OUT}; {no_dose} of {all_drugs} drug mentions had no usual dose")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
