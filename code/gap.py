"""The knowledge-action gap: every question about every current drug, paired with the plan.

Questions (answers read as keep / reduce / stop / increase; a flag is reduce or stop)
  V    three-option appraisal, verdict format, alone                vignettes, NHANES
  A3   three-option appraisal, probability format, alone            vignettes
  A4   four-option appraisal, probability format, alone             vignettes, NHANES
  Q    direct question ("What should be done with this drug now?") vignettes, NHANES
  C    three-option appraisal, verdict format, after the plan       vignettes, NHANES
  Qc   direct question after the plan                              vignettes
  F    three-option appraisal, verdict format, before the plan      vignettes, NHANES
  F4   four-option appraisal, verdict format, before the plan       vignettes, NHANES
  P    the plan written alone; P2 the plan written after F; P3 the plan written after F4   vignettes, NHANES
Gap against the plan: b = the answer flags a drug the plan did not reduce or stop; c = the plan reduced or
stopped a drug the answer did not flag; gap = b - c, as a share of drug decisions. Rates are pooled within
each developer and averaged over the 11 developers; exact sign test over the 16 models; 2,000 bootstrap
resamples of patients.

    python gap.py [--rows rows.json] [--out results] [B] [--exclude-dev Anthropic] [--unresolved-keep]
"""
import argparse, json, math, collections, sys
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('B', nargs='?', type=int, default=2000)
ap.add_argument('--rows', default='rows.json'); ap.add_argument('--out', default='results')
ap.add_argument('--exclude-dev', default=None)
ap.add_argument('--unresolved-keep', action='store_true', help='sensitivity: unresolved adjustments read as unchanged')
args = ap.parse_args()
B, EXCL, UK = args.B, args.exclude_dev, args.unresolved_keep
CODE = {'keep': 0, 'reduce': 1, 'stop': 2, 'increase': 3}
T_ = json.load(open(args.rows, encoding='utf-8'))
S, N = T_['vignette_drugs'], T_['nhanes_drugs']
if EXCL:
    S, N = [t for t in S if t['dev'] != EXCL], [t for t in N if t['dev'] != EXCL]
if UK:
    S, N = ([{**t, 'P': 'keep'} if t['how'] == 'unresolved' else t for t in T] for T in (S, N))
# the 19 vignettes most of the three physician raters judged implausible
DROP19 = set(json.load(open(HERE / 'realism_exclusions.json', encoding='utf-8')))


def arrays(T, keys):
    cases = sorted({t['case'] for t in T}); ci = {c: i for i, c in enumerate(cases)}
    out = {k: np.array([CODE[t[k]] if t.get(k) else -1 for t in T]) for k in keys}
    out.update(case=np.array([ci[t['case']] for t in T]), ncase=len(cases), dev=np.unique([t['dev'] for t in T], return_inverse=True)[1],
               model=np.array([t['model'] for t in T]), restraint=np.array([t['etype'] == 'restraint' for t in T]),
               unanch=np.array([t['validated'] is None for t in T]), incr=np.array([t['validated'] == 'Increase dose' for t in T]),
               adjust=np.array([t['how'] in ('adjust', 'unresolved') for t in T]))
    rng = np.random.default_rng(20261025)
    out['cw'] = np.vstack([np.ones(len(cases))] + [rng.multinomial(len(cases), [1 / len(cases)] * len(cases)) for _ in range(B)])
    return out
fl = lambda a: (a == 1) | (a == 2)


def wmean(vals, mask, X, b=0):
    w = X['cw'][b][X['case']] * mask
    num, den = np.bincount(X['dev'], w * vals), np.bincount(X['dev'], w)
    ok = den > 0
    return float((num[ok] / den[ok]).mean()) if ok.any() else float('nan')


def ci(fn, X):
    v = [fn(b) for b in range(1, B + 1)]
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def sign_p(k, n):
    if n == 0: return 1.0
    t = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return float(min(1.0, 2 * t))


def gap(X, x, y='P', convention='B'):
    """x flags & y does not de-intensify (b); y de-intensifies & x does not flag (c)."""
    ok = (X[x] >= 0) & (X[y] >= 0)
    yd = fl(X[y]) | (X['adjust'] & (X[y] >= 0)) if (convention == 'A' and y in ('P', 'P2', 'P3')) else fl(X[y])
    bb = (fl(X[x]) & ~yd).astype(float); cc = (yd & ~fl(X[x])).astype(float)
    pm = {m: float((bb - cc)[ok & (X['model'] == m)].mean()) for m in np.unique(X['model']) if (ok & (X['model'] == m)).any()}
    k, n = sum(v > 0 for v in pm.values()), sum(v != 0 for v in pm.values())
    flagged = fl(X[x]).astype(float)
    return {'b': wmean(bb, ok, X), 'c': wmean(cc, ok, X), 'gap': wmean(bb - cc, ok, X), 'gap_ci': ci(lambda b: wmean(bb - cc, ok, X, b), X),
            'disagree': wmean(bb + cc, ok, X), 'disagree_ci': ci(lambda b: wmean(bb + cc, ok, X, b), X),
            'discordance': wmean(bb, ok & fl(X[x]), X), 'models_b_gt_c': k, 'models_nonzero': n, 'sign_p': sign_p(k, n),
            'per_model': pm, 'n': int(ok.sum())}


def flags(X, k):
    ok = X[k] >= 0
    return {'flagged': wmean(fl(X[k]).astype(float), ok, X), 'flagged_ci': ci(lambda b: wmean(fl(X[k]).astype(float), ok, X, b), X),
            'increase': wmean((X[k] == 3).astype(float), ok, X), 'n': int(ok.sum())}


def reference(X, k):
    ok = X[k] >= 0; f = fl(X[k]).astype(float)
    return {'found': wmean(f, ok & X['restraint'], X), 'found_ci': ci(lambda b: wmean(f, ok & X['restraint'], X, b), X),
            'no_reference_flagged': wmean(f, ok & X['unanch'], X), 'no_reference_ci': ci(lambda b: wmean(f, ok & X['unanch'], X, b), X),
            'increase_found': wmean((X[k] == 3).astype(float), ok & X['incr'], X)}


def paired_diff(X, k1, k2, mask_fn):
    """difference in a reference rate between two elicitations on the same drugs, with CI."""
    ok = (X[k1] >= 0) & (X[k2] >= 0) & mask_fn(X)
    f1, f2 = fl(X[k1]).astype(float), fl(X[k2]).astype(float)
    return {'diff': wmean(f1 - f2, ok, X), 'ci': ci(lambda b: wmean(f1 - f2, ok, X, b), X)}


def paired_inc(X, k1, k2, mask_fn=lambda X: True):
    """difference in the share of drugs increased between two elicitations on the same drugs, with CI and models."""
    ok = (X[k1] >= 0) & (X[k2] >= 0) & mask_fn(X)
    d = (X[k1] == 3).astype(float) - (X[k2] == 3).astype(float)
    pm = {m: float(d[ok & (X['model'] == m)].mean()) for m in np.unique(X['model']) if (ok & (X['model'] == m)).any()}
    k, n = sum(v > 0 for v in pm.values()), sum(v != 0 for v in pm.values())
    return {'k1': wmean((X[k1] == 3).astype(float), ok, X), 'k2': wmean((X[k2] == 3).astype(float), ok, X), 'diff': wmean(d, ok, X),
            'ci': ci(lambda b: wmean(d, ok, X, b), X), 'models_pos': k, 'models_nonzero': n, 'sign_p': sign_p(k, n), 'n': int(ok.sum())}


KS = ['V', 'A3', 'A4', 'Q', 'C', 'Qc', 'F', 'P', 'P2', 'F4', 'P3']
XS = arrays(S, KS); KN = ['V', 'A4', 'Q', 'C', 'F', 'P', 'P2', 'F4', 'P3']; XN = arrays(N, KN)
out = {'B': B, 'synthetic': {'n_rows': len(S), 'n_cases': XS['ncase'], 'n_plans': len({(t['case'], t['model']) for t in S}),
                             'flags': {k: flags(XS, k) for k in KS}, 'reference': {k: reference(XS, k) for k in KS},
                             'gap_vs_plan': {k: gap(XS, k) for k in ['V', 'A3', 'A4', 'Q', 'C', 'Qc']},
                             'gap_vs_plan_conventionA': {k: gap(XS, k, convention='A') for k in ['V', 'A4', 'Q', 'C']},
                             'between': {'A3 vs A4': gap(XS, 'A3', 'A4'), 'A4 vs Q': gap(XS, 'A4', 'Q'), 'V vs A3': gap(XS, 'V', 'A3'),
                                         'V vs C': gap(XS, 'V', 'C'), 'Q vs Qc': gap(XS, 'Q', 'Qc'), 'P2 vs P': gap(XS, 'P2', 'P'),
                                         'F vs P2': gap(XS, 'F', 'P2'), 'P3 vs P': gap(XS, 'P3', 'P'), 'P3 vs P2': gap(XS, 'P3', 'P2'),
                                         'F4 vs P3': gap(XS, 'F4', 'P3'), 'F vs F4': gap(XS, 'F', 'F4')},
                             'order': {'found': paired_diff(XS, 'P2', 'P', lambda X: X['restraint']),
                                       'no_reference': paired_diff(XS, 'P2', 'P', lambda X: X['unanch'])},
                             'order4': {'found': paired_diff(XS, 'P3', 'P', lambda X: X['restraint']),
                                        'no_reference': paired_diff(XS, 'P3', 'P', lambda X: X['unanch']),
                                        'found_vs_P2': paired_diff(XS, 'P3', 'P2', lambda X: X['restraint']),
                                        'no_reference_vs_P2': paired_diff(XS, 'P3', 'P2', lambda X: X['unanch'])},
                             'increase': {'P2 vs P': paired_inc(XS, 'P2', 'P'), 'P3 vs P': paired_inc(XS, 'P3', 'P'), 'P3 vs P2': paired_inc(XS, 'P3', 'P2'),
                                          'P3 vs P, drugs to increase': paired_inc(XS, 'P3', 'P', lambda X: X['incr'])},
                             'appraisal_vs_direct': {'found': paired_diff(XS, 'A3', 'Q', lambda X: X['restraint']),
                                                     'no_reference': paired_diff(XS, 'A3', 'Q', lambda X: X['unanch'])}},
       'nhanes': {'n_rows': len(N), 'n_cases': XN['ncase'], 'n_plans': len({(t['case'], t['model']) for t in N}),
                  'flags': {k: flags(XN, k) for k in KN},
                  'gap_vs_plan': {k: gap(XN, k) for k in ['V', 'A4', 'Q', 'C']},
                  'gap_vs_plan_conventionA': {k: gap(XN, k, convention='A') for k in ['V', 'A4', 'Q', 'C']},
                  'between': {'V vs A4': gap(XN, 'V', 'A4'), 'A4 vs Q': gap(XN, 'A4', 'Q'), 'V vs C': gap(XN, 'V', 'C'), 'P2 vs P': gap(XN, 'P2', 'P'), 'F vs P2': gap(XN, 'F', 'P2'),
                              'P3 vs P': gap(XN, 'P3', 'P'), 'P3 vs P2': gap(XN, 'P3', 'P2'), 'F4 vs P3': gap(XN, 'F4', 'P3'), 'F vs F4': gap(XN, 'F', 'F4')},
                  'increase': {'P2 vs P': paired_inc(XN, 'P2', 'P'), 'P3 vs P': paired_inc(XN, 'P3', 'P'), 'P3 vs P2': paired_inc(XN, 'P3', 'P2')}}}
# sensitivity: the 19 cases most physicians rated implausible removed (synthetic)
S19 = [t for t in S if t['case'] not in DROP19]; X19 = arrays(S19, KS)
out['synthetic']['drop19'] = {'n_cases': X19['ncase'], 'gap_vs_plan': {k: gap(X19, k) for k in ['V', 'A3', 'A4', 'Q', 'C', 'Qc']},
                              'between': {'A3 vs A4': gap(X19, 'A3', 'A4')}}
# transitions when "increase" is removed, synthetic and NHANES
for key, T, a, b_ in (('synthetic', S, 'A4', 'A3'), ('nhanes', N, 'A4', 'V')):
    c = collections.Counter((t[a], t[b_]) for t in T if t.get(a) and t.get(b_)); n = sum(c.values())
    out[key]['transitions_' + a + '_to_' + b_] = {f'{x}->{y}': v / n for (x, y), v in sorted(c.items()) if x != y}
OUT = Path(args.out); OUT.mkdir(exist_ok=True)
out['excluded_developer'] = EXCL
out['unresolved_keep'] = UK
for key_, T_ in (('synthetic', S), ('nhanes', N)): out[key_]['n_unresolved'] = sum(t_['how'] == 'unresolved' for t_ in T_)
json.dump(out, open(OUT / ('gap' + (f'_without_{EXCL.lower()}' if EXCL else '') + ('_unresolved_keep' if UK else '') + '.json'), 'w', encoding='utf-8'), indent=1, default=float)

pc = lambda v: f'{100 * v:5.1f}'
for corpus in ('synthetic', 'nhanes'):
    O = out[corpus]; print(f"\n=== {corpus}: rows {O['n_rows']}, plans {O['n_plans']}, cases {O['n_cases']}")
    for k, v in O['flags'].items():
        R = O.get('reference', {}).get(k)
        extra = f" | found {pc(R['found'])} no-ref {pc(R['no_reference_flagged'])} incr {pc(R['increase_found'])}" if R else ''
        print(f"  {k:3} flagged {pc(v['flagged'])} [{pc(v['flagged_ci'][0])},{pc(v['flagged_ci'][1])}] increase {pc(v['increase'])}{extra}")
    for sec in ('gap_vs_plan', 'gap_vs_plan_conventionA', 'between'):
        print(' ', sec)
        for k, g in O[sec].items():
            print(f"    {k:9} b {pc(g['b'])} c {pc(g['c'])} gap {pc(g['gap'])} [{pc(g['gap_ci'][0])},{pc(g['gap_ci'][1])}] disc {pc(g['discordance'])} | {g['models_b_gt_c']}/{g['models_nonzero']} p={g['sign_p']:.2g}")
print('\norder', {k: (pc(v['diff']), [pc(x) for x in v['ci']]) for k, v in out['synthetic']['order'].items()})
print('appraisal vs direct', {k: (pc(v['diff']), [pc(x) for x in v['ci']]) for k, v in out['synthetic']['appraisal_vs_direct'].items()})
print('drop19', {k: (pc(g['gap']), [pc(x) for x in g['gap_ci']], f"{g['models_b_gt_c']}/{g['models_nonzero']}") for k, g in {**out['synthetic']['drop19']['gap_vs_plan'], **out['synthetic']['drop19']['between']}.items()})
print('transitions', {k: {kk: pc(vv) for kk, vv in v.items()} for k, v in [(c, out[c][[x for x in out[c] if x.startswith('transitions')][0]]) for c in ('synthetic', 'nhanes')]})
