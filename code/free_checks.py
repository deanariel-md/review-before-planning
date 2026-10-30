"""Two checks on the vignettes.
1. Threshold or discrimination: does the probability of reduce or stop separate the drugs to stop or reduce
   from the unmarked drugs equally well under the three-option appraisal, the four-option appraisal and the
   direct question (area under the ROC curve)? At the three-option appraisal's false-alarm rate, what share
   of the drugs to stop or reduce do the other questions' probabilities find?
2. Proposed changes: yes/no answers about the 61 recommended additions and 189 harmful proposals, against
   whether the plan made the change (structured key); and the probability of yes read at the plan's own
   rate of harmful proposals made.

    python free_checks.py [--rows rows.json] [--out results] [--exclude-dev Anthropic]
"""
import argparse, json, math, collections
import numpy as np
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument('--rows', default='rows.json'); ap.add_argument('--out', default='results'); ap.add_argument('--exclude-dev', default=None)
args = ap.parse_args()
B = 2000
T_ = json.load(open(args.rows, encoding='utf-8'))
EXCL = args.exclude_dev
drug_rows, prop_rows = T_['vignette_drugs'], T_['vignette_proposals']
if EXCL: drug_rows, prop_rows = [r for r in drug_rows if r['dev'] != EXCL], [r for r in prop_rows if r['dev'] != EXCL]
rows = drug_rows + prop_rows
OQ = ['Continue unchanged', 'Reduce dose', 'Stop', 'Increase dose']; OA3 = ['Appropriate', 'Reduce dose', 'Stop']; OA4 = OA3 + ['Increase dose']
OPT = {'A3': OA3, 'A4': OA4, 'Q': OQ, 'Qc': OQ}
cases = sorted({r['case'] for r in rows}); ci_ = {c: i for i, c in enumerate(cases)}
rng = np.random.default_rng(20261025)
CW = np.vstack([np.ones(len(cases))] + [rng.multinomial(len(cases), [1 / len(cases)] * len(cases)) for _ in range(B)])
pct = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
def sign_p(k, n):
    if n == 0: return 1.0
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n))

# ---------- 1. threshold or knowledge ----------
D = []
for r in drug_rows:
    if r['etype'] == 'indicated': continue
    p = r['probs']
    if not all(p[k] for k in ('A3', 'A4', 'Q')): continue
    D.append(dict(case=ci_[r['case']], model=r['model'], dev=r['dev'], pos=r['etype'] == 'restraint',
                  s={k: (v['Reduce dose'] + v['Stop']) if v else np.nan for k, v in p.items()},
                  top={k: (max(OPT[k], key=lambda o_: v[o_]) in ('Reduce dose', 'Stop')) if v else None for k, v in p.items()}))
models = sorted({d['model'] for d in D}); dev_of = {d['model']: d['dev'] for d in D}
case = np.array([d['case'] for d in D]); y = np.array([d['pos'] for d in D], float); mod = np.array([d['model'] for d in D])
S = {k: np.array([d['s'][k] for d in D], float) for k in ('A3', 'A4', 'Q', 'Qc')}
T = {k: np.array([np.nan if d['top'][k] is None else float(d['top'][k]) for d in D]) for k in ('A3', 'A4', 'Q', 'Qc')}

def auc(s, yy, w):
    ok = ~np.isnan(s) & (w > 0); s, yy, w = s[ok], yy[ok], w[ok]
    u, idx = np.unique(s, return_inverse=True)
    wp = np.bincount(idx, w * yy, len(u)); wn = np.bincount(idx, w * (1 - yy), len(u))
    below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum())) if wp.sum() and wn.sum() else np.nan

def tpr_at(s, yy, w, fpr0):
    """TPR of score s at false-positive rate fpr0, on the ROC curve with ties as diagonal segments."""
    ok = ~np.isnan(s) & (w > 0); s, yy, w = s[ok], yy[ok], w[ok]
    u, idx = np.unique(s, return_inverse=True)
    wp = np.bincount(idx, w * yy, len(u))[::-1]; wn = np.bincount(idx, w * (1 - yy), len(u))[::-1]
    tp = np.concatenate([[0], np.cumsum(wp)]) / wp.sum(); fp = np.concatenate([[0], np.cumsum(wn)]) / wn.sum()
    return float(np.interp(fpr0, fp, tp))

def rate(x, mask, w):
    m = mask & ~np.isnan(x) & (w > 0)
    return float((x[m] * w[m]).sum() / w[m].sum()) if w[m].sum() else np.nan

def per_model(fn, b):
    w = CW[b][case]
    return {m: fn(mod == m, w) for m in models}

def dev_mean(pm):
    by = collections.defaultdict(list)
    for m, v in pm.items():
        if not np.isnan(v): by[dev_of[m]].append(v)
    return float(np.mean([np.mean(v) for v in by.values()]))

def stat(fn):
    est = dev_mean(per_model(fn, 0)); boot = [dev_mean(per_model(fn, b)) for b in range(1, B + 1)]
    pm = per_model(fn, 0); return est, pct(boot), pm

out = {'n_decisions': len(D), 'n_pos': int(y.sum()), 'n_neg': int((1 - y).sum()), 'auroc': {}, 'auroc_diff': {}, 'operating': {}, 'mean_prob': {}}
for k in ('A3', 'A4', 'Q', 'Qc'):
    e, c, pm = stat(lambda m, w, k=k: auc(S[k][m], y[m], w[m])); out['auroc'][k] = {'est': e, 'ci': c, 'per_model': pm}
for a, b_ in (('A3', 'A4'), ('A3', 'Q'), ('A4', 'Q')):
    e, c, pm = stat(lambda m, w, a=a, b_=b_: auc(S[a][m], y[m], w[m]) - auc(S[b_][m], y[m], w[m]))
    k_ = sum(v > 0 for v in pm.values()); n_ = sum(v != 0 for v in pm.values())
    out['auroc_diff'][f'{a} - {b_}'] = {'est': e, 'ci': c, 'models_pos': k_, 'models_nonzero': n_, 'sign_p': sign_p(k_, n_)}
# operating point of the three-option appraisal (most probable option), and the other questions at its false-alarm rate
fpr3 = lambda m, w: rate(T['A3'], m & (y == 0), w)
for k in ('A3', 'A4', 'Q'):
    out['operating'][k] = {'tpr': stat(lambda m, w, k=k: rate(T[k], m & (y == 1), w))[:2], 'fpr': stat(lambda m, w, k=k: rate(T[k], m & (y == 0), w))[:2]}
for k in ('A4', 'Q'):
    out['operating'][f'{k} at A3 false-alarm rate'] = {'tpr': stat(lambda m, w, k=k: tpr_at(S[k][m], y[m], w[m], fpr3(m, w)))[:2]}
    e, c, pm = stat(lambda m, w, k=k: tpr_at(S[k][m], y[m], w[m], fpr3(m, w)) - rate(T['A3'], m & (y == 1), w))
    k_ = sum(v > 0 for v in pm.values()); n_ = sum(v != 0 for v in pm.values())
    out['operating'][f'{k} at A3 FPR minus A3'] = {'est': e, 'ci': c, 'models_pos': k_, 'models_nonzero': n_, 'sign_p': sign_p(k_, n_)}
for k in ('A3', 'A4', 'Q'):
    out['mean_prob'][k] = {'pos': stat(lambda m, w, k=k: rate(S[k], m & (y == 1), w))[:2], 'neg': stat(lambda m, w, k=k: rate(S[k], m & (y == 0), w))[:2]}

# ---------- 2. additions ----------
Ad = []
for r in prop_rows:
    p = r['probs']
    Ad.append(dict(case=ci_[r['case']], model=r['model'], dev=r['dev'], ind=r['etype'] == 'indicated', plan=r['plan'],
                   yes={k: (None if not v else float(v['Yes'] > v['No'])) for k, v in p.items()}, py={k: (None if not v else v['Yes']) for k, v in p.items()}, name=r['name']))
case = np.array([d['case'] for d in Ad]); mod = np.array([d['model'] for d in Ad]); models = sorted(set(mod)); dev_of.update({d['model']: d['dev'] for d in Ad})
ind = np.array([d['ind'] for d in Ad]); plan = np.array([d['plan'] for d in Ad], float)
YES = {k: np.array([np.nan if d['yes'][k] is None else d['yes'][k] for d in Ad]) for k in ('A3', 'A4', 'Q', 'Qc')}
out['additions'] = {'n_rows': len(Ad), 'n_indicated_rows': int(ind.sum()), 'n_trap_rows': int((~ind).sum()),
                    'n_indicated_items': len({(d['case'], d['name']) for d in Ad if d['ind']}), 'n_trap_items': len({(d['case'], d['name']) for d in Ad if not d['ind']})}
for grp, gm in (('indicated', ind), ('trap', ~ind)):
    o = {'plan_yes': stat(lambda m, w: rate(plan, m & gm, w))[:2]}
    for k in ('A3', 'A4', 'Q', 'Qc'):
        bb = np.where(np.isnan(YES[k]), np.nan, ((YES[k] == 1) & (plan == 0)).astype(float))
        cc = np.where(np.isnan(YES[k]), np.nan, ((YES[k] == 0) & (plan == 1)).astype(float))
        g, gci, pm = stat(lambda m, w: rate(bb - cc, m & gm, w))
        k_ = sum(v > 0 for v in pm.values()); n_ = sum(v != 0 for v in pm.values())
        o[k] = {'answer_yes': stat(lambda m, w: rate(YES[k], m & gm, w))[:2], 'b': stat(lambda m, w: rate(bb, m & gm, w))[0],
                'c': stat(lambda m, w: rate(cc, m & gm, w))[0], 'gap': g, 'gap_ci': gci, 'disagree': stat(lambda m, w: rate(bb + cc, m & gm, w))[0],
                'models_pos': k_, 'models_nonzero': n_, 'sign_p': sign_p(k_, n_)}
    out['additions'][grp] = o

# additions as a separating task: P(yes) on indicated starts versus traps, and the plan's operating point
PY = {k: np.array([np.nan if d['py'][k] is None else d['py'][k] for d in Ad]) for k in ('A3', 'A4', 'Q', 'Qc')}
yi = ind.astype(float); out['additions']['roc'] = {}
pfpr = lambda m, w: rate(plan, m & ~ind, w)
out['additions']['roc']['plan'] = {'tpr': stat(lambda m, w: rate(plan, m & ind, w))[:2], 'fpr': stat(lambda m, w: rate(plan, m & ~ind, w))[:2]}
for k in ('A4', 'Q', 'Qc'):
    o = {'auroc': stat(lambda m, w, k=k: auc(PY[k][m], yi[m], w[m]))[:2],
         'tpr_at_plan_fpr': stat(lambda m, w, k=k: tpr_at(PY[k][m], yi[m], w[m], pfpr(m, w)))[:2]}
    e, c, pm = stat(lambda m, w, k=k: tpr_at(PY[k][m], yi[m], w[m], pfpr(m, w)) - rate(plan, m & ind, w))
    k_ = sum(v > 0 for v in pm.values()); n_ = sum(v != 0 for v in pm.values())
    o['minus_plan'] = {'est': e, 'ci': c, 'models_pos': k_, 'models_nonzero': n_, 'sign_p': sign_p(k_, n_)}
    out['additions']['roc'][k] = o
OUT = Path(args.out); OUT.mkdir(exist_ok=True)
out['excluded_developer'] = EXCL
json.dump(out, open(OUT / (f'free_checks_without_{EXCL.lower()}.json' if EXCL else 'free_checks.json'), 'w', encoding='utf-8'), indent=1, default=float)
P = lambda v: f'{100 * v:.1f}'
print('decisions', out['n_decisions'], 'pos', out['n_pos'], 'neg', out['n_neg'])
for k, v in out['auroc'].items(): print('AUROC', k, f"{v['est']:.3f}", [f'{x:.3f}' for x in v['ci']])
for k, v in out['auroc_diff'].items(): print('dAUROC', k, f"{v['est']:+.3f}", [f'{x:+.3f}' for x in v['ci']], f"{v['models_pos']}/{v['models_nonzero']} p={v['sign_p']:.2g}")
for k, v in out['operating'].items(): print('op', k, {kk: (P(vv[0]), [P(x) for x in vv[1]]) if isinstance(vv, (list, tuple)) else vv for kk, vv in v.items()} if 'minus' not in k else (P(v['est']), [P(x) for x in v['ci']], f"{v['models_pos']}/{v['models_nonzero']} p={v['sign_p']:.2g}"))
for k, v in out['mean_prob'].items(): print('meanP', k, 'pos', P(v['pos'][0]), 'neg', P(v['neg'][0]))
A_ = out['additions']; print('additions', {k: A_[k] for k in A_ if k.startswith('n_')})
for grp in ('indicated', 'trap'):
    o = A_[grp]; print(grp, 'plan yes', P(o['plan_yes'][0]))
    for k in ('A3', 'A4', 'Q', 'Qc'):
        v = o[k]; print(f"  {k:3} yes {P(v['answer_yes'][0])} b {P(v['b'])} c {P(v['c'])} gap {P(v['gap'])} [{P(v['gap_ci'][0])},{P(v['gap_ci'][1])}] disagree {P(v['disagree'])} {v['models_pos']}/{v['models_nonzero']} p={v['sign_p']:.2g}")
R_ = A_['roc']; print('plan', 'tpr', P(R_['plan']['tpr'][0]), 'fpr', P(R_['plan']['fpr'][0]))
for k in ('A4', 'Q', 'Qc'):
    v = R_[k]; print(k, 'auroc', f"{v['auroc'][0]:.3f}", [f'{x:.3f}' for x in v['auroc'][1]], 'tpr at plan fpr', P(v['tpr_at_plan_fpr'][0]), [P(x) for x in v['tpr_at_plan_fpr'][1]], 'minus plan', P(v['minus_plan']['est']), [P(x) for x in v['minus_plan']['ci']], f"{v['minus_plan']['models_pos']}/{v['minus_plan']['models_nonzero']} p={v['minus_plan']['sign_p']:.2g}")
