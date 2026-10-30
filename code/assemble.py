"""Assemble the analysis tables from the vignettes, the NHANES records and the stored model responses.

    python assemble.py --vignettes vignettes.json --nhanes nhanes_records/ --responses responses/ --out rows.json

Responses are not distributed. They are expected as responses/<question>/<record id>__<model>.json, each
holding the model's reply as {"raw_response": "..."}, with these question folders:
  plan                   management plan, written alone
  plan_after_appraisal   management plan written after the three-option appraisal, same conversation (test 3)
  verdict_alone          three-option appraisal, verdict format, alone
  verdict_after_plan     three-option appraisal, verdict format, after the plan, same conversation
  verdict_before_plan    three-option appraisal, verdict format, before the plan (first turn of test 3)
  verdict4_before_plan   four-option appraisal, verdict format, before the plan (first turn of test 3, second arm)
  plan_after_appraisal4  management plan written after the four-option appraisal, same conversation
  appraisal3_alone       three-option appraisal, probability format, alone (vignettes)
  appraisal4_alone       four-option appraisal, probability format, alone
  direct_alone           direct question, probability format, alone (for the vignettes it also carries the proposals)
  direct_after_plan      direct question after the plan, same conversation (vignettes; also the proposals)
Record ids are the vignette ids (e.g. AFB-001) and NHANES-<SEQN>. Model keys are in gapstudy/models.py.
"""
import argparse, json
from pathlib import Path
from gapstudy.parsing import load_response, norm_probs
from gapstudy.direction import plan_decision
from gapstudy.drugs import same_drug
from gapstudy.models import MODELS, DEV
from gapstudy.key.match import plan_made

LAB = {'continue unchanged': 'keep', 'appropriate': 'keep', 'reduce dose': 'reduce', 'reduce': 'reduce', 'stop': 'stop',
       'increase dose': 'increase', 'increase': 'increase'}
OQ = ['Continue unchanged', 'Reduce dose', 'Stop', 'Increase dose']; OA3 = ['Appropriate', 'Reduce dose', 'Stop']; OA4 = OA3 + ['Increase dose']
YN = ['Yes', 'No']
REF = {'stop': 'Stop', 'reduce dose': 'Reduce dose', 'increase dose': 'Increase dose', None: None}


def top(p, opts):
    return LAB[max(opts, key=lambda o: p[o]).lower()] if p else None


class Responses:
    def __init__(self, root): self.root, self.c = Path(root), {}
    def get(self, question, rid, model):
        k = (question, rid, model)
        if k not in self.c: self.c[k] = load_response(self.root / question / f'{rid}__{model}.json')
        return self.c[k]
    def verdict(self, question, rid, model, name):
        j = self.get(question, rid, model)
        if not j: return None
        for x in j.get('judgments') or []:
            if same_drug(x.get('medication') or '', name): return LAB.get(str(x.get('verdict') or '').lower())
        return None


def vignette_rows(vignettes, R):
    drugs, props = [], []
    for v in vignettes:
        rid = v['id']; doses = {d['name']: f"{d['dose']} {d['frequency']}" for d in v['current_drugs']}
        for mk in MODELS:
            plan = R.get('plan', rid, mk)
            if not plan: continue
            p2, p3 = R.get('plan_after_appraisal', rid, mk), R.get('plan_after_appraisal4', rid, mk)
            a3, a4 = R.get('appraisal3_alone', rid, mk) or {}, R.get('appraisal4_alone', rid, mk) or {}
            q, qc = R.get('direct_alone', rid, mk) or {}, R.get('direct_after_plan', rid, mk) or {}
            for d in v['current_drugs']:
                qid = f"{d['id']}_q1"; dt = f"{d['dose']} {d['frequency']}"; val = REF[d['reference_answer']]
                dec, how = plan_decision(plan, d['name'], dt)
                p = {'A3': norm_probs(a3.get(qid), OA3), 'A4': norm_probs(a4.get(qid), OA4), 'Q': norm_probs(q.get(qid), OQ),
                     'Qc': norm_probs(qc.get(qid), OQ)}
                drugs.append(dict(case=rid, model=mk, dev=DEV[mk], name=d['name'], validated=val, how=how,
                                  etype=('restraint' if val in ('Stop', 'Reduce dose') else 'indicated') if val else None,
                                  V=R.verdict('verdict_alone', rid, mk, d['name']), A3=top(p['A3'], OA3), A4=top(p['A4'], OA4),
                                  Q=top(p['Q'], OQ), C=R.verdict('verdict_after_plan', rid, mk, d['name']), Qc=top(p['Qc'], OQ),
                                  F=R.verdict('verdict_before_plan', rid, mk, d['name']), P=LAB[dec.lower()],
                                  P2=LAB[plan_decision(p2, d['name'], dt)[0].lower()] if p2 else None,
                                  F4=R.verdict('verdict4_before_plan', rid, mk, d['name']),
                                  P3=LAB[plan_decision(p3, d['name'], dt)[0].lower()] if p3 else None, probs=p))
            for x in v['proposed_changes']:
                qid = f"{x['id']}_q3"
                props.append(dict(case=rid, model=mk, dev=DEV[mk], name=x['proposal'],
                                  etype='trap' if x['reference_answer'] == 'no' else 'indicated',
                                  plan=plan_made(rid, x['id'], plan, doses),
                                  probs={'A3': norm_probs(a3.get(qid), YN), 'A4': norm_probs(a4.get(qid), YN),
                                         'Q': norm_probs(q.get(qid), YN), 'Qc': norm_probs(qc.get(qid), YN)}))
    return drugs, props


def nhanes_rows(records, R):
    out = []
    for case in records:
        rid = case['case_id']; meds = case['current_medications']
        for mk in MODELS:
            plan = R.get('plan', rid, mk)
            if not plan: continue
            a4 = R.get('appraisal4_alone', rid, mk) or {}; q = R.get('direct_alone', rid, mk) or {}
            p2, p3 = R.get('plan_after_appraisal', rid, mk), R.get('plan_after_appraisal4', rid, mk)
            for i, m in enumerate(meds, 1):
                dt = f"{m.get('dose', '')} {m.get('frequency', '')}"; dec, how = plan_decision(plan, m['name'], dt)
                out.append(dict(case=rid, model=mk, dev=DEV[mk], name=m['name'], etype=None, validated=None, how=how,
                                V=R.verdict('verdict_alone', rid, mk, m['name']), A4=top(norm_probs(a4.get(f'M{i}_q1'), OA4), OA4),
                                Q=top(norm_probs(q.get(f'M{i}_q1'), OQ), OQ),
                                C=R.verdict('verdict_after_plan', rid, mk, m['name']), F=R.verdict('verdict_before_plan', rid, mk, m['name']),
                                P=LAB[dec.lower()], P2=LAB[plan_decision(p2, m['name'], dt)[0].lower()] if p2 else None,
                                F4=R.verdict('verdict4_before_plan', rid, mk, m['name']),
                                P3=LAB[plan_decision(p3, m['name'], dt)[0].lower()] if p3 else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vignettes', default='vignettes.json'); ap.add_argument('--nhanes', default='nhanes_records')
    ap.add_argument('--responses', default='responses'); ap.add_argument('--out', default='rows.json')
    a = ap.parse_args()
    vignettes = json.load(open(a.vignettes, encoding='utf-8'))
    records = [json.load(open(f, encoding='utf-8')) for f in sorted(Path(a.nhanes).glob('NHANES-*.json'))]
    R = Responses(a.responses)
    drugs, props = vignette_rows(vignettes, R)
    nh = nhanes_rows(records, R)
    json.dump({'vignette_drugs': drugs, 'vignette_proposals': props, 'nhanes_drugs': nh}, open(a.out, 'w', encoding='utf-8'))
    print(f'{len(drugs)} vignette drug decisions, {len(props)} plan-proposal pairs, {len(nh)} NHANES drug decisions -> {a.out}')


if __name__ == '__main__':
    main()
