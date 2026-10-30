"""Reading a plan's decision on a current drug: continue, stop, reduce or increase.

Each plan action naming the drug is matched by drug name. A stop action is a stop. An adjust action is
given a direction by ordered rules, read from its detail text only (the rationale is excluded because
it is full of phrases such as "increases bleeding risk"):
  R0b  a stop verb in the detail                      -> down
  R1   both the current dose and a new dose parse      -> compare magnitudes (skipped when the detail
                                                          quotes a ceiling such as "maximum 3 g/day")
  R2   unambiguous downward language                   -> down
  R3   unambiguous upward language                     -> up
  R4   both                                            -> conflict (unresolved)
  R5   neither                                         -> unclear (unresolved)
Unresolved adjustments are counted as reductions in the main analysis and as unchanged in the
sensitivity analysis.
"""
import re
from .drugs import same_drug

DOWN = re.compile(r'\b(reduc|decreas|lower|taper|de-?escalat|down-?titrat|halv|wean|de-?intensif|'
                  r'discontinu|cut back|scale back|less frequent)\w*', re.I)
UP = re.compile(r'\b(increas|up-?titrat|escalat|rais|maximi|intensif|augment|step up|higher dose|'
                r'more frequent)\w*', re.I)
DOSE = re.compile(r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|units?|iu)\b', re.I)
RISK_UP = re.compile(r'\b(increas\w*|rais\w*|higher)\s+(?:the\s+)?(?:risk|bleed\w*|fall\w*|'
                     r'mortalit\w*|harm|toxicit\w*|hazard|chance|likelihood|danger)', re.I)
CEILING = re.compile(r'\b(max(?:imum)?|not to exceed|up to|ceiling|cap)\b', re.I)
STOPWORD = re.compile(r'\b(stop|discontinu\w*|cease|hold|withdraw\w*)\b', re.I)


def doses(t): return [float(m.group(1)) for m in DOSE.finditer(t or '')]


def classify(action, current_dose_text):
    """-> (direction, rule); direction in {down, up, conflict, unclear}."""
    detail = action.get('detail') or ''
    if STOPWORD.search(detail): return 'down', 'R0b_stop_verb_in_detail'
    cur = doses(current_dose_text)
    new = [] if CEILING.search(detail) else doses(detail)
    if cur and new:
        if max(new) < min(cur): return 'down', 'R1_dose_compare'
        if min(new) > max(cur): return 'up', 'R1_dose_compare'
    d = bool(DOWN.search(detail)); u = bool(UP.search(detail)) and not RISK_UP.search(detail)
    if d and not u: return 'down', 'R2_keyword_down'
    if u and not d: return 'up', 'R3_keyword_up'
    if d and u: return 'conflict', 'R4_conflict'
    return 'unclear', 'R5_unclear'


DEFER = r"^\s*(?:if|consider\w*|may|might|could|should|when|unless|pending)\b"
REDUCE_KW = re.compile(r"\b(reduc|decreas|lower|halv|down-?titrat|cut back|less frequent)\w*", re.I)
STOP_KW = re.compile(r"\b(stop|discontinu\w*|cease|withdraw\w*)\b", re.I)


def plan_decision(plan, med, dose_txt, unresolved='Reduce dose', deferred_continue=False):
    """The plan's decision on one current drug -> (decision, how).
    decision in {Continue unchanged, Reduce dose, Stop, Increase dose}; how in {stop, adjust, unresolved, none}."""
    acts = [a for a in plan.get('actions') or [] if same_drug(a.get('target') or '', med)]
    if any(a.get('action') == 'stop' for a in acts): return 'Stop', 'stop'
    adj = [a for a in acts if a.get('action') in ('adjust', 'reduce')]
    if deferred_continue: adj = [a for a in adj if not re.search(DEFER, a.get('detail') or '', re.I)]
    out = set()
    for a in adj:
        d, rule = classify(a, dose_txt) if a.get('action') == 'adjust' else ('down', 'reduce')
        det = a.get('detail') or ''
        if d == 'down':
            cur, new = doses(dose_txt), ([] if CEILING.search(det) else doses(det))
            lower = bool(cur and new and max(new) < min(cur))
            out.add('Reduce dose' if (lower or REDUCE_KW.search(det) or not STOP_KW.search(det)) else 'Stop')
        elif d == 'up': out.add('Increase dose')
        else: out.add(None)
    if not out: return 'Continue unchanged', 'none'
    if len(out) == 1 and None not in out: return out.pop(), 'adjust'
    return unresolved, 'unresolved'
