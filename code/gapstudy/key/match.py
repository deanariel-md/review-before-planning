"""The structured key: did a plan make a proposed change? One rule per proposal (rules_a.py, rules_b.py,
acceptable alternatives in rules_alt.py), drug and test vocabularies in classes.py, operators in dsl.py.
The key reads the plan's actions only; whether the change was right is the reference answer's job.

Explicit actions count: a start, an upward adjustment, a stop, a switch, a test or a referral. A drug
recommended inside another action's detail also counts when a directive verb precedes it, with no
negation, condition or deferral in the same sentence. An action whose detail opens with a condition or
deferral is not done now."""
import re
from ..direction import classify
from . import rules_a, rules_b, rules_alt
from .dsl import ANY
RULES = {**rules_a.RULES, **rules_b.RULES}
for _k, _alt in rules_alt.ALT.items():
    RULES[_k] = ANY(RULES[_k], _alt)
UNIT = r"(?:mg|mcg|µg|microgram|units?\b)"


def new_dose(text):
    t = text.lower()
    m = re.search(r"\bto\s+(\d+(?:\.\d+)?)\s*(?:/\s*\d+(?:\.\d+)?\s*)?" + UNIT, t)
    if m: return float(m.group(1))
    v = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*" + UNIT, t)]
    return max(v) if v else None


def rx(p, s): return re.search(p, s or '', re.I) is not None


START = ('start', 'add')
EXPLICIT_ONLY = False   # sensitivity: count explicit actions only
VIA_DETAIL = set()      # ids of actions matched only through their detail text
NO_START = ('do not start', 'do_not_start')
DIRECTIVE = (r"(?:substitut\w*|switch(?:ing)? to|replac\w* (?:\w+ )?with|chang\w* to|\buse\b|\busing\b|recommend\w*|"
             r"prefer\w*|\bstart\w*|initiat\w*|\bbegin\b|\badd\b|\badding\b|\boffer\w*|\btry\b|transition\w* to|in favou?r of)")
NEG = r"\b(?:not|no|avoid|never|without|don't|instead of|rather than)\b"
COND = r"\b(?:if|unless|consider\w*|discuss\w*|may|might|could|possibl\w*|future|before|prior to|later|should|review)\b"
REF_NEG_T = r"^\s*(?:no|none|not|avoid|defer|decline|stop|discontinue|do not)\b"
REF_NEG_D = r"not indicated|do not|\bdefer|\bavoid|declin|not recommended|no longer|discontinu|\bstop\b"
DEFER_D = r"^\s*(?:if|consider\w*|may|might|could|should|when|unless|pending)\b"
DEFER_T = r"\bpending\b|\bconsider|\bif (?:needed|indicated|appropriate)|\bpossible\b|\bdiscussion\b"


def deferred(tgt, det): return rx(DEFER_D, det) or rx(DEFER_T, tgt)


GENERIC = r"alternative|analgesi|\bpain\b|therap|agent|medication|supplement|regimen|treatment"


def mention(pat, text):
    """Windows of text in which pat is recommended (directive verb, no negation)."""
    out = []
    for m in re.finditer(DIRECTIVE + r"[^.;]{0,40}?(?:" + pat + r")", text, re.I):
        clause = re.split(r"[.;,:]", text[:m.start()])[-1]
        sentence = re.split(r"[.;]", text[:m.start()])[-1]
        if not re.search(NEG, clause[-16:], re.I) and not re.search(COND, sentence, re.I):
            out.append(text[m.start():m.end() + 30])
    return out


def s_ok(rule, window):
    _, _, exc, req, mx = rule
    if exc and rx(exc, window): return False
    if req and not rx(req, window): return False
    if mx is not None:
        nd = new_dose(window)
        if nd is None or nd > mx: return False
    return True


def ev(rule, acts, doses):
    """Actions satisfying the rule (list, possibly empty), or None if the rule fails."""
    k = rule[0]
    if k == 'ANY':
        hs = [ev(r, acts, doses) for r in rule[1:]]
        hs = [h for h in hs if h is not None]
        return uniq([a for h in hs for a in h]) if hs else None
    if k == 'ALL':
        hs = [ev(r, acts, doses) for r in rule[1:]]
        return None if any(h is None for h in hs) else uniq([a for h in hs for a in h])
    if k == 'NOT':
        return [] if ev(rule[1], acts, doses) is None else None
    hits = []
    for a in acts:
        typ, tgt, det = a.get('action'), a.get('target') or '', a.get('detail') or ''
        td = tgt + ' ' + det
        if k == 'S':
            if typ in NO_START: continue
            if typ in START and rx(rule[1], tgt):
                if s_ok(rule, td) and not deferred(tgt, det): hits.append(a)
            elif not EXPLICIT_ONLY and typ in START and rx(GENERIC, tgt) and not rx(r"discuss|consider|evaluat|assess|review", tgt) and rx(rule[1], det) and not deferred(tgt, det):
                m = re.search(rule[1], det, re.I); before = re.split(r"[.;]", det[:m.start()])[-1]
                if not (rx(COND, before) or rx(NEG, before)) and s_ok(rule, td): hits.append(a); VIA_DETAIL.add(id(a))
            elif not EXPLICIT_ONLY and typ != 'refer' and not any(rx(rule[1], n) for n in doses) and any(s_ok(rule, w) for w in mention(rule[1], det)):
                hits.append(a); VIA_DETAIL.add(id(a))
        elif k == 'STOP' and typ == 'stop' and rx(rule[1], tgt) and not (rule[2] and rx(rule[2], det)):
            hits.append(a)
        elif k == 'OFF' and rx(rule[1], tgt) and (typ in ('stop', 'reduce') or (typ == 'adjust' and (
                classify(a, cur_dose(rule[1], tgt, doses))[0] == 'down' or rx(r"taper|wean|discontinu|cross[- ]?taper|switch", det)))):
            hits.append(a)
        elif k == 'REF' and typ in ('refer', 'order_test') and rx(rule[1], td) and not (rx(REF_NEG_T, tgt) or rx(REF_NEG_D, td) or deferred(tgt, det)):
            hits.append(a)
        elif k == 'ACT' and typ in rule[1] and rx(rule[2], tgt):
            hits.append(a)
        elif k == 'UP' and typ == 'adjust' and rx(rule[1], tgt):
            if classify(a, cur_dose(rule[1], tgt, doses))[0] == 'up' and not (rule[3] and not rx(rule[3], td)) and not deferred(tgt, det):
                if rule[2] is None:
                    hits.append(a)
                else:
                    nd = new_dose(det + ' ' + tgt)
                    if nd is not None and nd >= rule[2]: hits.append(a)
        elif k == 'SW' and typ in ('stop', 'adjust') and rx(rule[1], tgt):
            starts = [b for b in acts if b.get('action') in START and rx(rule[2], b.get('target') or '')]
            if rx(rule[2], det) or starts:
                hits.append(a); hits.extend(starts)
    return hits if hits else None


def uniq(xs):
    seen = set(); return [x for x in xs if not (id(x) in seen or seen.add(id(x)))]


def cur_dose(pat, tgt, doses):
    return next((d for n, d in doses.items() if rx(pat, n) or n.lower() in tgt.lower()), '')


def plan_made(vignette_id, proposal_id, plan, doses):
    """True if the plan made the proposed change. doses: {current drug name: 'dose frequency'}."""
    return ev(RULES[(vignette_id, proposal_id)], plan.get('actions') or [], doses) is not None
