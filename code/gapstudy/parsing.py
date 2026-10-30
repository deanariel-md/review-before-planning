"""Reading model responses."""
import json, re


def extract_json(text):
    """The JSON object in a response, tolerating a code fence and trailing commas; None if there is none."""
    if not text: return None
    m = re.search(r'```(?:json)?\s*(.*?)```', text, re.S)
    s = m.group(1) if m else text
    i, j = s.find('{'), s.rfind('}')
    if i < 0 or j < 0: return None
    for cand in (s[i:j + 1], re.sub(r',\s*([}\]])', r'\1', s[i:j + 1])):
        try: return json.loads(cand)
        except Exception: pass
    return None


def load_response(path):
    """Parsed JSON of a stored response file ({"raw_response": "..."}), or None if missing or unparsable."""
    try:
        if not path.exists(): return None
        return extract_json(json.loads(path.read_text(encoding='utf-8')).get('raw_response') or '')
    except Exception:
        return None


def norm_probs(d, opts):
    """Probabilities for the listed options, renormalised to sum to 1; None if the answer gives none."""
    if not isinstance(d, dict): return None
    m = {}
    for k, v in d.items():
        kk = str(k).strip().lower()
        for o in opts:
            if kk == o.lower():
                try: m[o] = float(v)
                except Exception: pass
    if not m: return None
    s = sum(m.values())
    return {o: (m.get(o, 0.0) / s if s > 0 else 0.0) for o in opts}
