"""Drug-name matching between a model's free-text target and a medication record.

Offline and deterministic: string normalisation, a small brand-to-generic synonym map, and the
components of parenthetical or combination names. Matching is conservative: a substring or a shared
long token over the synonym sets of both names, excluding formulation words.
"""
from __future__ import annotations
from functools import lru_cache

_BRAND_TO_GENERIC: dict[str, str] = {
    # NSAIDs
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "brufen": "ibuprofen",
    "nurofen": "ibuprofen",
    "voltaren": "diclofenac",
    "naprosyn": "naproxen",
    "aleve": "naproxen",
    "celebrex": "celecoxib",
    # Sulfonylureas (UK/US spelling)
    "glyburide": "glibenclamide",
    "diabeta": "glibenclamide",
    "micronase": "glibenclamide",
    "amaryl": "glimepiride",
    "glucotrol": "glipizide",
    # Biguanide
    "glucophage": "metformin",
    # Anticoagulants
    "coumadin": "warfarin",
    "marevan": "warfarin",
    "eliquis": "apixaban",
    "xarelto": "rivaroxaban",
    "pradaxa": "dabigatran",
    # Beta-blockers
    "zebeta": "bisoprolol",
    "concor": "bisoprolol",
    # ACE inhibitors
    "altace": "ramipril",
    "tritace": "ramipril",
    "vasotec": "enalapril",
    "lisinopril": "lisinopril",
    # Loop diuretic
    "lasix": "furosemide",
    "frusemide": "furosemide",  # older spelling
    # PPIs
    "prilosec": "omeprazole",
    "losec": "omeprazole",
    "nexium": "esomeprazole",
    "protonix": "pantoprazole",
    "prevacid": "lansoprazole",
    # SSRIs
    "zoloft": "sertraline",
    "lustral": "sertraline",
    "prozac": "fluoxetine",
    # Statins
    "lipitor": "atorvastatin",
    "crestor": "rosuvastatin",
    "zocor": "simvastatin",
    # Analgesics
    "tylenol": "acetaminophen",
    "paracetamol": "acetaminophen",
    "panadol": "acetaminophen",
    # Insulins (collapse common long-acting brands to a generic token for matching)
    "lantus": "insulin glargine",
    "toujeo": "insulin glargine",
    "levemir": "insulin detemir",
    "tresiba": "insulin degludec",
}


@lru_cache(maxsize=4096)
def norm(s: str) -> str:
    """Lowercase, keep alphanumerics and spaces, collapse whitespace."""
    cleaned = "".join(c if (c.isalnum() or c.isspace()) else " " for c in s.lower())
    return " ".join(cleaned.split())


def _parenthetical_parts(name: str) -> list[str]:
    """'glibenclamide (glyburide)' -> ['glibenclamide', 'glyburide'].

    Also splits on '/' and ',' so 'amlodipine/valsartan' yields both components.
    """
    swapped = name.replace("(", "|").replace(")", "|").replace("/", "|").replace(",", "|")
    return [p.strip() for p in swapped.split("|") if p.strip()]


@lru_cache(maxsize=1)
def _generic_to_brands() -> dict[str, list[str]]:
    """Reverse of _BRAND_TO_GENERIC: normalized generic -> [normalized brand, ...] (built once)."""
    rev: dict[str, list[str]] = {}
    for brand, gen in _BRAND_TO_GENERIC.items():
        rev.setdefault(norm(gen), []).append(norm(brand))
    return rev


@lru_cache(maxsize=4096)
def synonyms(name: str) -> frozenset[str]:
    """Return the normalized synonym set for a drug name.

    Includes the name itself, each parenthetical/slash component, and brand→generic mappings
    in BOTH directions (so 'Lasix' matches a record of 'furosemide' and vice versa). Multiword
    names are kept intact for precision (we do not split 'insulin glargine' into bare tokens).
    """
    out: set[str] = set()
    rev = _generic_to_brands()
    for cand in [name, *_parenthetical_parts(name)]:
        n = norm(cand)
        if not n:
            continue
        out.add(n)
        generic = _BRAND_TO_GENERIC.get(n)
        if generic:
            out.add(norm(generic))
        out.update(rev.get(n, ()))
    return frozenset(out)


# Formulation/route/salt/class words that are >=5 chars (so they'd otherwise pass _token_overlap's
# length rule) but carry no drug-IDENTITY information and commonly co-occur across UNRELATED drugs:
# two different modified-release tablets, two different inhalers, two different insulins. A check of
# within-case pairs found four false matches from exactly this, e.g.
# same_drug("gliclazide (modified-release)", "diclofenac sodium (modified-release)") -> True on
# "modified"/"release" alone, and same_drug("insulin glargine", "insulin aspart") -> True on
# "insulin" alone. Enumerate conservatively (only words actually observed to collide); do not grow
# this list speculatively.
_FORMULATION_STOPWORDS = frozenset({
    "sodium", "insulin", "inhaler", "modified", "release", "extended", "sustained",
    # Route descriptor, not an ingredient. Observed colliding in the built NHANES cases:
    # 'dorzolamide ophthalmic' ~ 'latanoprost ophthalmic' are two different drugs.
    "ophthalmic",
})


def _token_overlap(a: str, b: str, min_len: int = 5) -> bool:
    """True if a and b share a token of length >= min_len that isn't a formulation/class stopword
    (guards against 'acid'/'oral' noise, and against 'insulin'/'inhaler'-style false positives
    between genuinely different drugs)."""
    shared = (set(a.split()) & set(b.split())) - _FORMULATION_STOPWORDS
    return any(len(tok) >= min_len for tok in shared)


def _is_formulation_only(s: str) -> bool:
    """True if EVERY token in s is a formulation/class stopword (e.g. 'modified release' extracted
    from a parenthetical like '(modified-release)'). Such a fragment carries no drug identity and
    must never itself be treated as a standalone synonym match, even via exact equality or
    substring -- the failure mode _token_overlap's stoplist alone does not reach, since
    synonyms() emits parenthetical fragments as freestanding synonym-set members."""
    toks = s.split()
    return bool(toks) and all(t in _FORMULATION_STOPWORDS for t in toks)


def same_drug(name_a: str, name_b: str) -> bool:
    """Conservative identity match between two drug names (order-independent).

    Match if any normalized synonym of one is a substring of / equal to a synonym of the other,
    or they share a long (>=5 char) token — excluding, in both checks, synonym fragments and
    tokens that are pure formulation/class noise (_FORMULATION_STOPWORDS). Designed to err toward
    *not* matching; ambiguous cases are surfaced upstream for human review rather than guessed.
    """
    syns_a = synonyms(name_a)
    syns_b = synonyms(name_b)
    for sa in syns_a:
        for sb in syns_b:
            if not sa or not sb:
                continue
            if _is_formulation_only(sa) or _is_formulation_only(sb):
                continue
            if sa == sb or sa in sb or sb in sa:
                return True
            if _token_overlap(sa, sb):
                return True
    return False
