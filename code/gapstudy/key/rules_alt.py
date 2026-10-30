"""Acceptable alternatives to indicated starts. A plan that starts the alternative has addressed the
indication. Alternatives already inside the base rule's class (paracetamol, vitamin D3, calcium citrate,
other statins, other laxatives) need no entry. Merged into the base rule as ANY(base, alternative)."""
from .classes import *
from .dsl import *
TOP_NSAID = S(r"diclofenac|ketoprofen|ibuprofen|piroxicam|\bnsaids?\b", req=TOPICAL)
CAPSAICIN = S(r"capsaicin")
LIDOCAINE = S(r"lidocaine|lignocaine")
GABA_RENAL = ANY(S(r"gabapentin|pregabalin", req=RENAL_ADJ), S(r"pregabalin", max_dose=75), S(r"gabapentin", max_dose=300))
ALT = {
 ('AFB-007', 'X3'): TOP_NSAID,
 ('BONE-001', 'X4'): S(r"zoledron"),
 ('BONE-007', 'X1'): ANY(GABA_RENAL, LIDOCAINE),
 ('BONE-008', 'X5'): S(r"zoledron"),
 ('CMM-001', 'X2'): TOP_NSAID,
 ('CMM-003', 'X3'): TOP_NSAID,
 ('CMM-004', 'X4'): TOP_NSAID,
 ('CMM-005', 'X4'): S(r"escitalopram|citalopram|paroxetine"),
 ('CMM-006', 'X3'): TOP_NSAID,
 ('CMM-007', 'X4'): S(r"escitalopram|mirtazapine"),
 ('CMM-008', 'X4'): TOP_NSAID,
 ('CRM-001', 'X3'): TOP_NSAID,
 ('CRM-003', 'X2'): TOP_NSAID,
 ('CRM-004', 'X3'): CAPSAICIN,
 ('CRM-007', 'X2'): ANY(CAPSAICIN, TOP_NSAID),
 ('CRM-008', 'X3'): ANY(CAPSAICIN, TOP_NSAID),
 ('DCO-001', 'X1'): S(r"gabapentin|pregabalin"),
 ('DCO-004', 'X3'): S(r"gabapentin|pregabalin"),
 ('DCO-005', 'X4'): ANY(LIDOCAINE, CAPSAICIN),
 ('LLE-002', 'X4'): S(r"ipratropium"),
 ('LLE-007', 'X4'): S(r"ipratropium"),
 ('PGI-001', 'X4'): S(ACETAMINOPHEN),
 ('PGI-002', 'X3'): TOP_NSAID,
 ('PGI-005', 'X2'): S(r"denosumab"),
 ('PGI-008', 'X2'): S(r"pregabalin|" + ACETAMINOPHEN),
 ('RAC-001', 'X3'): TOP_NSAID,
 ('RAC-002', 'X2'): TOP_NSAID,
 ('SEED-001', 'X1'): S(ACETAMINOPHEN),
}
