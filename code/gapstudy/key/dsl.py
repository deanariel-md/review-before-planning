"""Rule vocabulary for matching a stored plan's actions to a proposal. A rule is a nested tuple;
evaluation returns the list of plan actions that satisfy it (possibly empty), or None if it fails.
  S(rx, exclude, req, max_dose)  a start/add action whose target matches rx; exclude and req are
                                 tested on target + detail; max_dose caps the dose stated
  UP(rx, min_dose, req)          an upward adjustment of a matching current drug (direction
                                 rules); with min_dose, the new dose stated reaches it; req tested on its detail
  OFF(rx)                        the matching drug is stopped, reduced, or tapered
  STOP(rx, exclude)              a stop action on a matching drug; exclude tested on its detail
  SW(from_rx, to_rx)             the from-drug is stopped or adjusted and the to-drug is started or
                                 named in that action's detail
  REF(rx)                        a refer or order_test action whose target or detail matches rx
  ACT(types, rx)                 an action of the listed types whose target matches rx
  ANY(...), ALL(...), NOT(r)     union, conjunction, absence
"""
def S(rx, exclude=None, req=None, max_dose=None): return ('S', rx, exclude, req, max_dose)
def UP(rx, min_dose=None, req=None): return ('UP', rx, min_dose, req)
def OFF(rx): return ('OFF', rx)
def STOP(rx, exclude=None): return ('STOP', rx, exclude)
def SW(a, b): return ('SW', a, b)
def REF(rx): return ('REF', rx)
def ACT(types, rx): return ('ACT', tuple(types), rx)
def ANY(*r): return ('ANY',) + r
def ALL(*r): return ('ALL',) + r
def NOT(r): return ('NOT', r)
