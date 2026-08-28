"""Employer-name canonicalisation.

USCIS and DOL publish no shared employer identifier, so the two sources can only
be joined on the employer name as typed -- and the two agencies type it
differently. USCIS abbreviates where DOL spells out:

    USCIS: UNIV OF N CAROLINA AT CHARLOTTE
    DOL:   UNIVERSITY OF NORTH CAROLINA AT CHARLOTTE

`canon` pushes both sides through the same reduction so the two forms converge on
one key. The abbreviation map deliberately targets the short form, so a name that
is already abbreviated is left alone while its spelled-out twin is pulled onto it.
"""
import re

SUFFIXES = {
    "INC", "INCORPORATED", "LLC", "LLP", "LP", "LTD", "LIMITED", "CORP",
    "CORPORATION", "CO", "COMPANY", "PC", "PLLC", "PA", "PLC", "GROUP",
    "HOLDINGS", "USA", "US", "THE", "AND",
}

ABBREV = {
    "UNIVERSITY": "UNIV", "UNIVERSITIES": "UNIV",
    "INTERNATIONAL": "INTL", "NATIONAL": "NATL",
    "TECHNOLOGIES": "TECH", "TECHNOLOGY": "TECH", "TECHNOLOGIE": "TECH",
    "SERVICES": "SVCS", "SERVICE": "SVCS",
    "SYSTEMS": "SYS", "SYSTEM": "SYS",
    "SOLUTIONS": "SOLNS", "SOLUTION": "SOLNS",
    "CENTER": "CTR", "CENTRE": "CTR", "CENTERS": "CTR",
    "MEDICAL": "MED", "MEDICINE": "MED",
    "HOSPITAL": "HOSP", "HOSPITALS": "HOSP",
    "ASSOCIATES": "ASSOC", "ASSOCIATION": "ASSOC", "ASSOCIATE": "ASSOC",
    "MANAGEMENT": "MGMT", "CONSULTING": "CONSULT", "CONSULTANTS": "CONSULT",
    "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
    "AMERICA": "AMER", "AMERICAN": "AMER",
    "INSTITUTE": "INST", "LABORATORY": "LAB", "LABORATORIES": "LAB",
    "DEPARTMENT": "DEPT", "RESEARCH": "RES", "SOFTWARE": "SW",
    "INFORMATION": "INFO", "COMMUNITY": "CMTY", "REGIONAL": "REG",
}

_PUNCT = re.compile(r"[^A-Z0-9 ]+")
_WS = re.compile(r"\s+")


def canon(name):
    """Reduce an employer name to a comparable join key.

    Uppercase, strip punctuation, map long forms onto short forms, then drop
    corporate suffixes. Returns "" for anything that is not a string.
    """
    if not isinstance(name, str):
        return ""
    s = _PUNCT.sub(" ", name.upper())
    s = _WS.sub(" ", s).strip()
    # Punctuation removal turns "U.S." into two tokens while an unpunctuated
    # "US" remains one. Recombine these forms before suffix removal so the
    # agencies produce the same key.
    s = re.sub(r"\bU S A\b", "USA", s)
    s = re.sub(r"\bU S\b", "US", s)
    tokens = [ABBREV.get(t, t) for t in s.split()]
    tokens = [t for t in tokens if t not in SUFFIXES]
    return " ".join(tokens)
