"""Logical 32-bit permutation helpers. Project-authored, MIT licensed."""
def validate_mapping(mapping):
    if sorted(mapping) != list(range(32)):
        raise ValueError("mapping must be a permutation of 0..31")

def logical_to_physical(word, mapping):
    validate_mapping(mapping)
    out = 0
    for logical, physical in enumerate(mapping):
        out |= ((int(word) >> logical) & 1) << physical
    return out

def physical_to_logical(word, mapping):
    validate_mapping(mapping)
    out = 0
    for logical, physical in enumerate(mapping):
        out |= ((int(word) >> physical) & 1) << logical
    return out
