"""Names and file prefixes of the six benchmark datasets."""
from pathlib import Path
import re


PAPER_DATASETS = {
    'Email-Enron': ('email-Enron', 143, 10883),
    'NDC-Classes': ('NDC-classes', 1161, 49724),
    'NDC-Substances': ('NDC-substances', 5311, 112405),
    'Users-Threads': ('threads-ask-ubuntu', 125602, 192947),
    'Congress-Bills': ('congress-bills', 1718, 260851),
    'Threads-Math-Sx': ('threads-math-sx', 176445, 719792),
}


def name_key(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())


ALIASES = {}
for paper_name, (prefix, _, _) in PAPER_DATASETS.items():
    ALIASES[name_key(paper_name)] = prefix
    ALIASES[name_key(prefix)] = prefix


def canonical_name(name):
    return ALIASES.get(name_key(name), name)


def dataset_status(root=None):
    from .data import locate
    result = []
    for paper_name, (prefix, nodes, events) in PAPER_DATASETS.items():
        row = dict(name=paper_name, prefix=prefix, expected_nodes=nodes, expected_events=events)
        try:
            f = locate(root, prefix)
            row.update(available=True, nverts=str(f))
        except FileNotFoundError as e:
            row.update(available=False, error=str(e))
        result.append(row)
    return result
