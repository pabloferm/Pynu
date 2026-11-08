"""The published test inputs (PYNU_REFERENCE) are the files of the Harvard Dataverse
archive, unchanged: every file listed in tests/reference/inputs.md5 is present and has
that checksum."""
import hashlib
import os

from conftest import REF


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def test_reference_inputs_match_the_archive(reference_dir):
    bad = []
    with open(os.path.join(REF, "inputs.md5")) as fh:
        for line in fh:
            want, rel = line.split()
            path = os.path.join(reference_dir, rel)
            if not os.path.exists(path):
                bad.append(f"{rel}: missing")
            elif _md5(path) != want:
                bad.append(f"{rel}: checksum differs")
    assert not bad, "PYNU_REFERENCE is not the published archive:\n" + "\n".join(bad)
