"""
checks that a model file is the one that was trained and scored, before the unit trusts it
a field unit can be lost or left unattended, and a swapped model could quietly stop finding people
so each model's sha-256 is kept in models/checksums.txt, written by the training scripts, and checked on loading

a file listed there with a different checksum is refused, and the unit falls back to a detector it can trust
a file that is not listed loads with a warning, as a model straight from modelarts would before it is recorded
a checksum can also be given directly, as NEUSE_MINDYOLO_SHA256 does for the mindyolo model
"""
import hashlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)

checksums_name = "checksums.txt"


class integrity_error(ValueError):
    # a model file whose checksum does not match the one recorded for it
    pass


def sha256(path):
    # the file's sha-256 as hex, read in pieces so a large model does not fill the memory
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for piece in iter(lambda: f.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def listed(folder):
    # the checksums recorded in a folder's checksums.txt, file name to hex, empty when there is none
    found = {}
    path = Path(folder) / checksums_name
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition(" ")
        found[name.strip()] = digest.lower()
    return found


def verify(path, expected=None):
    # checks a model file before it is loaded
    # returns True when it matches its checksum, False when there is no checksum for it
    # raises integrity_error when it does not match
    path = Path(path)
    expected = (expected or listed(path.parent).get(path.name) or "").lower()
    if not expected:
        log.warning("%s has no recorded checksum, so it cannot be checked, see src/integrity.py", path.name)
        return False
    actual = sha256(path)
    if actual != expected:
        raise integrity_error(f"{path.name} does not match its recorded checksum, it may have been changed or damaged")
    return True


def record(path):
    # writes or updates a model file's checksum in its folder's checksums.txt, called after training
    path = Path(path)
    sums = listed(path.parent)
    sums[path.name] = sha256(path)
    lines = ["# sha-256 of each trained model, checked by the unit before loading, see src/integrity.py"]
    lines += [f"{digest} {name}" for name, digest in sorted(sums.items())]
    (path.parent / checksums_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
