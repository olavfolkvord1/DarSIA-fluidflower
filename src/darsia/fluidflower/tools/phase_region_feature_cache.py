"""Disposable, hash-verified image features independent of review annotations."""
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np

from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools.render_concentration_movie import write

FORMAT = 'regional-image-features-v1'
ARRAYS = ('features', 'domain', 'change', 'density')
VALUES = ('change_floor', 'native_shape', 'spacing_m')


class FeatureCache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def folder(self, identity):
        key = hashlib.sha256(json.dumps(dict(format=FORMAT, inputs=identity), sort_keys=True).encode()).hexdigest()
        return self.root / key

    def load(self, identity):
        folder = self.folder(identity)
        receipt = folder / 'receipt.json'
        if not receipt.is_file():
            return None
        recorded = json.loads(receipt.read_text())
        if recorded['identity'] != json.loads(json.dumps(identity)):
            raise ValueError('Cached image feature identity changed')
        archive = folder / recorded['file']
        if sha256_file(archive) != recorded['sha256']:
            raise ValueError('Cached image features changed')
        with np.load(archive, allow_pickle=False) as values:
            result = {key: values[key] for key in ARRAYS}
        result.update(recorded['values'])
        return result

    def save(self, identity, prepared):
        size = sum(prepared[key].nbytes for key in ARRAYS)
        if shutil.disk_usage(self.root).free < size * 2 + 2 * 1024**3:
            return False  # Optional scratch data must never exhaust local storage.
        folder = self.folder(identity)
        folder.mkdir(exist_ok=True)
        filename = 'features-' + uuid.uuid4().hex + '.npz'
        path = folder / filename
        with path.open('wb') as stream:
            # Uncompressed arrays avoid spending CPU on ephemeral SSD cache data.
            np.savez(stream, **{key: prepared[key] for key in ARRAYS})
        values = {key: prepared[key] for key in VALUES}
        values['native_shape'] = list(values['native_shape'])
        values['spacing_m'] = list(values['spacing_m'])
        write(folder / 'receipt.json', dict(identity=identity, file=filename,
            sha256=sha256_file(path), values=values))
        return True
