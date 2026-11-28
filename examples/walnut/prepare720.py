"""Prepare the 720-view benchmark from the eight Meaney walnut ZIPs (NumPy, Pillow)."""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def prepare(archives, output):
    views = np.empty((720, 296, 280), dtype='<f4')
    seen = set()
    for part in range(1, 9):
        with zipfile.ZipFile(archives / f'20201111_walnut_projections_{part}.zip') as z:
            for name in z.namelist():
                if not name.endswith('.tif'):
                    continue
                i = int(Path(name).stem.rsplit('_', 1)[1]) - 1
                if i == 720:  # 360 degrees duplicates the first view.
                    continue
                if not 0 <= i < 720 or i in seen:
                    raise ValueError(f'unexpected or duplicate view: {name}')
                with Image.open(io.BytesIO(z.read(name))) as image:
                    raw = np.asarray(image)
                    air = raw[:128, :128].mean(dtype='f8')
                    binned = np.roll(raw, -5, axis=1).reshape(296, 8, 280, 8).mean(axis=(1, 3))
                views[i] = -np.log(np.maximum(binned, 0.5) / air)
                seen.add(i)
    if len(seen) != 720 or not np.isfinite(views).all():
        raise ValueError('incomplete scan or invalid normalization')
    output.mkdir(parents=True, exist_ok=True)
    views.tofile(output / 'walnut720.f32')
    (output / 'walnut720.geom').write_text('CFDK1 720 296 280 0.4 0.4 210.66 553.74 0.0 1.0\n')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('usage: prepare720.py ZIP_DIRECTORY OUTPUT_DIRECTORY')
    prepare(Path(sys.argv[1]), Path(sys.argv[2]))
