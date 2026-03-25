"""快速测试 DNG 写入"""
import sys
import os
import numpy as np
sys.stdout.write("Starting DNG test...\n")
sys.stdout.flush()

from tifffile import TiffWriter, TiffFile

bayer = np.random.randint(0, 4095, (480, 640), dtype=np.uint16)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "test_write.dng")
os.makedirs(os.path.dirname(path), exist_ok=True)

sys.stdout.write(f"Writing to: {path}\n")
sys.stdout.flush()

try:
    with TiffWriter(path, bigtiff=False) as tif:
        tif.write(
            bayer,
            photometric=32803,
            compression=1,
            bitspersample=16,
            subfiletype=0,
            metadata=None,
            extratags=[
                (50706, 1, 4, (1, 4, 0, 0)),
                (50707, 1, 4, (1, 4, 0, 0)),
                (33421, 3, 2, (2, 2)),
                (33422, 1, 4, (0, 1, 1, 2)),
                (50714, 4, 1, 0),
                (50717, 4, 1, 4095),
                (50721, 12, 9, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
                (50778, 3, 1, 21),
                (50728, 12, 3, (1.0, 1.0, 1.0)),
            ],
        )
    fsize = os.path.getsize(path)
    sys.stdout.write(f"DNG written: {fsize} bytes\n")

    with TiffFile(path) as t:
        page = t.pages[0]
        sys.stdout.write(f"Read back: shape={page.shape}, dtype={page.dtype}\n")
        for tag in page.tags.values():
            if tag.code > 33000:
                sys.stdout.write(f"  Tag {tag.code}: {tag.name} = {tag.value}\n")
    sys.stdout.write("SUCCESS\n")
except Exception as e:
    sys.stdout.write(f"ERROR: {e}\n")
    import traceback
    traceback.print_exc()

sys.stdout.flush()
