"""Read-only frozen-v7/native-image preflight; no converted images required."""
import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np

HASHES = {
    'fisheye3dod_infos_train.pkl': '4fc1cac224ba5bc5e663f3a2cfe072c29c9be1a548d94ea8bdc31851288da3c6',
    'fisheye3dod_infos_val.pkl': '2b7593f804550516c86fd4262a49a90159379a11cf83e886aa20ffbde27732e8',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[3]
    calibration = repo / 'projects/raybev_lite/configs/calibration_baselink.json'
    if hashlib.sha256(calibration.read_bytes()).hexdigest() != '3c26fe85192a37eb982f6911e7a768da039f04f7c161d0b6c08f3508c6acd9e3':
        raise ValueError('Calibration changed from frozen v7 contract')
    total = 0
    for name, digest in HASHES.items():
        raw = (a.annotations / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('Reviewed annotation changed: ' + name)
        document = pickle.loads(raw)  # Trusted, hash-checked local annotation.
        for row in document['data_list']:
            cameras = row['cam_info']['cam_fisheye']
            if len(cameras) != 4:
                raise ValueError('Expected four cameras')
            for info in cameras.values():
                path = (a.root / info['cam_path']).resolve()
                if a.root.resolve() not in path.parents or not path.is_file():
                    raise FileNotFoundError(str(path))
                total += 1
    anchor = np.load(str(repo / 'projects/ROSA/datasets/combined_bags_camera_manual_review_v7/rosa/anchors/train_centers300_seed3407.npy'))
    if anchor.shape != (300, 11) or not np.isfinite(anchor).all():
        raise ValueError('Invalid TRAIN constructor anchors')
    print('PASS: frozen v7 labels/calibration; {} native images exist; runtime resize only'.format(total))


if __name__ == '__main__':
    main()
