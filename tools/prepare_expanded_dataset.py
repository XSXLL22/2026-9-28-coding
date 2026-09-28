"""Verify, clip and conservatively quarantine perceptually similar split overlaps.

pHash screening is not proof of shared video identity or complete leakage removal.
Original files/splits are retained; excluded samples are never moved into another split.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.common import file_hash, write_json
from tools.prepare_public_pilot import clip_yolo_row


def image_signature(path):
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        image.load()
        if image.getexif().get(274, 1) != 1:
            raise ValueError(f'EXIF orientation needs explicit label-aware review: {path}')
        # Do not rotate pixels: labels use the stored image coordinate system.
        rgb = image.convert('RGB')
        gray = ImageOps.grayscale(rgb)
        small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
        dct = cv2.dct(small)[:8, :8].ravel()
        median = np.median(dct[1:])
        bits = dct[1:] > median  # exclude DC; 63 informative bits
        phash = sum(int(bit) << index for index, bit in enumerate(bits))
        return {'width': rgb.width, 'height': rgb.height, 'phash63': f'{phash:016x}',
                'decoded_rgb_sha256': hashlib.sha256(f'{rgb.size}:'.encode() + rgb.tobytes()).hexdigest(),
                'gray_std': float(np.std(small))}


def similarity_pairs(records, distance=6):
    pairs = []
    for i, first in enumerate(records):
        for j in range(i + 1, len(records)):
            second = records[j]
            hamming = (int(first['phash63'], 16) ^ int(second['phash63'], 16)).bit_count()
            exact = (first['image_sha256'] == second['image_sha256'] or
                     first['decoded_rgb_sha256'] == second['decoded_rgb_sha256'])
            if exact or hamming <= distance:
                pairs.append({'a': i, 'b': j, 'phash_distance': hamming,
                              'reason': 'exact_pixels_or_file' if exact else 'perceptual_candidate',
                              'cross_split': first['split'] != second['split']})
    return pairs


def quarantine(records, pairs):
    """Connected components keep test, else one val, else one train. Never reassign."""
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for pair in pairs:
        a, b = find(pair['a']), find(pair['b'])
        parents[b] = a
    components = {}
    for index in range(len(records)):
        components.setdefault(find(index), []).append(index)
    excluded, groups = {}, []
    for members in components.values():
        if len(members) < 2:
            continue
        tests = [i for i in members if records[i]['split'] == 'test']
        if tests:
            kept = tests  # held-out data not curated according to model outcomes
        else:
            candidates = [i for i in members if records[i]['split'] == 'val'] or members
            kept = [min(candidates, key=lambda i: records[i]['image'])]
        group_id = f'similarity-{len(groups):04d}'
        groups.append({'group_id': group_id, 'members': members, 'kept': kept})
        for i in members:
            if i not in kept:
                excluded[i] = group_id
    return excluded, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--phash-distance', type=int, default=6)
    args = parser.parse_args()
    if not 0 <= args.phash_distance <= 63:
        parser.error('phash-distance must be in [0,63]')
    source, output = args.source.resolve(), args.output.resolve()
    provenance = json.loads((source / 'provenance.json').read_text(encoding='utf-8'))
    hashes = {r['local_path']: r['sha256'] for r in provenance['files']}
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'source_manifest_sha256': file_hash(source / 'provenance.json'),
              'phash_distance': args.phash_distance, 'records': [], 'label_changes': [],
              'limitations': ['Original video/location groups remain unknown.',
                              'pHash candidates may be false positives; visually distinct related frames may be missed.',
                              'Conservative quarantine is not a leakage-free benchmark certification.']}
    try:
        clean_labels = {}
        for split in ('train', 'val', 'test'):
            for upstream in provenance['selection'][split]:
                image = source / 'images' / split / Path(upstream).name
                label = source / 'labels' / split / (image.stem + '.txt')
                for path in (image, label):
                    if file_hash(path) != hashes[path.relative_to(ROOT).as_posix()]:
                        raise ValueError(f'Raw provenance mismatch: {path}')
                record = {'image': str(image), 'label': str(label), 'split': split,
                          'image_sha256': file_hash(image), 'source_label_sha256': file_hash(label),
                          'source_group_id': None, **image_signature(image)}
                rows = []
                for number, row in enumerate(label.read_text(encoding='utf-8').splitlines(), 1):
                    if not row.strip():
                        continue
                    clipped, changed = clip_yolo_row(row)
                    rows.append(clipped)
                    if changed:
                        report['label_changes'].append({'label': str(label), 'line': number,
                                                        'before': row, 'after': clipped})
                clean_labels[str(image)] = '\n'.join(rows) + ('\n' if rows else '')
                report['records'].append(record)
        pairs = similarity_pairs(report['records'], args.phash_distance)
        excluded, groups = quarantine(report['records'], pairs)
        report.update(similarity_pairs=pairs, similarity_groups=groups,
                      excluded_counts=dict(Counter(report['records'][i]['split'] for i in excluded)))
        for split in ('train', 'val', 'test'):
            (output / 'images' / split).mkdir(parents=True)
            (output / 'labels' / split).mkdir(parents=True)
        for index, record in enumerate(report['records']):
            record['included'] = index not in excluded
            record['quarantine_group'] = excluded.get(index)
            if index in excluded:
                continue
            image = Path(record['image'])
            destination = output / 'images' / record['split'] / image.name
            label = output / 'labels' / record['split'] / (image.stem + '.txt')
            shutil.copy2(image, destination)
            label.write_text(clean_labels[str(image)], encoding='utf-8')
            record.update(prepared_image=str(destination), prepared_label_sha256=file_hash(label))
        (output / 'data.yaml').write_text('path: ' + json.dumps(output.as_posix()) +
            '\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: smoke\n  1: fire\n', encoding='utf-8')
        from training.baseline import audit_dataset
        audit = audit_dataset(output / 'data.yaml')
        write_json(output / 'audit.json', audit)
        retained_cross = [p for p in pairs if p['cross_split'] and p['a'] not in excluded and p['b'] not in excluded]
        if retained_cross:
            raise AssertionError('A screened cross-split pair was retained')
        # A small contact sheet allows inspecting candidates without exposing test scenes.
        from PIL import Image, ImageDraw
        visible = [p for p in pairs if all(report['records'][p[k]]['split'] != 'test' for k in ('a','b'))][:12]
        if visible:
            sheet = Image.new('RGB', (640, 205 * len(visible)), 'white')
            draw = ImageDraw.Draw(sheet)
            for row, pair in enumerate(visible):
                for column, key in enumerate(('a','b')):
                    record = report['records'][pair[key]]
                    with Image.open(record['image']) as im:
                        thumb = im.convert('RGB'); thumb.thumbnail((310,175))
                    x, y = column * 320, row * 205
                    sheet.paste(thumb, (x,y+25))
                    draw.text((x+3,y+3), f"{record['split']} {Path(record['image']).name} d={pair['phash_distance']}", fill='black')
            sheet.save(output / 'similarity_candidates.jpg')
        report.update(status='complete', splits=audit['splits'], retained_cross_split_candidates=0,
                      test_note='Only integrity/signature checks; test images are not shown in review or used for model selection.')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'preparation.json', report)
    print(json.dumps({k:report[k] for k in ('status','splits','excluded_counts')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
