"""Keep one shared validation set uncontaminated by screened old-training overlaps."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.common import file_hash, write_json
from training.baseline import audit_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--old-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepared, output = args.prepared.resolve(), args.output.resolve()
    preparation = json.loads((prepared/'preparation.json').read_text(encoding='utf-8'))
    if preparation['status'] != 'complete':
        raise ValueError('Preparation must be complete')
    old_audit = json.loads(args.old_audit.read_text(encoding='utf-8'))
    old_hashes = {r['image_sha256'] for r in old_audit['files'] if r['split']=='train'}
    records = preparation['records']
    if not old_hashes <= {r['image_sha256'] for r in records}:
        raise ValueError('Expanded raw screening did not cover every old training image')
    forbidden = {i for i,r in enumerate(records) if r['image_sha256'] in old_hashes}
    for group in preparation['similarity_groups']:
        if forbidden.intersection(group['members']):
            forbidden.update(group['members'])
    output.mkdir(parents=True, exist_ok=False)
    report = {'status':'running', 'preparation_sha256':file_hash(prepared/'preparation.json'),
              'old_audit_sha256':file_hash(args.old_audit), 'excluded_validation':[], 'included_validation':[]}
    try:
        (output/'images/val').mkdir(parents=True)
        (output/'labels/val').mkdir(parents=True)
        for i,r in enumerate(records):
            if r['split'] != 'val' or not r['included']:
                continue
            if i in forbidden:
                report['excluded_validation'].append(r['image'])
                continue
            image = Path(r['prepared_image'])
            label = prepared/'labels/val'/image.with_suffix('.txt').name
            if file_hash(image) != r['image_sha256'] or file_hash(label) != r['prepared_label_sha256']:
                raise ValueError('Prepared data changed')
            shutil.copy2(image, output/'images/val'/image.name)
            shutil.copy2(label, output/'labels/val'/label.name)
            report['included_validation'].append(str(image))
        (output/'data.yaml').write_text('path: '+json.dumps(output.as_posix()) +
            '\ntrain: '+json.dumps((prepared/'images/train').as_posix()) +
            '\nval: images/val\ntest: '+json.dumps((prepared/'images/test').as_posix()) +
            '\nnames:\n  0: smoke\n  1: fire\n', encoding='utf-8')
        audit = audit_dataset(output/'data.yaml')
        write_json(output/'audit.json', audit)
        report.update(status='complete', splits=audit['splits'],
                      limitation='No screened similarity group overlaps old training; unknown video grouping and missed similarities remain possible.')
    except Exception as error:
        report.update(status='failed', error=str(error)); raise
    finally:
        write_json(output/'comparison_preparation.json', report)
    print(json.dumps({'status':report['status'], 'val_images':report['splits']['val']['images'],
                      'old_training_overlap_exclusions':len(report['excluded_validation'])}))


if __name__ == '__main__':
    main()
