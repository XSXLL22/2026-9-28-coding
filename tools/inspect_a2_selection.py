"""Inspect locally generated checkpoints to distinguish pre/post-closure selection."""
import argparse
import csv
from pathlib import Path

from runtime.common import configure_ultralytics, file_hash, write_json


def infer_selection(metrics, rows):
    """Match checkpoint metrics to CSV's six-significant-digit serialization.

    final_eval overwrites best.pt train_results with the complete last.pt history.
    Its history length is NOT the checkpoint selection epoch. Ambiguity stays null.
    """
    keys = ('metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)')
    if any(key not in metrics for key in keys):
        return {'epoch': None, 'matching_epochs': [], 'method': 'missing_checkpoint_metrics'}
    matches = [int(row['epoch']) for row in rows if all(row[key] == format(float(metrics[key]), '.6g') for key in keys)]
    return {'epoch': matches[0] if len(matches) == 1 else None, 'matching_epochs': matches,
            'method': 'inferred_from_unique_checkpoint_metrics_match_to_rounded_csv'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='experiments/p2_a2_close_mosaic')
    args = parser.parse_args()
    output = Path(args.run)/'selection_verified.json'
    if output.exists(): raise FileExistsError(output)
    configure_ultralytics()
    import torch
    torch.set_num_threads(1)
    reference_path = Path('experiments/p2_expanded_train/fit/weights/best.pt')
    candidate_path = Path(args.run)/'train/fit/weights/best.pt'
    # Trusted checkpoints produced locally by this project's completed training.
    reference = torch.load(reference_path, map_location='cpu', weights_only=False)
    candidate = torch.load(candidate_path, map_location='cpu', weights_only=False)
    a, b = reference['model'].state_dict(), candidate['model'].state_dict()
    differences = sorted(key for key in a.keys() | b.keys() if key not in a or key not in b or not torch.equal(a[key], b[key]))
    reference_selection = infer_selection(reference.get('train_metrics', {}), list(csv.DictReader((reference_path.parent.parent/'results.csv').open(encoding='utf-8'))))
    candidate_selection = infer_selection(candidate.get('train_metrics', {}), list(csv.DictReader((candidate_path.parent.parent/'results.csv').open(encoding='utf-8'))))
    epoch = candidate_selection['epoch']
    write_json(output, {'status': 'complete', 'reference_sha256': file_hash(reference_path), 'candidate_sha256': file_hash(candidate_path),
                        'reference_selection': reference_selection,
                        'candidate_selection': candidate_selection,
                        'selected_after_mosaic_closure': epoch > 40 if epoch is not None else None,
                        'model_state_dict_equal': not differences, 'different_state_entries': differences,
                        'note': 'Checkpoint history length is NOT a selection epoch: final_eval overwrites it. Epoch is an inference from unique stored metrics/CSV correspondence; ambiguity returns null. Tensor equality compares parameters and buffers.'})


if __name__ == '__main__': main()
