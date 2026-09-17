#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def grabcut(image):
    height, width = image.shape[:2]
    margin = max(3, min(height, width) // 20)
    labels = np.zeros((height, width), np.uint8)
    cv2.grabCut(image, labels, (margin, margin, width - 2 * margin, height - 2 * margin), np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_RECT)
    return np.where((labels == 1) | (labels == 3), 255, 0).astype(np.uint8)


def dark_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    signal = cv2.normalize(cv2.subtract(cv2.GaussianBlur(gray, (0, 0), 31), gray), None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(signal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= max(64, mask.size // 300):
            cleaned[labels == index] = 255
    return cleaned


def count_components(mask):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    minimum = max(64, mask.size // 300)
    return sum(stats[index, cv2.CC_STAT_AREA] >= minimum for index in range(1, count))


def passes_quality(mask):
    binary = mask > 0
    foreground = np.count_nonzero(binary)
    fraction = foreground / mask.size
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    areas = [stats[index, cv2.CC_STAT_AREA] for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= max(64, mask.size // 300)]
    border = np.concatenate((binary[:2].ravel(), binary[-2:].ravel(), binary[:, :2].ravel(), binary[:, -2:].ravel()))
    return bool(areas) and 0.05 <= fraction <= 0.55 and max(areas) / foreground >= 0.70 and np.count_nonzero(border) / foreground <= 0.02


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', default='../results/buchong824_pipeline/prediction')
    parser.add_argument('--output', default='../results/buchong824_pipeline/count')
    parser.add_argument('--min-iou', type=float, default=0.60)
    args = parser.parse_args()
    prediction, output = Path(args.prediction).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((prediction / 'detections.csv').open(encoding='utf-8')))
    image_totals, batch_totals = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    for index, row in enumerate(rows, 1):
        image = cv2.imread(str(prediction / row['crop']))
        mask = grabcut(image)
        other = dark_mask(image)
        intersection = np.count_nonzero((mask > 0) & (other > 0))
        union = np.count_nonzero((mask > 0) | (other > 0))
        score = intersection / max(1, union)
        cells = count_components(mask)
        status = 'reliable' if score >= args.min_iou and cells > 0 and passes_quality(mask) else 'review_required'
        mask_path = Path('masks') / status / row['batch'] / Path(row['image']).stem / Path(row['crop']).name
        (output / mask_path).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(output / mask_path), mask)
        row.update({'estimated_cells': cells, 'mask_status': status, 'method_iou': f'{score:.6f}', 'mask': str(mask_path)})
        key = (row['batch'], row['image'])
        for totals in (image_totals[key], batch_totals[row['batch']]):
            totals['detections'] += 1; totals['estimated_cells'] += cells
            totals[f'{status}_detections'] += 1; totals[f'{status}_cells'] += cells
        print(f'[{index}/{len(rows)}] {row["batch"]}/{row["image"]} #{row["detection_id"]}: {cells} ({status})')
    fields = list(rows[0])
    with (output / 'per_detection.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary_fields = ['detections', 'estimated_cells', 'reliable_detections', 'reliable_cells', 'review_required_detections', 'review_required_cells']
    image_rows = [{'batch': key[0], 'image': key[1], **{field: values.get(field, 0) for field in summary_fields}} for key, values in sorted(image_totals.items())]
    batch_rows = [{'batch': key, **{field: values.get(field, 0) for field in summary_fields}} for key, values in sorted(batch_totals.items())]
    for name, data, fields in [('per_image.csv', image_rows, ['batch', 'image', *summary_fields]), ('per_batch.csv', batch_rows, ['batch', *summary_fields])]:
        with (output / name).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(data)
    total = {field: sum(row.get(field, 0) for row in batch_rows) for field in summary_fields}
    with (output / 'total.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields); writer.writeheader(); writer.writerow(total)


if __name__ == '__main__':
    main()
