#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def positions(length, tile, overlap):
    step = int(tile * (1 - overlap))
    values = list(range(0, max(1, length - tile + 1), step))
    values.append(max(0, length - tile))
    return sorted(set(values))


def nms(rows, threshold):
    kept = []
    for cls in sorted(set(row['class_id'] for row in rows)):
        pending = sorted((row for row in rows if row['class_id'] == cls), key=lambda row: row['confidence'], reverse=True)
        while pending:
            best = pending.pop(0)
            kept.append(best)
            x1, y1, x2, y2 = (best[key] for key in ('x1', 'y1', 'x2', 'y2'))
            area = (x2 - x1) * (y2 - y1)
            survivors = []
            for row in pending:
                xx1, yy1 = max(x1, row['x1']), max(y1, row['y1'])
                xx2, yy2 = min(x2, row['x2']), min(y2, row['y2'])
                intersection = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                other_area = (row['x2'] - row['x1']) * (row['y2'] - row['y1'])
                if intersection / max(1, area + other_area - intersection) < threshold:
                    survivors.append(row)
            pending = survivors
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='../Datas/05_inference_and_intermediate/buchong-824')
    parser.add_argument('--weights', default='../120/results/yolov8m/yolov8m/weights/best.pt')
    parser.add_argument('--output', default='../results/buchong824_pipeline/prediction')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--tile', type=int, default=1024)
    parser.add_argument('--overlap', type=float, default=0.20)
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, output_rows = YOLO(args.weights), []
    images = sorted(source.glob('*/*.png'))
    for image_index, path in enumerate(images, 1):
        image = cv2.imread(str(path)); height, width = image.shape[:2]
        tiles = [(image[y:y + args.tile, x:x + args.tile], x, y) for y in positions(height, args.tile, args.overlap) for x in positions(width, args.tile, args.overlap)]
        detections = []
        for result, (_, offset_x, offset_y) in zip(model.predict([item[0] for item in tiles], imgsz=args.tile, conf=args.conf, device=args.device, verbose=False), tiles):
            for box, cls, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box
                detections.append({'class_id': int(cls), 'confidence': float(confidence), 'x1': max(0, x1 + offset_x), 'y1': max(0, y1 + offset_y), 'x2': min(width, x2 + offset_x), 'y2': min(height, y2 + offset_y)})
        detections = nms(detections, 0.50)
        preview, proliferating_id = image.copy(), 0
        for row in detections:
            x1, y1, x2, y2 = (int(round(row[key])) for key in ('x1', 'y1', 'x2', 'y2'))
            color = [(0, 0, 255), (0, 200, 255), (255, 0, 0)][row['class_id']]
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 3)
            cv2.putText(preview, f'{model.names[row["class_id"]]} {row["confidence"]:.2f}', (x1, max(25, y1 - 6)), 0, 0.7, color, 2)
            if row['class_id'] != 2:
                continue
            proliferating_id += 1
            pad_x, pad_y = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
            crop = image[max(0, y1 - pad_y):min(height, y2 + pad_y), max(0, x1 - pad_x):min(width, x2 + pad_x)]
            crop_path = Path('crops') / path.parent.name / path.stem / f'{path.stem}_prolif_{proliferating_id:03d}.png'
            (output / crop_path).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(output / crop_path), crop)
            output_rows.append({'batch': path.parent.name, 'image': path.name, 'detection_id': proliferating_id, 'confidence': f'{row["confidence"]:.6f}', 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'crop': str(crop_path)})
        preview_path = output / 'visualizations' / path.parent.name / path.name
        preview_path.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(preview_path), preview)
        print(f'[{image_index}/{len(images)}] {path.parent.name}/{path.name}: {proliferating_id}')
    fields = ['batch', 'image', 'detection_id', 'confidence', 'x1', 'y1', 'x2', 'y2', 'crop']
    with (output / 'detections.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(output_rows)


if __name__ == '__main__':
    main()
