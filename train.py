#!/usr/bin/env python3
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../Datas/01_active_training/sup_chip_joint_scratch_v1/data.yaml')
    parser.add_argument('--device', default='0')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--project', default='../results/sup_chip_joint_scratch_v1')
    parser.add_argument('--name', default='yolov8m_scratch')
    args = parser.parse_args()
    YOLO('yolov8m.yaml').train(
        data=args.data, device=args.device, epochs=args.epochs, project=args.project,
        name=args.name, pretrained=False, patience=80, batch=16, imgsz=640,
        optimizer='SGD', lr0=0.01, lrf=0.01, close_mosaic=30, mixup=0.2,
        cutmix=0.6, seed=20260914,
    )


if __name__ == '__main__':
    main()
