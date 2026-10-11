"""
Full YOLO26n fine-tuning run - heart localization on ACDC.
Uses the patient-disjoint split/dataset already extracted by mri_yolo_dataset.py.
"""
import time

from ultralytics import YOLO

if __name__ == "__main__":
    t0 = time.time()
    model = YOLO("yolo26n.pt")
    results = model.train(
        data="results/mri_yolo_dataset/data.yaml",
        epochs=50,
        patience=15,
        imgsz=320,
        batch=16,
        device=0,
        project="results/mri_yolo_training",
        name="final",
        plots=True,
        verbose=True,
    )
    print(f"YOLO_TRAIN_TOTAL_TIME_SEC {time.time() - t0:.1f}")

    metrics = model.val(data="results/mri_yolo_dataset/data.yaml", split="val")
    print(f"YOLO_FINAL_VAL_mAP50 {metrics.box.map50:.4f}")
    print(f"YOLO_FINAL_VAL_mAP50-95 {metrics.box.map:.4f}")
    print("YOLO_TRAINING_DONE")
