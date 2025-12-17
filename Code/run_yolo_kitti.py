import os
import glob
import pandas as pd
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import json

INPUT_DIR = "data/kitti/sequences/00/image_0"
OUTPUT_DIR = "experiments/yolo/kitti_00"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/annotated", exist_ok=True)

# Load YOLO
model = YOLO("yolov8s.pt")

############################################################
# Only process VALID KITTI SEQ-00 frames (000000–004540)
############################################################
images = []
for i in range(4541):
    f = f"{INPUT_DIR}/{i:06d}.png"
    if os.path.exists(f):
        images.append(f)

print(f"Processing {len(images)} KITTI frames...")

all_dets = []
json_output = {}

for img_path in tqdm(images, desc="Running YOLO on KITTI"):
    img = cv2.imread(img_path)
    frame_id = int(os.path.basename(img_path).split(".")[0])

    results = model(img, verbose=False)[0]

    dets_frame = []
    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        xyxy = box.xyxy[0].tolist()

        dets_frame.append({
            "frame": frame_id,
            "cls": cls,
            "conf": conf,
            "x1": xyxy[0],
            "y1": xyxy[1],
            "x2": xyxy[2],
            "y2": xyxy[3]
        })

        all_dets.append(dets_frame[-1])

    json_output[str(frame_id)] = dets_frame

    # Save annotated frame
    annotated = results.plot()
    cv2.imwrite(f"{OUTPUT_DIR}/annotated/{frame_id:06d}.jpg", annotated)

# Save CSV
df = pd.DataFrame(all_dets)
df.to_csv(f"{OUTPUT_DIR}/detections.csv", index=False)

# Save JSON
with open(f"{OUTPUT_DIR}/detections.json", "w") as f:
    json.dump(json_output, f, indent=4)

print("\n=== YOLO COMPLETE ===")
print(f"CSV saved: {OUTPUT_DIR}/detections.csv")
print(f"Annotated images saved to: {OUTPUT_DIR}/annotated/")
import os
import glob
import pandas as pd
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import json

INPUT_DIR = "data/kitti/sequences/00/image_0"
OUTPUT_DIR = "experiments/yolo/kitti_00"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/annotated", exist_ok=True)

# Load YOLO
model = YOLO("yolov8s.pt")

all_dets = []
json_output = {}

images = sorted(glob.glob(f"{INPUT_DIR}/*.png"))

for img_path in tqdm(images, desc="Running YOLO on KITTI"):
    img = cv2.imread(img_path)
    frame_id = int(os.path.basename(img_path).split(".")[0])

    results = model(img, verbose=False)[0]

    dets_frame = []
    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        xyxy = box.xyxy[0].tolist()

        dets_frame.append({
            "frame": frame_id,
            "cls": cls,
            "conf": conf,
            "x1": xyxy[0],
            "y1": xyxy[1],
            "x2": xyxy[2],
            "y2": xyxy[3]
        })

        all_dets.append(dets_frame[-1])

    json_output[str(frame_id)] = dets_frame

    # Save annotated frame
    annotated = results.plot()
    cv2.imwrite(f"{OUTPUT_DIR}/annotated/{frame_id:06d}.jpg", annotated)

# Save CSV
df = pd.DataFrame(all_dets)
df.to_csv(f"{OUTPUT_DIR}/detections.csv", index=False)

# Save JSON
with open(f"{OUTPUT_DIR}/detections.json", "w") as f:
    json.dump(json_output, f, indent=4)

print("\n=== YOLO COMPLETE ===")
print(f"CSV saved: {OUTPUT_DIR}/detections.csv")
print(f"Annotated images saved to: {OUTPUT_DIR}/annotated/")

