#!/usr/bin/env python3
import pandas as pd
import os
import xlsxwriter

# Define paths
OUTDIR = "../results/final_package"
os.makedirs(OUTDIR, exist_ok=True)

# Input CSV files
distances_csv = "../results/distances.csv"
min_distances_csv = "../results/min_distances.csv"
near_misses_csv = "../results/near_misses.csv"

# Images to embed
IMAGE_FILES = {
    "Top-Down Trajectory": "traj_topdown.png",
    "3D SLAM Trajectory": "traj_3d.png",
    "Fusion Tracks": "fusion_tracks.png",
    "Boxes Snapshot": "boxes_snapshot.png",
    "Near-Miss Summary": "near_miss_events.png"
}

# Output Excel file
report_path = os.path.join(OUTDIR, "report.xlsx")

# Create workbook
workbook = xlsxwriter.Workbook(report_path)
ws = workbook.add_worksheet("Summary")

title_format = workbook.add_format({
    'bold': True,
    'font_size': 16
})

header_format = workbook.add_format({
    'bold': True,
    'font_size': 12
})

ws.write(0, 0, "SLAM + YOLO Tracking Report", title_format)
ws.write(2, 0, "This report summarizes:", header_format)
ws.write(3, 0, "- SLAM Trajectories")
ws.write(4, 0, "- YOLO Object Tracks in World Coordinates")
ws.write(5, 0, "- Bounding Box Reconstruction")
ws.write(6, 0, "- Object Distances & Near-Miss Detection")
ws.write(7, 0, "- Key Visualizations Embedded Below")

row = 10

# Insert images
for label, filename in IMAGE_FILES.items():
    img_path = os.path.join(OUTDIR, filename)
    ws.write(row, 0, label, header_format)
    ws.write(row + 1, 0, img_path)
    
    if os.path.exists(img_path):
        ws.insert_image(row, 2, img_path, {
            'x_scale': 0.4,
            'y_scale': 0.4
        })
    else:
        ws.write(row + 1, 2, "(Image Missing)")
    row += 12

# Add data sheets
def add_csv_sheet(workbook, sheet_name, csv_path):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    ws_data = workbook.add_worksheet(sheet_name)

    # Write header
    for col, name in enumerate(df.columns):
        ws_data.write(0, col, name, header_format)

    # Write data rows
    for r in range(len(df)):
        for c in range(len(df.columns)):
            ws_data.write(r+1, c, df.iloc[r, c])

    # Auto-adjust columns
    for c in range(len(df.columns)):
        ws_data.set_column(c, c, max(12, len(df.columns[c]) + 2))

add_csv_sheet(workbook, "Distances", distances_csv)
add_csv_sheet(workbook, "Min Distances", min_distances_csv)
add_csv_sheet(workbook, "Near Misses", near_misses_csv)

workbook.close()
print("✔ Excel report created:", report_path)

