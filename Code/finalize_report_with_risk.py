#!/usr/bin/env python3
import pandas as pd
import os
import xlsxwriter
from datetime import datetime

OUTDIR = "../results/final_package"
os.makedirs(OUTDIR, exist_ok=True)

# Inputs
FILES = {
    "Distances": "../results/distances.csv",
    "Min Distances": "../results/min_distances.csv",
    "Near Misses": "../results/near_misses.csv",
    "Risk Predictions": "../results/final_package/risk_predictions.csv"
}

# Images
IMAGES = [
    ("Top-Down SLAM Trajectory",      "traj_topdown.png"),
    ("3D SLAM Trajectory",            "traj_3d.png"),
    ("SLAM + YOLO Fusion Tracks",     "fusion_tracks.png"),
    ("3D Boxes Snapshot",             "boxes_snapshot.png"),
    ("Near Miss Events",              "near_miss_events.png"),
    ("Risk-Time Scatter Plot",        "risk_plot.png"),
    ("Predicted Future Risk Map",     "risk_map_future.png")
]

report_path = os.path.join(OUTDIR, "report.xlsx")
workbook = xlsxwriter.Workbook(report_path)

# ——————————————————————————————————————————————
# STYLES
# ——————————————————————————————————————————————
title = workbook.add_format({
    'bold': True, 'font_size': 24
})
subtitle = workbook.add_format({
    'bold': True, 'font_size': 14
})
header = workbook.add_format({
    'bold': True, 'font_size': 11, 'bg_color': '#D9E1F2', 'border': 1
})
center = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
border_box = workbook.add_format({'border': 1})
footer_fmt = workbook.add_format({'align': 'right', 'italic': True, 'font_size': 9})

# ——————————————————————————————————————————————
# EXECUTIVE SUMMARY PAGE
# ——————————————————————————————————————————————
ws0 = workbook.add_worksheet("Executive Summary")
ws0.set_column("A:A", 55)
ws0.set_column("B:B", 35)

ws0.write("A1", "Autonomous Navigation: SLAM + YOLO + Behavior + Risk Analysis", title)
ws0.write("A3", f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle)
ws0.write("A5", "Author: Ranjeet", subtitle)

ws0.write("A8", "This report contains:", subtitle)
items = [
    "✓ Full SLAM trajectory visualization (2D + 3D)",
    "✓ YOLO object tracking fused with ego-motion",
    "✓ Near-miss event detection (<5m)",
    "✓ Per-object behavior classification",
    "✓ Collision risk estimation & TTC analysis",
    "✓ Future state prediction (short-horizon)",
]
r = 10
for item in items:
    ws0.write(r, 0, item)
    r += 1

# Footer
ws0.write(40, 1, "Autonomous Perception Report — Page 1", footer_fmt)

# ——————————————————————————————————————————————
# TABLE OF CONTENTS
# ——————————————————————————————————————————————
ws_toc = workbook.add_worksheet("Contents")
ws_toc.set_column("A:A", 40)

ws_toc.write("A1", "Table of Contents", title)

toc_items = [
    ("Executive Summary", "Executive Summary"),
    ("Summary (Images)", "Summary"),
    ("Distances", "Distances"),
    ("Min Distances", "Min Distances"),
    ("Near Misses", "Near Misses"),
    ("Risk Predictions", "Risk Predictions"),
]

row = 4
for name, sheet in toc_items:
    ws_toc.write_url(row, 0, f"internal:'{sheet}'!A1", string=f"→ {name}")
    row += 2

ws_toc.write(40, 0, "Autonomous Perception Report — Page 2", footer_fmt)

# ——————————————————————————————————————————————
# SUMMARY PAGE WITH ALL IMAGES (PROFESSIONAL)
# ——————————————————————————————————————————————
ws = workbook.add_worksheet("Summary")
ws.set_column("A:A", 38)
ws.set_column("B:B", 55)

ws.write("A1", "Summary of Visual Outputs", title)

row = 4
for label, filename in IMAGES:
    img_path = os.path.join(OUTDIR, filename)
    
    ws.write(row, 0, label, subtitle)

    # Allocate a nice image container (merged 12 rows)
    ws.merge_range(row, 1, row + 11, 1, "", border_box)

    if os.path.exists(img_path):
        ws.insert_image(row, 1, img_path, {
            'x_scale': 0.48,
            'y_scale': 0.48,
            'x_offset': 8,
            'y_offset': 5
        })
    else:
        ws.write(row + 3, 1, "(missing image)", center)

    row += 14

# Footer
ws.write(100, 1, "Autonomous Perception Report — Page 3", footer_fmt)

# ——————————————————————————————————————————————
# DATA SHEETS
# ——————————————————————————————————————————————
for sheet_name, csv_path in FILES.items():

    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)
    ws2 = workbook.add_worksheet(sheet_name)
    ws2.set_default_row(18)

    for col in range(len(df.columns)):
        ws2.set_column(col, col, max(12, len(df.columns[col]) + 3))

    # Header
    for c, name in enumerate(df.columns):
        ws2.write(0, c, name, header)

    # Data
    for r in range(len(df)):
        for c in range(len(df.columns)):
            ws2.write(r + 1, c, df.iloc[r, c])

    # Conditional Formatting
    if "min_distance" in df.columns:
        col_idx = df.columns.tolist().index("min_distance")
        ws2.conditional_format(1, col_idx, len(df), col_idx, {
            'type': '3_color_scale'
        })

    if "risk_class" in df.columns:
        col_idx = df.columns.tolist().index("risk_class")
        ws2.conditional_format(1, col_idx, len(df), col_idx, {
            'type': 'text',
            'criteria': 'containing',
            'value': 'HIGH',
            'format': workbook.add_format({'bg_color': '#FFC7CE'})
        })

    ws2.write(len(df) + 4, len(df.columns) - 1, f"Page: {sheet_name}", footer_fmt)

workbook.close()
print("✔ Professional report created:", report_path)
#!/usr/bin/env python3
import pandas as pd
import os
import xlsxwriter

OUTDIR = "../results/final_package"
os.makedirs(OUTDIR, exist_ok=True)

# Inputs
FILES = {
    "distances": "../results/distances.csv",
    "min_distances": "../results/min_distances.csv",
    "near_misses": "../results/near_misses.csv",
    "risk_predictions": "../results/final_package/risk_predictions.csv"
}

# Images (thumbnails)
IMAGES = [
    ("Top-Down Trajectory",      "traj_topdown.png"),
    ("3D Trajectory",            "traj_3d.png"),
    ("Fusion Tracks",            "fusion_tracks.png"),
    ("Boxes Snapshot",           "boxes_snapshot.png"),
    ("Near-Miss Summary",        "near_miss_events.png"),
    ("Risk Plot",                "risk_plot.png"),
    ("Future Risk Map",          "risk_map_future.png")
]

report_path = os.path.join(OUTDIR, "report.xlsx")
workbook = xlsxwriter.Workbook(report_path)

# Styles
title = workbook.add_format({'bold': True, 'font_size': 18})
subtitle = workbook.add_format({'bold': True, 'font_size': 13})
header = workbook.add_format({'bold': True, 'font_size': 11})
center = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

# Summary sheet
ws = workbook.add_worksheet("Summary")

ws.set_column("A:A", 35)     # description
ws.set_column("B:B", 60)     # image area
ws.set_default_row(30)

ws.write("A1", "SLAM + YOLO Full Report", title)
ws.write("A3", "This summary contains:", subtitle)
ws.write("A4", "✓ SLAM Trajectories")
ws.write("A5", "✓ YOLO 3D Tracks")
ws.write("A6", "✓ Near-Miss Analysis")
ws.write("A7", "✓ Behavior Classification")
ws.write("A8", "✓ Risk Prediction & Collision Forecast")

row = 10

# Insert images cleanly
for label, filename in IMAGES:
    img_path = os.path.join(OUTDIR, filename)
    
    # Label cell
    ws.write(row, 0, label, subtitle)
    
    # Reserve area for image (merge 10 rows)
    ws.merge_range(row, 1, row + 9, 1, "", center)

    if os.path.exists(img_path):
        ws.insert_image(row, 1, img_path, {
            'x_scale': 0.45,
            'y_scale': 0.45,
            'x_offset': 10,
            'y_offset': 5
        })
    else:
        ws.write(row + 3, 1, "(missing image)")

    row += 12

# Add data sheets
for sheet_name, csv_path in FILES.items():
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)
    ws2 = workbook.add_worksheet(sheet_name)

    # Set nice widths
    for col in range(len(df.columns)):
        ws2.set_column(col, col, max(12, len(df.columns[col]) + 3))

    # Write headers
    for c, name in enumerate(df.columns):
        ws2.write(0, c, name, header)

    # Write data
    for r in range(len(df)):
        for c in range(len(df.columns)):
            ws2.write(r + 1, c, df.iloc[r, c])

workbook.close()
print("✔ Clean, aligned Excel report created:", report_path)
#!/usr/bin/env python3
import pandas as pd
import os
import xlsxwriter

OUTDIR = "../results/final_package"
os.makedirs(OUTDIR, exist_ok=True)

# Paths
distances = "../results/distances.csv"
min_dist = "../results/min_distances.csv"
near = "../results/near_misses.csv"
risk = "../results/final_package/risk_predictions.csv"
risk_plot = "../results/final_package/risk_plot.png"
risk_map = "../results/final_package/risk_map_future.png"

# Images to embed (if present)
IMAGES = {
    "Top-Down Trajectory": os.path.join(OUTDIR, "traj_topdown.png"),
    "3D Trajectory": os.path.join(OUTDIR, "traj_3d.png"),
    "Fusion Tracks": os.path.join(OUTDIR, "fusion_tracks.png"),
    "Boxes Snapshot": os.path.join(OUTDIR, "boxes_snapshot.png"),
    "Near-Miss Summary": os.path.join(OUTDIR, "near_miss_events.png"),
    "Risk Plot": risk_plot,
    "Risk Map (Future)": risk_map
}

report_path = os.path.join(OUTDIR, "report.xlsx")
wb = xlsxwriter.Workbook(report_path)

# Summary sheet
ws = wb.add_worksheet("Summary")
title_fmt = wb.add_format({'bold': True, 'font_size': 16})
hdr_fmt = wb.add_format({'bold': True, 'font_size': 12})
ws.write(0,0,"FINAL SLAM + YOLO Report (with Risk)", title_fmt)
ws.write(2,0,"Summary:", hdr_fmt)
ws.write(3,0,"This workbook contains trajectory, detection, tracking, near-miss analysis, behavior classification and risk predictions. See data sheets for numeric outputs and thumbnails for quick visual reference.")

row = 8
for label, imgpath in IMAGES.items():
    ws.write(row, 0, label, hdr_fmt)
    ws.write(row+1, 0, imgpath)
    if os.path.exists(imgpath):
        try:
            ws.insert_image(row, 2, imgpath, {'x_scale':0.35, 'y_scale':0.35})
        except Exception as e:
            ws.write(row,2, "(could not insert)")
    else:
        ws.write(row+1, 2, "(missing)")
    row += 10

# helper to add CSV as sheet
def add_csv(sheetname, csvpath):
    if not os.path.exists(csvpath):
        return
    df = pd.read_csv(csvpath)
    ws2 = wb.add_worksheet(sheetname)
    # header
    for c, col in enumerate(df.columns):
        ws2.write(0, c, col, hdr_fmt)
    # rows
    for r in range(len(df)):
        for c in range(len(df.columns)):
            val = df.iloc[r, c]
            # try to write numeric types as numbers
            try:
                ws2.write(r+1, c, float(val))
            except:
                ws2.write(r+1, c, str(val))
    # set column widths
    for c, col in enumerate(df.columns):
        width = max(12, min(50, len(col) + 6))
        ws2.set_column(c, c, width)

# Add data sheets
add_csv("distances", distances)
add_csv("min_distances", min_dist)
add_csv("near_misses", near)
add_csv("risk_predictions", risk)

wb.close()
print("Saved final Excel report:", report_path)

