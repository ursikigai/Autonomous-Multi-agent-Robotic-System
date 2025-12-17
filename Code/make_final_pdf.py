#!/usr/bin/env python3
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# Base directory where images are stored
BASE = "/Users/ranjeet/thesis_multiagent/results/final_package"

OUT_PDF = os.path.join(BASE, "final_report.pdf")

styles = getSampleStyleSheet()
title = styles["Title"]
heading = styles["Heading2"]
normal = styles["BodyText"]

doc = SimpleDocTemplate(
    OUT_PDF,
    pagesize=A4,
    rightMargin=40,
    leftMargin=40,
    topMargin=40,
    bottomMargin=40
)

story = []

# ------------------------------
#  TITLE PAGE
# ------------------------------
story.append(Paragraph("Autonomous Navigation Full Report", title))
story.append(Spacer(1, 20))
story.append(Paragraph("SLAM + YOLO + Behavior + Risk Analysis", heading))
story.append(Spacer(1, 40))

story.append(Paragraph("Author: Ranjeet", normal))
story.append(Paragraph("Generated Automatically", normal))
story.append(PageBreak())

# ------------------------------
#  IMAGES TO INCLUDE
# ------------------------------
images = [
    ("Top-Down SLAM Trajectory", "traj_topdown.png"),
    ("3D SLAM Trajectory", "traj_3d.png"),
    ("SLAM + YOLO Fusion Tracks", "fusion_tracks.png"),
    ("Object Boxes Snapshot", "boxes_snapshot.png"),
    ("Near-Miss Summary", "near_miss_events.png"),
    ("Risk-Time Scatter Plot", "risk_plot.png"),
    ("Predicted Future Risk Map", "risk_map_future.png")
]

# ------------------------------
#  ADD IMAGES
# ------------------------------
for label, imgname in images:
    path = os.path.join(BASE, imgname)
    
    story.append(Paragraph(label, heading))
    story.append(Spacer(1, 10))
    
    if os.path.exists(path):
        # Scale to fit A4 width cleanly
        story.append(Image(path, width=6.5*inch, height=4.5*inch))
    else:
        story.append(Paragraph("(Missing image)", normal))
    
    story.append(Spacer(1, 20))
    story.append(PageBreak())

# Build PDF
doc.build(story)
print("PDF saved to:", OUT_PDF)
