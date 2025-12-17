#!/usr/bin/env python3
"""
scripts/step_g_generate_pdf_report.py
Generates slam_fusion_report.pdf from saved outputs.
"""

import os
import csv
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle, Paragraph, SimpleDocTemplate, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image as PILImage

ROOT = "."
OUT_PDF = "slam_fusion_report.pdf"

# Files we expect
files = {
    "topdown": os.path.join(ROOT, "fused_topdown.png"),
    "drift":    os.path.join(ROOT, "slam_drift.png"),
    "fused":    os.path.join(ROOT, "fused.ply"),
    "report":   os.path.join(ROOT, "slam_report.csv"),
    "fused_path": os.path.join(ROOT, "fused_path.csv"),
    "mp4":      os.path.join(ROOT, "fused_rotating.mp4"),
}

def scaled_image(path, max_w_mm=170, max_h_mm=120):
    """
    Returns a (reportlab Image, w, h) scaled to fit in mm.
    """
    if not os.path.exists(path):
        return None, 0, 0
    im = PILImage.open(path)
    px_w, px_h = im.size
    dpi = im.info.get("dpi", (96,96))[0]
    # fallback dpi -> approximate 96 dpi
    if not dpi or dpi == 0:
        dpi = 96.0
    # convert px -> mm: mm = px / dpi * 25.4
    mm_w = px_w / dpi * 25.4
    mm_h = px_h / dpi * 25.4
    scale = min(max_w_mm / mm_w, max_h_mm / mm_h, 1.0)
    disp_w = mm_w * scale * mm
    disp_h = mm_h * scale * mm
    rl_img = Image(path, width=disp_w, height=disp_h)
    return rl_img, disp_w, disp_h

def read_csv_report(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r") as f:
        r = csv.reader(f)
        for row in r:
            rows.append(row)
    return rows

def read_fused_path(path, max_rows=10):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r") as f:
        r = csv.reader(f)
        for i, row in enumerate(r):
            if i >= max_rows:
                break
            rows.append(row)
    return rows

def build_pdf():
    doc = SimpleDocTemplate(OUT_PDF, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    flow = []

    # Title
    h = Paragraph("<b>SLAM Fusion Report</b>", styles['Title'])
    flow.append(h)
    flow.append(Spacer(1, 6*mm))

    # Summary box
    summary_lines = []
    summary_lines.append("This report contains results from the multi-agent SLAM fusion pipeline.")
    if os.path.exists(files["fused"]):
        summary_lines.append("Fused point cloud: %s" % files["fused"])
    if os.path.exists(files["mp4"]):
        summary_lines.append("Rotation video: %s" % files["mp4"])
    summary = Paragraph("<br/>".join(summary_lines), styles['Normal'])
    flow.append(summary)
    flow.append(Spacer(1, 6*mm))

    # Add topdown image
    img_top, w, h = scaled_image(files["topdown"], max_w_mm=170, max_h_mm=120)
    if img_top:
        flow.append(Paragraph("<b>Fused Top-down Map</b>", styles['Heading2']))
        flow.append(img_top)
        flow.append(Spacer(1, 6*mm))
    else:
        flow.append(Paragraph("<b>Fused Top-down Map</b> (missing)", styles['Heading2']))

    # Add drift plot
    img_drift, w2, h2 = scaled_image(files["drift"], max_w_mm=170, max_h_mm=90)
    if img_drift:
        flow.append(Paragraph("<b>SLAM Drift Over Time</b>", styles['Heading2']))
        flow.append(img_drift)
        flow.append(Spacer(1, 6*mm))
    else:
        flow.append(Paragraph("<b>SLAM Drift Over Time</b> (missing)", styles['Heading2']))

    # Add small table: slam_report.csv
    flow.append(Paragraph("<b>Per-agent statistics</b>", styles['Heading2']))
    rows = read_csv_report(files["report"])
    if rows:
        # render header bold
        data = []
        for i, r in enumerate(rows):
            if i == 0:
                data.append([Paragraph("<b>%s</b>"%c, styles['Normal']) for c in r])
            else:
                data.append(r)
        table = Table(data, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
        ]))
        flow.append(table)
        flow.append(Spacer(1, 6*mm))
    else:
        flow.append(Paragraph("No slam_report.csv found.", styles['Normal']))

    # Add snippet of fused_path.csv
    flow.append(Paragraph("<b>Fused trajectory (first rows)</b>", styles['Heading2']))
    path_rows = read_fused_path(files["fused_path"], max_rows=12)
    if path_rows:
        table = Table(path_rows, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('FONTSIZE', (0,0), (-1,-1), 9),
        ]))
        flow.append(table)
    else:
        flow.append(Paragraph("No fused_path.csv found.", styles['Normal']))

    flow.append(Spacer(1, 8*mm))
    flow.append(Paragraph("Generated by the Thesis pipeline.", styles['Italic']))

    doc.build(flow)
    print("Saved", OUT_PDF)

if __name__ == "__main__":
    build_pdf()

