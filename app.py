import sys, subprocess, importlib

# pin versions that work on Streamlit Cloud
REQUIRED = [
    ("opencv-python-headless==4.9.0.80", "cv2"),
    ("numpy==1.26.4", "numpy"),
    ("pandas==2.2.2", "pandas"),
    ("matplotlib==3.8.4", "matplotlib"),
    ("streamlit==1.36.0", "streamlit"),
]

def _ensure_deps():
    missing = []
    for pip_spec, mod_name in REQUIRED:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            missing.append(pip_spec)
    if missing:
        # install quietly; first run may take ~1–2 min
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-warn-script-location",
             "--disable-pip-version-check", "--quiet", *missing]
        )
        # import after install to populate globals (so later code can use them)
        for pip_spec, mod_name in REQUIRED:
            globals()[mod_name] = importlib.import_module(mod_name)

_ensure_deps()

import os, io, math, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------
# Core algorithm (locked-in)
# ----------------------------

def segment_green(frame):
    """Robust segmentation on green channel with denoise + Otsu + cleanup."""
    g = frame[:, :, 1]
    g = cv2.bilateralFilter(g, d=7, sigmaColor=30, sigmaSpace=7)
    _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def feret_max_on_points(pts, n_angles=720):
    """
    Dense 'rotating calipers' to get Feret maximum (longest chord).
    Returns (L_px, (p1, p2), theta_max)
    """
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    best_rng = -1.0
    best_pair = None
    best_theta = 0.0
    for th in angles:
        d = np.array([np.cos(th), np.sin(th)])
        proj = pts @ d
        i_lo = int(np.argmin(proj))
        i_hi = int(np.argmax(proj))
        rng = float(proj[i_hi] - proj[i_lo])
        if rng > best_rng:
            best_rng = rng
            best_pair = (pts[i_lo], pts[i_hi])
            best_theta = th
    return best_rng, best_pair, best_theta

def chord_length_at(mask, center_point, dir_perp, half_len=800, step=0.5):
    """
    Length of the chord across 'mask' along dir_perp passing through center_point.
    Returns (length_px, p1, p2)
    """
    h, w = mask.shape
    t_vals = np.arange(-half_len, half_len + step, step)
    xs = center_point[0] + t_vals * dir_perp[0]
    ys = center_point[1] + t_vals * dir_perp[1]
    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)
    vals = mask[yi, xi] > 0
    if not vals.any():
        return 0.0, None, None
    c = len(t_vals) // 2
    Lidx = c
    while Lidx > 0 and vals[Lidx - 1]:
        Lidx -= 1
    Ridx = c
    while Ridx < len(vals) - 1 and vals[Ridx + 1]:
        Ridx += 1
    p1 = (xs[Lidx], ys[Lidx])
    p2 = (xs[Ridx], ys[Ridx])
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1])), p1, p2

def measure_single_frame_true_waist(img_bgr, central_zone_frac=0.30, n_angles=720):
    """
    Measure L (Feret max on contour) and W (minimum chord inside mask),
    but restrict W search to ±(central_zone_frac) of the Feret length around the centroid (Option A).
    Returns dict with L_px, W_px, ratio, annotated image, centroid, axis info.
    """
    mask = segment_green(img_bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    cont = cnt.reshape(-1, 2).astype(np.float64)
    mean = cont.mean(axis=0)

    # Longest axis (Feret max) and its angle
    Lpx, (p1L, p2L), theta_max = feret_max_on_points(cont, n_angles=n_angles)
    u = np.array([np.cos(theta_max), np.sin(theta_max)])  # major axis dir
    u /= (np.linalg.norm(u) + 1e-9)
    vdir = np.array([-u[1], u[0]])  # perpendicular

    # CENTRAL ZONE restriction for width (± central_zone_frac of Feret length)
    half_center = central_zone_frac * (Lpx / 2.0)
    # sample density proportional to L for stability
    samples = max(21, int(Lpx * 0.6))
    s_vals = np.linspace(-half_center, half_center, samples)

    best_w = 1e18
    best_seg = None
    for s in s_vals:
        cpt = (mean[0] + s * u[0], mean[1] + s * u[1])
        wlen, a, b = chord_length_at(mask, cpt, vdir, half_len=800, step=0.5)
        if wlen > 0 and wlen < best_w:
            best_w = wlen
            best_seg = (a, b)

    if best_seg is None:
        return None

    Wpx = float(best_w)
    pW1, pW2 = best_seg

    # Centroid
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    else:
        cx = cy = np.nan

    # Annotate
    ann = img_bgr.copy()
    cv2.polylines(ann, [cnt.astype(np.int32)], True, (255, 255, 255), 2)
    cv2.line(ann, (int(p1L[0]), int(p1L[1])), (int(p2L[0]), int(p2L[1])), (255, 255, 255), 3)
    cv2.line(ann, (int(pW1[0]), int(pW1[1])), (int(pW2[0]), int(pW2[1])), (255, 255, 255), 3)
    ratio = Lpx / max(Wpx, 1e-6)
    cv2.putText(ann, f"L:{Lpx:.1f}px  W:{Wpx:.1f}px  L/W:{ratio:.2f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return {
        "L_px": float(Lpx),
        "W_px": float(Wpx),
        "ratio": float(ratio),
        "centroid": (float(cx), float(cy)),
        "theta_max": float(theta_max),
        "annotated": ann,
    }

def process_video_true_waist(
    video_bytes,
    minutes_per_frame=10.0,
    undeformed_L0_um=20.0,
    central_zone_frac=0.30,
    n_angles=720,
    codec="mp4v"
):
    """
    Full video pipeline:
      - Reads video from bytes
      - For each frame: compute L, W, ratio with central-zone restriction
      - Writes annotated video
      - Returns dataframe and file paths
    """
    # write temp input video
    tmp_dir = tempfile.mkdtemp(prefix="cell_LW_")
    in_path = os.path.join(tmp_dir, "input.mp4")
    with open(in_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video.")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_meta = cap.get(cv2.CAP_PROP_FPS)
    if not fps_meta or fps_meta <= 1e-6:
        fps_meta = 5.0  # reasonable default for timelapse/ROI clips

    out_path = os.path.join(tmp_dir, "annotated_true_waist.mp4")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    vout = cv2.VideoWriter(out_path, fourcc, fps_meta, (W, H))

    records = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        res = measure_single_frame_true_waist(
            frame, central_zone_frac=central_zone_frac, n_angles=n_angles
        )

        ann = frame.copy()
        if res is not None:
            ann = res["annotated"]
            records.append(
                dict(
                    frame=frame_idx,
                    L_px=res["L_px"],
                    W_px=res["W_px"],
                    ratio=res["ratio"],
                    cx_px=res["centroid"][0],
                    cy_px=res["centroid"][1],
                )
            )
        else:
            records.append(
                dict(
                    frame=frame_idx,
                    L_px=np.nan,
                    W_px=np.nan,
                    ratio=np.nan,
                    cx_px=np.nan,
                    cy_px=np.nan,
                )
            )
        vout.write(ann)

    cap.release()
    vout.release()

    df = pd.DataFrame(records)
    if not df.empty:
        df["time_min"] = (df["frame"] - df["frame"].min()) * float(minutes_per_frame)

        # Calibration: pick undeformed frames as top quartile of W, compute L0(px) median
        if df["W_px"].notna().sum() > 0:
            q75 = df["W_px"].quantile(0.75)
            undeformed = df[df["W_px"] >= q75]
            L0_px = float(undeformed["L_px"].median()) if not undeformed.empty else np.nan
        else:
            L0_px = np.nan

        scale_um_per_px = (
            float(undeformed_L0_um) / L0_px if (L0_px == L0_px and L0_px > 0) else np.nan
        )
        df["L_um"] = df["L_px"] * scale_um_per_px
        df["W_um"] = df["W_px"] * scale_um_per_px
    else:
        L0_px = np.nan
        scale_um_per_px = np.nan
        df["time_min"] = []

    csv_path = os.path.join(tmp_dir, "true_waist_metrics.csv")
    df.to_csv(csv_path, index=False)

    # quick plots
    plot1 = os.path.join(tmp_dir, "L_W_vs_time_px.png")
    plot2 = os.path.join(tmp_dir, "L_over_W_vs_time.png")
    if not df.empty:
        plt.figure(figsize=(6, 3.5))
        plt.plot(df["time_min"], df["L_px"], "o-", label="L (px)")
        plt.plot(df["time_min"], df["W_px"], "o-", label="W (px)")
        plt.xlabel("Time (min)")
        plt.ylabel("Pixels")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot1, dpi=180)
        plt.close()

        plt.figure(figsize=(6, 3.5))
        plt.plot(df["time_min"], df["ratio"], "o-")
        plt.xlabel("Time (min)")
        plt.ylabel("L/W")
        plt.tight_layout()
        plt.savefig(plot2, dpi=180)
        plt.close()

    return {
        "tmp_dir": tmp_dir,
        "annotated_video": out_path,
        "csv": csv_path,
        "plot_LW": plot1,
        "plot_ratio": plot2,
        "L0_px": L0_px,
        "scale_um_per_px": scale_um_per_px,
    }

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Cell L/W Deformation Analyzer", layout="centered")
st.title("Cell L/W Deformation Analyzer (central-zone waist)")

st.markdown(
    """
This app measures **L (longest axis)** and **W (true waist)** for green-channel cell videos/images, 
using the locked-in algorithm:

- **L** = Feret maximum on the *actual contour*  
- **W** = minimum chord **inside the mask**, but **restricted** to the cell **central zone** (±30% of L)  
- Calibration: assumes **undeformed L₀ = 20 µm** (from wide frames, top-quartile W)  
"""
)

with st.sidebar:
    st.header("Analysis Settings")
    minutes_per_frame = st.number_input("Minutes per frame", min_value=0.1, value=10.0, step=0.1)
    undeformed_L0_um = st.number_input("Undeformed length L₀ (µm)", min_value=0.1, value=20.0, step=0.1)
    central_zone_frac = st.slider("Central zone ± (fraction of L)", 0.10, 0.50, 0.30, 0.05)
    n_angles = st.select_slider("Angular resolution", options=[360, 720, 1440], value=720)
    st.caption("Higher angular resolution = slightly more accurate but slower.")

tab1, tab2 = st.tabs(["📽 Video", "🖼 Single Image"])

with tab1:
    st.subheader("Analyze a Video")
    up = st.file_uploader("Upload MP4/AVI video", type=["mp4", "avi", "mov"], accept_multiple_files=False)
    if up is not None:
        if st.button("Run analysis on video"):
            with st.spinner("Processing…"):
                result = process_video_true_waist(
                    up.read(),
                    minutes_per_frame=minutes_per_frame,
                    undeformed_L0_um=undeformed_L0_um,
                    central_zone_frac=central_zone_frac,
                    n_angles=int(n_angles),
                )
            st.success("Done!")

            st.write(f"**Baseline L₀(px)**: {result['L0_px']:.2f}")
            st.write(f"**Scale** (µm/px): {result['scale_um_per_px']:.6f}")

            st.video(result["annotated_video"])
            st.image(result["plot_LW"], caption="L & W vs time (px)", use_column_width=True)
            st.image(result["plot_ratio"], caption="L/W vs time", use_column_width=True)

            with open(result["csv"], "rb") as f:
                st.download_button("⬇️ Download CSV", f, file_name="true_waist_metrics.csv", mime="text/csv")

            with open(result["annotated_video"], "rb") as f:
                st.download_button("⬇️ Download annotated video", f, file_name="annotated_true_waist.mp4", mime="video/mp4")

with tab2:
    st.subheader("Measure a Single Image")
    up_img = st.file_uploader("Upload image (PNG/JPG/TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=False)
    if up_img is not None:
        file_bytes = np.asarray(bytearray(up_img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        res = measure_single_frame_true_waist(img_bgr, central_zone_frac=central_zone_frac, n_angles=int(n_angles))
        if res is None:
            st.error("No cell detected. Try a clearer ROI or adjust segmentation.")
        else:
            st.write(f"**L (px)**: {res['L_px']:.1f}")
            st.write(f"**W (px)**: {res['W_px']:.1f}")
            st.write(f"**L/W**: {res['ratio']:.2f}")

            # preview PNG
            ok, buf = cv2.imencode(".png", res["annotated"])
            if ok:
                st.image(buf.tobytes(), caption="Annotated image", use_column_width=True)
                st.download_button("⬇️ Download annotated image", buf.tobytes(), file_name="annotated_true_waist.png", mime="image/png")

