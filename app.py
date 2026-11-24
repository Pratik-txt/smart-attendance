import os
import math
import random
import sqlite3
from datetime import datetime, timezone, timedelta
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp
from PIL import Image, ImageOps
from deepface import DeepFace
from streamlit_geolocation import streamlit_geolocation

# --- CONFIGURATION ---
CLASSROOM_LAT = 29.870309763764837       # Replace with your classroom latitude
CLASSROOM_LON = 77.89623227358898        # Replace with your classroom longitude
ALLOWED_RADIUS_METERS = 150.0 # Tighten to 50.0m once testing on phones in class
SPOOF_SCORE_LIMIT = 45        # Blocks phone/tablet screens at >= 45/100 spoof risk
FACE_MATCH_LIMIT = 0.58       # Cosine distance threshold for eye-aligned SFace crops
FACES_DIR = "registered_faces"
DB_FILE = "attendance.db"

CHALLENGES = {
    "BLINK": "Close BOTH eyes completely",
    "MOUTH": "Open your mouth wide (Say 'Ahhh')",
    "TURN": "Turn your head clearly to the left or right"
}

st.set_page_config(page_title="Smart Attendance System", layout="centered")

if "challenge" not in st.session_state:
    st.session_state.challenge = random.choice(list(CHALLENGES.keys()))

def reset_challenge():
    choices = [k for k in CHALLENGES if k != st.session_state.get("challenge")]
    st.session_state.challenge = random.choice(choices)

# --- DATABASE SETUP (SQLite locally + Google Sheets on Streamlit Cloud) ---
def init_sqlite():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            student_id TEXT,
            timestamp TEXT,
            latitude REAL,
            longitude REAL,
            distance_m REAL,
            status TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_attendance(student_id, lat, lon, distance_m):
    ist = timezone(timedelta(hours=5, minutes=30))
    timestamp = datetime.now(ist).strftime("%Y-%m-%d %I:%M:%S %p")
    record = pd.DataFrame([{
        "student_id": student_id,
        "timestamp": timestamp,
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "distance_m": round(distance_m, 1),
        "status": "PRESENT"
    }])

    conn = sqlite3.connect(DB_FILE)
    record.to_sql("attendance", conn, if_exists="append", index=False)
    conn.close()

    try:
        has_gsheets = "connections" in st.secrets and "gsheets" in st.secrets["connections"]
    except Exception:
        has_gsheets = False

    if has_gsheets:
        from streamlit_gsheets import GSheetsConnection
        gs_conn = st.connection("gsheets", type=GSheetsConnection)
        existing = gs_conn.read(ttl=0)
        updated = pd.concat([existing, record], ignore_index=True)
        gs_conn.update(data=updated)

# --- GEOFENCING (Browser Location) ---
def haversine_meters(lat1, lon1, lat2, lon2):
    r = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# --- OPTICAL SCREEN-SPOOF & 3D LIVENESS DETECTION ---
try:
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh

def detect_screen_spoof(img_bgr, face_bbox):
    h, w, _ = img_bgr.shape
    fx1, fy1, fx2, fy2 = face_bbox
    face_w = max(1, fx2 - fx1)
    face_h = max(1, fy2 - fy1)

    spoof_score = 0
    reasons = []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Straight phone-edge detector in the border band immediately outside the face
    pad_x, pad_y = int(face_w * 0.65), int(face_h * 0.45)
    ox1, oy1 = max(0, fx1 - pad_x), max(0, fy1 - pad_y)
    ox2, oy2 = min(w, fx2 + pad_x), min(h, fy2 + pad_y)

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 135)
    border_edges = np.zeros_like(edges)
    border_edges[oy1:oy2, ox1:ox2] = edges[oy1:oy2, ox1:ox2]
    ix1, iy1 = max(0, fx1 - int(face_w * 0.08)), max(0, fy1 - int(face_h * 0.08))
    ix2, iy2 = min(w, fx2 + int(face_w * 0.08)), min(h, fy2 + int(face_h * 0.08))
    border_edges[iy1:iy2, ix1:ix2] = 0

    min_len = max(35, int(face_h * 0.42))
    lines = cv2.HoughLinesP(border_edges, 1, np.pi / 180, threshold=40, minLineLength=min_len, maxLineGap=12)
    straight_count = 0
    if lines is not None:
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]
            angle = abs(math.degrees(math.atan2(y2_l - y1_l, x2_l - x1_l)))
            if (65 <= angle <= 115) or (angle <= 25) or (angle >= 155):
                straight_count += 1

    if straight_count >= 2:
        spoof_score += 45
        reasons.append(f"straight phone edges ({straight_count})")

    # 2. Uncompressed 2D FFT subpixel screen-grid check
    face_gray = gray[fy1:fy2, fx1:fx2]
    if face_gray.size > 0:
        resized = cv2.resize(face_gray, (128, 128)).astype(np.float32)
        window = np.outer(np.hanning(128), np.hanning(128))
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(resized * window)))
        yy, xx = np.ogrid[:128, :128]
        dist_center = np.sqrt((xx - 64) ** 2 + (yy - 64) ** 2)
        mid_high_band = fft_mag[(dist_center >= 16) & (dist_center <= 56)]
        if mid_high_band.size > 0:
            peak_ratio = float(np.percentile(mid_high_band, 99.5)) / (float(np.median(mid_high_band)) + 1e-6)
            if peak_ratio > 15.5:
                spoof_score += 35
                reasons.append(f"screen pixel grid ({peak_ratio:.1f})")

    # 3. YCrCb backlight color-gamut compression check
    face_bgr = img_bgr[fy1:fy2, fx1:fx2]
    if face_bgr.size > 0:
        ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
        cr_mean = float(np.mean(ycrcb[:, :, 1]))
        cb_mean = float(np.mean(ycrcb[:, :, 2]))
        if (cr_mean - cb_mean) < 14.5:
            spoof_score += 30
            reasons.append(f"screen backlight tint ({cr_mean - cb_mean:.1f})")

    # 4. Dark phone bezel contrast & specular screen glare
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    face_v = hsv[fy1:fy2, fx1:fx2, 2]
    outer_v = hsv[oy1:oy2, ox1:ox2, 2].copy()
    outer_v[(fy1 - oy1):(fy2 - oy1), (fx1 - ox1):(fx2 - ox1)] = 128
    ring_pixels = outer_v[outer_v != 128]
    if face_v.size > 0 and ring_pixels.size > 0:
        dark_ring_ratio = float(np.mean(ring_pixels < 45))
        if (float(np.mean(face_v)) - float(np.mean(ring_pixels)) > 50.0) and dark_ring_ratio > 0.30:
            spoof_score += 30
            reasons.append("dark phone bezel contrast")

    face_hsv = hsv[fy1:fy2, fx1:fx2]
    if face_hsv.size > 0:
        glare_ratio = float(np.mean((face_hsv[:, :, 2] >= 246) & (face_hsv[:, :, 1] <= 24)))
        if glare_ratio > 0.05:
            spoof_score += 30
            reasons.append("screen glass glare")

    return spoof_score, reasons

def check_background_stability(img1_bgr, img2_bgr, bbox1, bbox2):
    h, w, _ = img1_bgr.shape
    g1 = cv2.resize(cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY), (160, 120)).astype(np.float32)
    g2 = cv2.resize(cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY), (160, 120)).astype(np.float32)

    mask = np.ones((120, 160), dtype=bool)
    for (x1, y1, x2, y2) in (bbox1, bbox2):
        sx1 = max(0, int((x1 / w) * 160) - 8)
        sy1 = max(0, int((y1 / h) * 120) - 8)
        sx2 = min(160, int((x2 / w) * 160) + 8)
        sy2 = min(120, int((y2 / h) * 120) + 8)
        mask[sy1:sy2, sx1:sx2] = False

    bg1, bg2 = g1[mask], g2[mask]
    if bg1.size < 200 or np.std(bg1) < 2.0 or np.std(bg2) < 2.0:
        return 1.0
    corr = float(np.corrcoef(bg1, bg2)[0, 1])
    return 1.0 if np.isnan(corr) else corr

def analyze_face(image_pil):
    img_rgb = np.array(image_pil.convert("RGB"))
    img_bgr = np.ascontiguousarray(img_rgb[:, :, ::-1])
    h, w, _ = img_rgb.shape

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as mesh:
        results = mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark

        # Scale coordinates by (w, h) so portrait mobile cameras & landscape laptops match identically
        def ear(idx):
            p = [np.array([lm[i].x * w, lm[i].y * h]) for i in idx]
            v1 = np.linalg.norm(p[1] - p[5])
            v2 = np.linalg.norm(p[2] - p[4])
            h_dist = np.linalg.norm(p[0] - p[3])
            return (v1 + v2) / (2.0 * h_dist + 1e-6)

        avg_ear = (ear([33, 160, 158, 133, 153, 144]) + ear([362, 385, 387, 263, 373, 380])) / 2.0

        lip_v = np.linalg.norm(np.array([lm[13].x * w, lm[13].y * h]) - np.array([lm[14].x * w, lm[14].y * h]))
        lip_h = np.linalg.norm(np.array([lm[78].x * w, lm[78].y * h]) - np.array([lm[308].x * w, lm[308].y * h]))
        mar = lip_v / (lip_h + 1e-6)

        eye_mid_x = (lm[133].x + lm[362].x) / 2.0
        inter_eye_w = abs(lm[362].x - lm[133].x) + 1e-6
        nose_shift = abs(lm[1].x - eye_mid_x) / inter_eye_w

        xs = [pt.x * w for pt in lm]
        ys = [pt.y * h for pt in lm]
        face_bbox = (
            max(0, int(min(xs))),
            max(0, int(min(ys))),
            min(w, int(max(xs))),
            min(h, int(max(ys)))
        )

        # Eye-align and crop face tightly for SFace recognition
        lx, ly = (lm[33].x + lm[133].x) * 0.5 * w, (lm[33].y + lm[133].y) * 0.5 * h
        rx, ry = (lm[362].x + lm[263].x) * 0.5 * w, (lm[362].y + lm[263].y) * 0.5 * h
        if lx > rx:
            lx, rx = rx, lx
            ly, ry = ry, ly

        angle = math.degrees(math.atan2(ry - ly, rx - lx))
        eyes_center = ((lx + rx) * 0.5, (ly + ry) * 0.5)
        rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        aligned_bgr = cv2.warpAffine(img_bgr, rot_mat, (w, h), flags=cv2.INTER_CUBIC)

        pts = np.array([[pt.x * w, pt.y * h, 1.0] for pt in lm])
        rot_pts = pts @ rot_mat.T
        rx1, ry1 = np.min(rot_pts, axis=0)
        rx2, ry2 = np.max(rot_pts, axis=0)
        rw, rh = rx2 - rx1, ry2 - ry1

        cx1 = max(0, int(rx1 - 0.15 * rw))
        cy1 = max(0, int(ry1 - 0.22 * rh))
        cx2 = min(w, int(rx2 + 0.15 * rw))
        cy2 = min(h, int(ry2 + 0.10 * rh))
        aligned_face = np.ascontiguousarray(aligned_bgr[cy1:cy2, cx1:cx2])

        if aligned_face.size == 0:
            return None

        return {
            "ear": avg_ear,
            "mar": mar,
            "nose_shift": nose_shift,
            "bbox": face_bbox,
            "aligned_face": aligned_face
        }

# --- STREAMLIT UI ---
init_sqlite()
st.title("Smart Proxy-Proof Attendance")

with st.expander("Enroll / Update Reference Face"):
    reg_id = st.text_input("Roll Number to Enroll", value="26114001", key="reg_id").strip()
    reg_photo = st.camera_input("Capture clear front-facing reference photo", key="reg_cam")
    if st.button("Save Reference Photo"):
        if reg_id and reg_photo:
            os.makedirs(FACES_DIR, exist_ok=True)
            save_path = os.path.join(FACES_DIR, f"{reg_id}.jpg")
            ImageOps.exif_transpose(Image.open(reg_photo)).convert("RGB").save(save_path, format="JPEG")
            st.success(f"Saved reference photo to {save_path}")
        else:
            st.warning("Enter a Roll Number and capture a photo first.")

student_id = st.text_input("Enter Student ID / Roll Number", value="26114001").strip()

st.subheader("Step 1: Verify Classroom Location")
loc = streamlit_geolocation()
location_ok = False
distance = None

if loc and loc.get("latitude") is not None:
    user_lat, user_lon = loc["latitude"], loc["longitude"]
    distance = haversine_meters(user_lat, user_lon, CLASSROOM_LAT, CLASSROOM_LON)
    if distance <= ALLOWED_RADIUS_METERS:
        st.success(f"Inside classroom zone ({distance:.1f} m away)")
        location_ok = True
    else:
        st.error(f"Outside allowed area ({distance:.1f} m away; max {ALLOWED_RADIUS_METERS:.0f} m)")
else:
    st.info("Tap the location icon above and allow browser GPS access.")

st.subheader("Step 2: Identity & Active Liveness Check")
col1, col2 = st.columns([3, 1])
with col1:
    st.warning(f"Active Challenge: **{CHALLENGES[st.session_state.challenge]}**")
with col2:
    if st.button("New Challenge"):
        reset_challenge()
        st.rerun()

front_photo = st.camera_input("1. Neutral face: Look straight at the camera with eyes open & mouth closed")
liveness_photo = st.camera_input(f"2. Perform challenge: {CHALLENGES[st.session_state.challenge]}")

if st.button("Mark Attendance", type="primary"):
    if not student_id:
        st.warning("Please enter your Student ID.")
    elif not location_ok:
        st.error("Location check failed. You must be inside the allowed area.")
    elif not front_photo or not liveness_photo:
        st.warning("Please capture both camera photos.")
    else:
        ref_path = os.path.join(FACES_DIR, f"{student_id}.jpg")
        if not os.path.exists(ref_path):
            st.error(f"No enrolled photo found at '{ref_path}'.")
        else:
            with st.spinner("Running optical anti-spoof & DeepFace verification..."):
                img1 = ImageOps.exif_transpose(Image.open(front_photo))
                img2 = ImageOps.exif_transpose(Image.open(liveness_photo))
                ref_img = ImageOps.exif_transpose(Image.open(ref_path))

                img1_bgr = np.ascontiguousarray(np.array(img1.convert("RGB"))[:, :, ::-1])
                img2_bgr = np.ascontiguousarray(np.array(img2.convert("RGB"))[:, :, ::-1])

                pose1 = analyze_face(img1)
                pose2 = analyze_face(img2)
                ref_pose = analyze_face(ref_img)

                if not pose1 or not pose2:
                    st.error("Face not clearly detected in one of the camera frames. Try again.")
                elif not ref_pose:
                    st.error(f"Could not detect a clear face in '{ref_path}'. Re-enroll above.")
                else:
                    score1, reasons1 = detect_screen_spoof(img1_bgr, pose1["bbox"])
                    score2, reasons2 = detect_screen_spoof(img2_bgr, pose2["bbox"])
                    max_spoof = max(score1, score2)
                    all_reasons = list(dict.fromkeys(reasons1 + reasons2))

                    bg_corr = check_background_stability(img1_bgr, img2_bgr, pose1["bbox"], pose2["bbox"])
                    if bg_corr < 0.72:
                        max_spoof += 30
                        all_reasons.append(f"held device/background shift ({bg_corr:.2f})")

                    if max_spoof >= SPOOF_SCORE_LIMIT:
                        st.error(f"Phone / Screen Spoof Blocked! (Risk: {max_spoof}/100 — {', '.join(all_reasons)})")
                    else:
                        neutral_ok = (pose1["ear"] >= 0.19) and (pose1["mar"] < 0.35) and (pose1["nose_shift"] <= 0.28)

                        ch = st.session_state.challenge
                        if ch == "BLINK":
                            challenge_ok = pose2["ear"] < 0.18
                        elif ch == "MOUTH":
                            challenge_ok = pose2["mar"] >= 0.45
                        else:
                            challenge_ok = pose2["nose_shift"] >= 0.38

                        if not neutral_ok:
                            st.error(f"Photo 1 failed (Eyes: {pose1['ear']:.2f}, Mouth: {pose1['mar']:.2f}, Turn: {pose1['nose_shift']:.2f}). Hold phone at eye level with eyes open and mouth closed.")
                        elif not challenge_ok:
                            st.error(f"Liveness check failed! You did not complete: {CHALLENGES[ch]}.")
                        else:
                            ref_match = DeepFace.verify(
                                img1_path=pose1["aligned_face"],
                                img2_path=ref_pose["aligned_face"],
                                model_name="SFace",
                                detector_backend="skip",
                                enforce_detection=False,
                                distance_metric="cosine"
                            )

                            if ref_match["distance"] <= FACE_MATCH_LIMIT:
                                save_attendance(student_id, loc["latitude"], loc["longitude"], distance)
                                reset_challenge()
                                st.success(f"Attendance marked for {student_id}! (Match: {ref_match['distance']:.2f}, Spoof Risk: {max_spoof}/100)")
                            else:
                                st.error(f"Face mismatch! Score {ref_match['distance']:.2f} exceeded limit {FACE_MATCH_LIMIT:.2f}.")

with st.expander("View Attendance Records"):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM attendance ORDER BY timestamp DESC", conn)
    conn.close()
    st.dataframe(df, use_container_width=True)