import cv2
import numpy as np

# ---------------- Configuration ----------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MARKER_DICT = cv2.aruco.DICT_5X5_1000
APPROACH_AREA_THRESHOLD = 3000  # smaller => marker is farther away
CENTER_OFFSET_THRESHOLD = 10     # pixels

PRINT_EVERY_N_FRAMES = 30  # debug: state log cadence (frames)
# dictionary fallback logging is event-driven (success/failure after N frames)
LOG_LOST_EVERY_N_FRAMES = 60


# ---------------- ArUco Setup ----------------
# Debug fallback: try a few common dictionaries because a mismatch leads to
# ids=None and rejected_count=0 (no candidates proposed).
DICTIONARY_CANDIDATES = [
    cv2.aruco.DICT_5X5_1000,
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_6X6_250,
    cv2.aruco.DICT_7X7_1000,
]

parameters = cv2.aruco.DetectorParameters()

# ---- Detector tuning (robust defaults) ----
# These values are intentionally conservative; they improve detection
# under blur/noise/uneven lighting without drastically increasing false positives.
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10
parameters.minMarkerPerimeterRate = 0.0015
parameters.maxMarkerPerimeterRate = 20.0
parameters.polygonalApproxAccuracyRate = 0.02
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX



def make_detector(marker_dict):
    aruco_dict_local = cv2.aruco.getPredefinedDictionary(marker_dict)
    # Create detector bound to this dictionary.
    return cv2.aruco.ArucoDetector(aruco_dict_local, parameters)




# start with the user-selected dictionary (kept as first candidate)
aruco_dict = cv2.aruco.getPredefinedDictionary(MARKER_DICT)

# Pre-create detectors once per candidate dictionary to avoid allocations in-loop
DETECTORS_BY_DICT = {
    d: make_detector(d) for d in [MARKER_DICT, *DICTIONARY_CANDIDATES]
}
# start with the primary dictionary
detector = DETECTORS_BY_DICT[MARKER_DICT]


cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

# ---- Camera exposure robustness ----
# Your diagnostics showed near-black frames (very low gray mean/std),
# so we try to force the camera to increase exposure.
# Values are camera-dependent; these are safe starting points.
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # enable/steer auto exposure (vendor-specific)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)        # lower/higher values vary by camera
cap.set(cv2.CAP_PROP_GAIN, 0)            # reduce gain; optional per camera


frame_count = 0
last_printed_ids = None
last_notfound_signature = None
ema_box_area = None  # exponential moving average of marker bounding-box area
EMA_ALPHA = 0.25  # smoothing factor (0..1). Higher => faster reaction.
consecutive_frames_without_ids = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    h, w = frame.shape[:2]
    center_x = w // 2
    center_y = h // 2

    # Draw center crosshair
    cv2.drawMarker(
        frame,
        (center_x, center_y),
        (0, 255, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=25,
        thickness=2,
    )

    # ---- Preprocessing for detection ----
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Conditional contrast enhancement: only apply when the frame looks low-contrast
    # (common cause of rejected_count staying high / ids=None).
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    if std_val < 25:  # heuristic threshold; avoids messing with well-contrasted frames
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Note: keep blur disabled by default; it can hurt edge/thresholding.



    # Try multiple dictionaries and pick the "best" result instead of stopping
    # at first non-None. This reduces flicker when a dictionary intermittently
    # yields a small/unstable candidate set.
    #
    # Score: (num_ids, -num_rejected)
    best = {
        "score": (-1, 0),
        "corners": None,
        "ids": None,
        "rejected": None,
        "det": detector,
        "dict": MARKER_DICT,
    }

    for d in [MARKER_DICT, *DICTIONARY_CANDIDATES]:
        det = DETECTORS_BY_DICT[d]

        c_try, ids_try, rej_try = det.detectMarkers(gray)

        # ids_try is (N,1) when found, otherwise None
        num_ids = 0 if ids_try is None else int(len(ids_try))
        num_rej = 0 if rej_try is None else int(len(rej_try))

        # If we have any ids, prefer the dictionary with more ids; if none,
        # prefer the one that produces fewer rejected candidates.
        score = (num_ids, -num_rej)
        if score > best["score"]:
            best.update({
                "score": score,
                "corners": c_try,
                "ids": ids_try,
                "rejected": rej_try,
                "det": det,
                "dict": d,
            })


    corners, ids, rejected = best["corners"], best["ids"], best["rejected"]
    detector = best["det"]

    if ids is None:
        consecutive_frames_without_ids += 1
        if PRINT_EVERY_N_FRAMES and (frame_count % LOG_LOST_EVERY_N_FRAMES == 0):
            rejected_count_try = 0 if rejected is None else len(rejected)
            signature = (rejected_count_try, consecutive_frames_without_ids, best["dict"])
            if signature != last_notfound_signature:
                last_notfound_signature = signature
                print(
                    f"Aruco not found yet: dict={best['dict']} rejected_count={rejected_count_try} (frames_without_ids={consecutive_frames_without_ids})"
                )
    else:
        consecutive_frames_without_ids = 0
        last_notfound_signature = None
        if PRINT_EVERY_N_FRAMES and (frame_count % PRINT_EVERY_N_FRAMES == 0):
            ids_list = ids.flatten().tolist() if ids is not None else None
            print(f"Aruco detected: dict={best['dict']} ids={ids_list}")




    # ------------ Console output (throttled / state-based) ------------
    # Keep this lightweight: we already log on loss/detection events above.
    if PRINT_EVERY_N_FRAMES and (frame_count % PRINT_EVERY_N_FRAMES == 0) and (ids is not None):
        ids_list = ids.flatten().tolist() if ids is not None else None
        rejected_count = 0 if rejected is None else len(rejected)
        signature = (tuple(ids_list) if ids_list is not None else None, rejected_count)
        if signature != last_printed_ids:
            print("Aruco state:", {"ids": ids_list, "rejected_count": rejected_count})
            last_printed_ids = signature



    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        pts = corners[0][0]  # (4, 2) array of corner points

        marker_x = int(np.mean(pts[:, 0]))
        marker_y = int(np.mean(pts[:, 1]))

        # Draw centroid
        cv2.circle(frame, (marker_x, marker_y), 5, (0, 0, 255), -1)

        # Error vector
        cv2.line(frame, (center_x, center_y), (marker_x, marker_y), (255, 0, 0), 2)

        dx = marker_x - center_x
        dy = marker_y - center_y

        commands = []

        if dx < -CENTER_OFFSET_THRESHOLD:
            commands.append("MOVE LEFT")
        elif dx > CENTER_OFFSET_THRESHOLD:
            commands.append("MOVE RIGHT")

        if dy < -CENTER_OFFSET_THRESHOLD:
            commands.append("MOVE UP")
        elif dy > CENTER_OFFSET_THRESHOLD:
            commands.append("MOVE DOWN")

        # Marker size metric for APPROACH logic
        # Use bounding-box area (more stable than polygon area for perspective changes).
        xs = pts[:, 0]
        ys = pts[:, 1]
        box_area = int((xs.max() - xs.min()) * (ys.max() - ys.min()))

        # Smooth the metric across frames to prevent threshold jitter.
        if ema_box_area is None:
            ema_box_area = box_area
        else:
            ema_box_area = (EMA_ALPHA * box_area) + ((1.0 - EMA_ALPHA) * ema_box_area)

        if ema_box_area < APPROACH_AREA_THRESHOLD:
            commands.append("APPROACH")


        if abs(dx) <= CENTER_OFFSET_THRESHOLD and abs(dy) <= CENTER_OFFSET_THRESHOLD:
            text = "LOCK ENGAGED"
            color = (0, 255, 0)
        else:
            text = " | ".join(commands) if commands else "TRACKING"
            color = (0, 0, 255)

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
    else:
        # reset size smoothing when the marker is lost
        ema_box_area = None

        # lighting/contrast diagnostics (helps explain persistent rejected_count=0)
        mean_val = float(np.mean(gray)) if gray is not None else 0.0

        std_val = float(np.std(gray)) if gray is not None else 0.0

        cv2.putText(
            frame,
            "TARGET LOST",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        min_val = float(np.min(gray)) if gray is not None else 0.0
        max_val = float(np.max(gray)) if gray is not None else 0.0

        cv2.putText(
            frame,
            f"gray mean={mean_val:.1f} std={std_val:.1f}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )
        # quick signal metric: unique grayscale values in a small downsample
        # (helps confirm camera feed isn't nearly constant/black)
        small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
        unique_count = int(np.unique(small).size)

        cv2.putText(
            frame,
            f"gray min={min_val:.0f} max={max_val:.0f} unique={unique_count}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

    frame_count += 1

    cv2.imshow("ArUco Tracking System", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()


