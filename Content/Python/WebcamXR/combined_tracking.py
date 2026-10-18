import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

# --------------------------------------------------
# FILES AND CONNECTION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
FACE_MODEL = MODEL_DIR / "face_landmarker.task"

# Defaults; override on the command line (see parse_args).
OSC_IP = "127.0.0.1"
OSC_PORT = 7000

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

for model in (HAND_MODEL, FACE_MODEL):
    if not model.exists():
        raise FileNotFoundError(f"Model file not found: {model}")

# Created in main() once the command line is parsed.
osc = None

mp_base = mp.tasks.BaseOptions
mp_vision = mp.tasks.vision
mp_running = mp_vision.RunningMode

# --------------------------------------------------
# HAND TRACKING
# --------------------------------------------------

hand_options = mp_vision.HandLandmarkerOptions(
    base_options=mp_base(model_asset_path=str(HAND_MODEL)),
    running_mode=mp_running.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# --------------------------------------------------
# FACE TRACKING
# --------------------------------------------------

face_options = mp_vision.FaceLandmarkerOptions(
    base_options=mp_base(model_asset_path=str(FACE_MODEL)),
    running_mode=mp_running.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_facial_transformation_matrixes=True,
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# BGR colors for the preview overlay.
LABEL_COLORS = {
    "right": (0, 255, 0),
    "left": (0, 165, 255),
}

# --------------------------------------------------
# HAND LABEL RESOLUTION
# --------------------------------------------------

# Consecutive frames a single visible hand must report a new label
# before we accept the switch.
LABEL_SWITCH_FRAMES = 6

_label_state = {
    "stable": None,
    "pending": None,
    "pending_count": 0,
    "prev_count": 0,
    "last_wrist_x": {},
}


def resolve_hand_labels(candidates):
    # MediaPipe can confuse left/right when a hand turns sideways.
    # candidates: [(landmarks, label, *extra)] -> same tuples with a
    # stable label; extra items are passed through unchanged.
    state = _label_state

    if len(candidates) >= 2:
        # Two hands: in the mirrored frame the user's right hand is on
        # the right (larger x).
        ordered = sorted(candidates[:2], key=lambda c: c[0][0].x)
        resolved = [
            (ordered[0][0], "left", *ordered[0][2:]),
            (ordered[1][0], "right", *ordered[1][2:]),
        ]
        state["stable"] = None
        state["pending"], state["pending_count"] = None, 0

    elif len(candidates) == 1:
        landmarks, label, *extra = candidates[0]
        wrist_x = landmarks[0].x

        if state["prev_count"] >= 2 and state["last_wrist_x"]:
            # Dropping from two hands to one: keep the hand whose last
            # wrist position is closest.
            last = state["last_wrist_x"]
            label = min(last, key=lambda k: abs(last[k] - wrist_x))
            state["stable"] = label
            state["pending"], state["pending_count"] = None, 0
        elif state["prev_count"] == 1 and state["stable"] is not None:
            if label != state["stable"]:
                if label == state["pending"]:
                    state["pending_count"] += 1
                else:
                    state["pending"], state["pending_count"] = label, 1
                if state["pending_count"] >= LABEL_SWITCH_FRAMES:
                    state["stable"] = label
                    state["pending"], state["pending_count"] = None, 0
            else:
                state["pending"], state["pending_count"] = None, 0
            label = state["stable"]
        else:
            state["stable"] = label
            state["pending"], state["pending_count"] = None, 0

        resolved = [(landmarks, label, *extra)]

    else:
        resolved = []

    state["prev_count"] = len(resolved)
    for hand in resolved:
        state["last_wrist_x"][hand[1]] = hand[0][0].x

    return resolved


def get_hand_candidates(hand_result):
    candidates = []

    for index, landmarks in enumerate(hand_result.hand_landmarks):
        if not (
            hand_result.handedness
            and index < len(hand_result.handedness)
            and hand_result.handedness[index]
        ):
            continue

        label = hand_result.handedness[index][0].category_name.lower()
        if label not in ("left", "right"):
            continue

        # MediaPipe reports inverted handedness on the mirrored frame
        # (verified with test recordings), so swap it.
        label = "right" if label == "left" else "left"

        world = None
        if index < len(hand_result.hand_world_landmarks):
            world = hand_result.hand_world_landmarks[index]

        candidates.append((landmarks, label, world))

    return candidates

# --------------------------------------------------
# FIST (GRAB) DETECTION
# --------------------------------------------------

# Per finger: distance(tip, wrist) / distance(knuckle, wrist), measured on
# MediaPipe world landmarks (metric, independent of hand orientation).
# Extended fingers measure ~1.6-2.2, curled fingers ~0.7-1.0.
FINGER_JOINTS = [(8, 5), (12, 9), (16, 13), (20, 17)]
FINGER_CURLED_BELOW = 1.3
FINGER_EXTENDED_ABOVE = 1.5
# A fist must hold this many consecutive frames before a grab starts. While
# a pointing hand moves, the index finger can look curled for 1-3 frames;
# without this the grab latches (release needs two open fingers) and blocks
# pointing for seconds.
GRAB_CONFIRM_FRAMES = 4

_grab_state = {"left": False, "right": False}
_fist_frames = {"left": 0, "right": 0}


def _distance(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def finger_ratios(world):
    wrist = world[0]
    return [
        _distance(world[tip], wrist)
        / max(_distance(world[knuckle], wrist), 1e-6)
        for tip, knuckle in FINGER_JOINTS
    ]


def _to_unreal(v):
    # MediaPipe world axes on the mirrored frame: x = user's right,
    # y = down, z = away from the camera (toward the user).
    # Unreal pawn axes: X = forward (toward the screen), Y = right, Z = up.
    return (-v[2], v[0], -v[1])


def _sub(a, b):
    return (a.x - b.x, a.y - b.y, a.z - b.z)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(v):
    length = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
    if length < 1e-6:
        return None
    return (v[0] / length, v[1] / length, v[2] / length)


def hand_orientation(world, label):
    # Returns (finger_dir, palm_normal) as unit vectors in Unreal pawn axes,
    # or None. finger_dir: wrist -> middle knuckle. palm_normal: points out
    # of the palm. Side vector runs pinky knuckle -> index knuckle, so the
    # cross product order depends on which hand it is.
    finger = _to_unreal(_sub(world[9], world[0]))
    side = _to_unreal(_sub(world[5], world[17]))
    normal = _cross(finger, side) if label == "right" else _cross(side, finger)
    normal = _normalize(normal)
    if normal is None:
        return None

    dot = sum(f * n for f, n in zip(finger, normal))
    finger = _normalize(tuple(f - dot * n for f, n in zip(finger, normal)))
    if finger is None:
        return None

    return finger, normal


def update_grab_states(hands):
    # Grab starts when all four fingers are curled (so pointing does not
    # grab) and ends once at least two fingers are extended again. The
    # gap between the thresholds prevents flickering.
    visible = set()

    for _, label, world in hands:
        visible.add(label)
        if world is None:
            continue

        ratios = finger_ratios(world)
        if _grab_state[label]:
            extended = sum(r > FINGER_EXTENDED_ABOVE for r in ratios)
            _grab_state[label] = extended < 2
        else:
            fist = all(r < FINGER_CURLED_BELOW for r in ratios)
            _fist_frames[label] = _fist_frames[label] + 1 if fist else 0
            _grab_state[label] = _fist_frames[label] >= GRAB_CONFIRM_FRAMES
        if _grab_state[label]:
            _fist_frames[label] = 0

    for label in _grab_state:
        if label not in visible:
            _grab_state[label] = False
            _fist_frames[label] = 0

    return dict(_grab_state)


# Trigger: while grabbing, extending the index finger "arms" the trigger;
# curling it again fires. Requiring the finger to be extended first stops
# the curled fist that started the grab from firing immediately.
_trigger_state = {
    "left": {"armed": False, "pressed": False},
    "right": {"armed": False, "pressed": False},
}


def update_trigger_states(hands, grabs):
    visible = set()

    for _, label, world in hands:
        visible.add(label)
        state = _trigger_state[label]

        if world is None or not grabs.get(label):
            state["armed"] = state["pressed"] = False
            continue

        index_ratio = finger_ratios(world)[0]
        if index_ratio > FINGER_EXTENDED_ABOVE:
            state["armed"] = True
            state["pressed"] = False
        elif state["armed"] and index_ratio < FINGER_CURLED_BELOW:
            state["pressed"] = True

    for label, state in _trigger_state.items():
        if label not in visible:
            state["armed"] = state["pressed"] = False

    return {label: state["pressed"] for label, state in _trigger_state.items()}


# Pointing (index extended, other fingers curled, not grabbing) held for
# POINT_HOLD_SECONDS aims a teleport; Unreal teleports when it ends.
# The hold time filters out brief, accidental pointing. Only the right hand
# aims, like the Quest teleport stick.
POINT_HOLD_SECONDS = 0.5
POINT_HAND = "right"
# Pointing is judged relative to the other fingers: an index finger aimed
# toward the camera is foreshortened (MediaPipe depth is weak), so its
# ratio can drop to ~1.3, but it still clearly exceeds the curled ones.
POINT_INDEX_ABOVE = 1.25
POINT_START_GAP = 0.4   # index - longest other finger, to start pointing
POINT_KEEP_GAP = 0.2    # looser, to keep pointing once started
POINT_OTHERS_BELOW = 1.4
# Detection dropouts shorter than this do not end the point.
POINT_GRACE_SECONDS = 0.25

_point_state = {
    "left": {"since": None, "last": None},
    "right": {"since": None, "last": None},
}

# Per-frame numbers behind the last point decision for POINT_HAND, shown by
# the preview's point debug panel. None when that hand is not visible.
_point_debug = None


def update_point_states(hands, grabs, now):
    global _point_debug
    _point_debug = None
    visible = set()

    for _, label, world in hands:
        visible.add(label)
        state = _point_state[label]
        pointing = False
        if label == POINT_HAND:
            _point_debug = {"grabbing": bool(grabs.get(label)), "ratios": None}
        if label == POINT_HAND and world is not None and not grabs.get(label):
            index, middle, ring, pinky = finger_ratios(world)
            others = max(middle, ring, pinky)
            keeping = state["since"] is not None
            gap = POINT_KEEP_GAP if keeping else POINT_START_GAP
            # The middle finger often relaxes to ~1.4-1.6 while aiming, so
            # once started only the gap to the index finger has to hold.
            pointing = (
                index > POINT_INDEX_ABOVE
                and index - others > gap
                and (keeping or others < POINT_OTHERS_BELOW)
            )
            _point_debug.update(
                ratios=(index, middle, ring, pinky),
                others=others,
                gap=gap,
                pointing=pointing,
            )

        if pointing:
            if state["since"] is None:
                state["since"] = now
            state["last"] = now
        elif state["last"] is None or now - state["last"] > POINT_GRACE_SECONDS:
            state["since"] = state["last"] = None

    for label, state in _point_state.items():
        if label not in visible:
            state["since"] = state["last"] = None

    return {
        label: state["since"] is not None
        and now - state["since"] >= POINT_HOLD_SECONDS
        for label, state in _point_state.items()
    }

# --------------------------------------------------
# WORLD LANDMARK SMOOTHING (One Euro filter)
# --------------------------------------------------

# Light settings: Unreal smooths position/rotation again, so this only
# removes frame-to-frame jitter in the metric hand shape used for
# orientation and gestures. Higher MIN_CUTOFF = less smoothing.
FILTER_MIN_CUTOFF = 3.0
FILTER_BETA = 10.0
FILTER_D_CUTOFF = 1.0


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self):
        self.prev = None
        self.prev_deriv = None
        self.prev_time = None

    def __call__(self, values, now):
        if self.prev is None:
            self.prev = values
            self.prev_deriv = np.zeros_like(values)
            self.prev_time = now
            return values

        dt = max(now - self.prev_time, 1e-3)
        deriv = (values - self.prev) / dt
        a_d = _alpha(FILTER_D_CUTOFF, dt)
        deriv = a_d * deriv + (1 - a_d) * self.prev_deriv
        cutoff = FILTER_MIN_CUTOFF + FILTER_BETA * np.abs(deriv)
        a = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) / dt)
        filtered = a * values + (1 - a) * self.prev

        self.prev, self.prev_deriv, self.prev_time = filtered, deriv, now
        return filtered


_world_filters = {}


def smooth_world_landmarks(hands, now):
    # Returns hands with filtered world landmarks; resets the filter of a
    # hand that disappeared so it does not drag a stale pose back in.
    out = []
    visible = set()

    for landmarks, label, world in hands:
        visible.add(label)
        if world is None:
            out.append((landmarks, label, world))
            continue

        raw = np.array([[p.x, p.y, p.z] for p in world], dtype=np.float64)
        filt = _world_filters.setdefault(label, OneEuroFilter())(raw, now)
        out.append((
            landmarks,
            label,
            [SimpleNamespace(x=v[0], y=v[1], z=v[2]) for v in filt],
        ))

    for label in list(_world_filters):
        if label not in visible:
            del _world_filters[label]

    return out

# --------------------------------------------------
# HAND DISTANCE (rotation independent depth)
# --------------------------------------------------


def hand_distance(landmarks, world, width, height):
    # Fits the metric 3D hand (world landmarks) to its 2D image points to
    # get the hand's distance from the camera in meters. Unlike palm width
    # in the image, this does not change when the hand rotates. The focal
    # length is a rough guess; Unreal only uses the ratio to the
    # calibration distance, so the absolute scale does not matter.
    if world is None:
        return 0.0

    obj = np.array([[p.x, p.y, p.z] for p in world], dtype=np.float64)
    img = np.array(
        [[p.x * width, p.y * height] for p in landmarks], dtype=np.float64
    )
    camera = np.array(
        [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    ok, _, tvec = cv2.solvePnP(obj, img, camera, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok or tvec[2][0] <= 0:
        return 0.0
    return float(tvec[2][0])

# --------------------------------------------------
# OSC OUTPUT
# --------------------------------------------------


def send_hand_data(hands, grabs, triggers, distances, points):
    # hands: resolved [(landmarks, label, world)].
    # Landmark message layout (existing receivers read only the first 63):
    #   0-62  21 landmarks x (x, y, z), normalized image coordinates
    #   63    grab flag (1 = fist)
    #   64-66 finger direction, Unreal pawn axes (all zero if unknown)
    #   67-69 palm normal, Unreal pawn axes (all zero if unknown)
    #   70    trigger flag (1 = index finger pulled while grabbing)
    #   71    hand distance from the camera in meters (0 if unknown)
    #   72    point flag (1 = pointing held long enough to aim a teleport)
    seen = set()

    for landmarks, label, world in hands:
        seen.add(label)

        flat_landmarks = []
        for point in landmarks:
            flat_landmarks.extend([
                float(point.x),
                float(point.y),
                float(point.z),
            ])
        flat_landmarks.append(1.0 if grabs.get(label) else 0.0)

        orientation = hand_orientation(world, label) if world else None
        if orientation:
            flat_landmarks.extend(float(c) for c in orientation[0] + orientation[1])
        else:
            flat_landmarks.extend([0.0] * 6)
        flat_landmarks.append(1.0 if triggers.get(label) else 0.0)
        flat_landmarks.append(float(distances.get(label, 0.0)))
        flat_landmarks.append(1.0 if points.get(label) else 0.0)

        wrist = landmarks[0]

        osc.send_message(f"/hand/{label}/landmarks", flat_landmarks)
        osc.send_message(
            f"/hand/{label}/wrist",
            [float(wrist.x), float(wrist.y), float(wrist.z)],
        )
        osc.send_message(f"/hand/{label}/valid", 1)

    # Invalidate hands that are not visible so Unreal does not keep a
    # stale pose.
    for label in ("left", "right"):
        if label not in seen:
            osc.send_message(f"/hand/{label}/valid", 0)


# A hand counts as covering the face when at least this many of its 21
# landmarks fall inside the face box shrunk by FACE_BOX_SHRINK on each side.
# Hands held beside the face (the normal calibration pose) put at most one
# point inside; a hand in front of the face puts most of them inside.
FACE_BOX_SHRINK = 0.15
FACE_OCCLUSION_POINTS = 6


def hand_covers_face(face, hands):
    # Face tracking is unreliable while a hand covers the face, so the
    # caller holds the last head pose instead of sending a jumpy one.
    xs = [p.x for p in face]
    ys = [p.y for p in face]
    dx = (max(xs) - min(xs)) * FACE_BOX_SHRINK
    dy = (max(ys) - min(ys)) * FACE_BOX_SHRINK
    left, right = min(xs) + dx, max(xs) - dx
    top, bottom = min(ys) + dy, max(ys) - dy

    for landmarks, _, _ in hands:
        inside = sum(
            left <= p.x <= right and top <= p.y <= bottom for p in landmarks
        )
        if inside >= FACE_OCCLUSION_POINTS:
            return True
    return False


def send_face_data(face_result, hands):
    if not (
        face_result.face_landmarks
        and face_result.facial_transformation_matrixes
    ):
        osc.send_message("/head/valid", 0)
        return

    if hand_covers_face(face_result.face_landmarks[0], hands):
        # Send nothing: Unreal keeps the last good head pose.
        return

    matrix = face_result.facial_transformation_matrixes[0]
    nose = face_result.face_landmarks[0][1]

    osc.send_message("/head/valid", 1)
    osc.send_message(
        "/head/matrix",
        [float(value) for value in matrix.flatten()],
    )
    osc.send_message(
        "/head/nose",
        [float(nose.x), float(nose.y), float(nose.z)],
    )

# --------------------------------------------------
# PREVIEW
# --------------------------------------------------


def draw_hand(frame, landmarks, label, grabbing, firing, pointing):
    height, width = frame.shape[:2]
    color = LABEL_COLORS.get(label, (255, 255, 255))
    points = [
        (int(point.x * width), int(point.y * height))
        for point in landmarks
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], color, 4 if grabbing else 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 120, 255), -1)

    x, y = points[0]
    cv2.putText(
        frame,
        label.capitalize()
        + (" (grab)" if grabbing else "")
        + (" FIRE" if firing else "")
        + (" (aim)" if pointing else ""),
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def draw_status(frame, face_detected, hand_count, fps):
    lines = [
        (
            f"Face: {'YES' if face_detected else 'NO'}",
            (0, 255, 0) if face_detected else (0, 0, 255),
        ),
        (f"Hands: {hand_count}", (255, 255, 255)),
        (f"FPS: {fps:.0f}", (255, 255, 255)),
        (f"OSC: {OSC_IP}:{OSC_PORT}", (255, 255, 0)),
    ]

    for row, (text, color) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (15, 30 + row * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def _draw_lines_bottom_left(frame, lines):
    # lines: [(text, bgr)], drawn upward from the bottom-left corner on a
    # dark background so they stay readable over any scene.
    height = frame.shape[0]
    row_height = 24
    top = height - 10 - row_height * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, top - 5), (330, height - 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for row, (text, color) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (12, top + 18 + row * row_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
        )


def draw_point_debug(frame, now):
    # Shows every condition of the point (teleport aim) gesture for
    # POINT_HAND, green when met, red when not, so a failed aim can be
    # traced to the exact threshold. Toggle with D in the preview.
    ok, bad, info = (0, 255, 0), (0, 0, 255), (255, 255, 255)
    lines = [(f"POINT DEBUG ({POINT_HAND} hand, D = hide)", (255, 255, 0))]
    debug = _point_debug

    if debug is None:
        lines.append(("hand not visible", bad))
    elif debug["grabbing"]:
        lines.append(("grabbing -> point blocked", bad))
    elif debug["ratios"] is None:
        lines.append(("no 3D landmarks", bad))
    else:
        index, middle, ring, pinky = debug["ratios"]
        others, gap = debug["others"], debug["gap"]
        phase = "keep" if gap == POINT_KEEP_GAP else "start"
        lines += [
            (f"I {index:.2f}  M {middle:.2f}  R {ring:.2f}  P {pinky:.2f}", info),
            (
                f"index  {index:.2f} > {POINT_INDEX_ABOVE:.2f}",
                ok if index > POINT_INDEX_ABOVE else bad,
            ),
            (
                f"gap    {index - others:.2f} > {gap:.2f} ({phase})",
                ok if index - others > gap else bad,
            ),
            (
                f"others {others:.2f} < {POINT_OTHERS_BELOW:.2f}"
                + (" (start only)" if phase == "keep" else ""),
                ok if phase == "keep" or others < POINT_OTHERS_BELOW else bad,
            ),
        ]

    state = _point_state[POINT_HAND]
    if state["since"] is None:
        lines.append(("state: idle", info))
    else:
        held = now - state["since"]
        if held >= POINT_HOLD_SECONDS:
            text, color = "state: AIMING", ok
        else:
            text, color = f"state: holding {held:.2f}/{POINT_HOLD_SECONDS:.2f} s", (0, 200, 255)
        if debug is None or not debug.get("pointing"):
            text += f" (grace {now - state['last']:.2f} s)"
        lines.append((text, color))

    _draw_lines_bottom_left(frame, lines)

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Webcam head and hand tracking that streams to Unreal over OSC."
    )
    parser.add_argument(
        "--osc-ip",
        default=OSC_IP,
        help="IP of the machine running Unreal (default: %(default)s).",
    )
    parser.add_argument("--osc-port", type=int, default=OSC_PORT)
    parser.add_argument(
        "--camera", type=int, default=CAMERA_INDEX, help="Webcam index."
    )
    parser.add_argument(
        "--point-debug",
        action="store_true",
        help="Start with the point gesture debug panel shown (toggle: D).",
    )
    return parser.parse_args()


def open_camera(index):
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(index)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check camera permissions."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    # Keep only the newest frame so tracking does not lag behind.
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return camera


def main():
    global osc, OSC_IP, OSC_PORT
    args = parse_args()
    OSC_IP, OSC_PORT = args.osc_ip, args.osc_port
    osc = SimpleUDPClient(OSC_IP, OSC_PORT)

    camera = open_camera(args.camera)
    show_point_debug = args.point_debug
    last_timestamp_ms = -1
    fps = 0.0
    last_frame_time = time.perf_counter()

    try:
        with (
            mp_vision.HandLandmarker.create_from_options(
                hand_options
            ) as hand_tracker,
            mp_vision.FaceLandmarker.create_from_options(
                face_options
            ) as face_tracker,
            # MediaPipe releases the GIL during inference, so running the
            # hand and face models in parallel roughly halves frame time.
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read a frame from the webcam.")
                    break

                # Mirror the frame so the preview behaves like a mirror.
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                # VIDEO mode requires strictly increasing timestamps.
                timestamp_ms = max(
                    time.monotonic_ns() // 1_000_000,
                    last_timestamp_ms + 1,
                )
                last_timestamp_ms = timestamp_ms

                hand_future = executor.submit(
                    hand_tracker.detect_for_video, mp_image, timestamp_ms
                )
                face_future = executor.submit(
                    face_tracker.detect_for_video, mp_image, timestamp_ms
                )
                hand_result = hand_future.result()
                face_result = face_future.result()

                raw_hands = resolve_hand_labels(
                    get_hand_candidates(hand_result)
                )
                # Distance uses the raw 3D shape so it stays consistent
                # with the 2D points of the same frame.
                height, width = frame.shape[:2]
                distances = {
                    label: hand_distance(landmarks, world, width, height)
                    for landmarks, label, world in raw_hands
                }
                hands = smooth_world_landmarks(raw_hands, time.perf_counter())
                grabs = update_grab_states(hands)
                triggers = update_trigger_states(hands, grabs)
                points = update_point_states(hands, grabs, time.perf_counter())
                send_hand_data(hands, grabs, triggers, distances, points)
                send_face_data(face_result, hands)

                now = time.perf_counter()
                frame_time = now - last_frame_time
                last_frame_time = now
                if frame_time > 0:
                    fps = fps * 0.9 + (1.0 / frame_time) * 0.1

                for landmarks, label, _ in hands:
                    draw_hand(
                        frame,
                        landmarks,
                        label,
                        grabs[label],
                        triggers[label],
                        points[label],
                    )

                draw_status(
                    frame,
                    bool(face_result.face_landmarks),
                    len(hands),
                    fps,
                )
                if show_point_debug:
                    draw_point_debug(frame, now)

                cv2.imshow("Combined Head and Hand Tracking", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("d"):
                    show_point_debug = not show_point_debug

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
