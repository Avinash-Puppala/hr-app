import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(A, B, C):
    """
    Computes the angle between vectors AB and BC.
    - 0° when fingers are straight
    - ~85-100° for MCP when fully flexed
    - ~100-120° for PIP when fully flexed
    - ~70-90° for DIP when fully flexed
    """
    A, B, C = np.array(A[:2]), np.array(B[:2]), np.array(C[:2])  # Use only (x, y)

    AB = A - B
    BC = C - B

    dot_product = np.dot(AB, BC)
    magnitude_AB = np.linalg.norm(AB)
    magnitude_BC = np.linalg.norm(BC)

    angle = np.degrees(np.arccos(np.clip(dot_product / (magnitude_AB * magnitude_BC), -1.0, 1.0)))

    return max(0, min(180, 180 - angle))  # Ensure angle remains in range

def calculate_mcp_angle(wrist, mcp, pip):
    """
    ✅ Fix MCP Calculation:
    - Uses **Wrist-MCP-PIP** (no adjacent MCP)
    - **No projection onto palm plane**
    - Ensures max flexion ~85-100°
    """
    wrist, mcp, pip = map(lambda p: np.array(p[:2]), [wrist, mcp, pip])

    v1 = wrist - mcp  # Wrist → MCP vector
    v2 = pip - mcp    # MCP → PIP vector

    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    angle = np.degrees(np.arccos(np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0)))

    return max(0, min(180, angle))  # Ensure MCP stays within valid range

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # High-resolution for accuracy
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("⚠️ No frame received. Retrying...")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    angles_info = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

            wrist = landmarks[0]  # ✅ Wrist reference for MCP
            fingers = {
                "Thumb": (1, 2, 3, 4),
                "Index": (5, 6, 7, 8),
                "Middle": (9, 10, 11, 12),
                "Ring": (13, 14, 15, 16),
                "Pinky": (17, 18, 19, 20),
            }

            colors = {
                "Thumb": (0, 255, 0),
                "Index": (255, 255, 0),
                "Middle": (0, 255, 255),
                "Ring": (255, 0, 255),
                "Pinky": (255, 140, 0),
            }

            for finger, (mcp, pip, dip, tip) in fingers.items():
                if landmarks:
                    # ✅ **New MCP Calculation (Only Wrist-MCP-PIP)**
                    mcp_angle = calculate_mcp_angle(landmarks[0], landmarks[mcp], landmarks[pip])

                    # ✅ **PIP: MCP-PIP-DIP**
                    pip_angle = calculate_angle(landmarks[mcp], landmarks[pip], landmarks[dip])

                    # ✅ **DIP: PIP-DIP-Tip**
                    dip_angle = calculate_angle(landmarks[pip], landmarks[dip], landmarks[tip])

                    angles_info.append((finger, mcp_angle, pip_angle, dip_angle))

    # Draw Sidebar for Angles
    sidebar_x = frame.shape[1] - 350
    cv2.rectangle(frame, (sidebar_x, 0), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)  # Dark gray for better contrast

    y_offset = 50
    font_scale, thickness = 1.5, 3  # Larger, clearer text

    for finger, mcp_angle, pip_angle, dip_angle in angles_info:
        text_color = colors[finger]

        cv2.putText(frame, f"{finger}:", (sidebar_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
        y_offset += 50

        cv2.putText(frame, f"MCP: {int(mcp_angle)}°", (sidebar_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
        y_offset += 50

        cv2.putText(frame, f"PIP: {int(pip_angle)}°", (sidebar_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
        y_offset += 50

        cv2.putText(frame, f"DIP: {int(dip_angle)}°", (sidebar_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
        y_offset += 60

    cv2.imshow('Hand Tracking with Corrected MCP', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
