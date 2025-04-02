import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(A, B, C):
    """
    Computes the angle between vectors AB and BC using only the (x, y) coordinates.
    We then return the bending angle as: bending = 180° - (raw angle)
    so that a straight joint (raw angle 180°) yields 0° bending.
    """
    # Use only the x, y coordinates for stability.
    A, B, C = np.array(A[:2]), np.array(B[:2]), np.array(C[:2])

    AB = A - B
    BC = C - B

    dot_product = np.dot(AB, BC)
    magnitude_AB = np.linalg.norm(AB)
    magnitude_BC = np.linalg.norm(BC)

    # Compute the raw angle (in degrees) between the vectors:
    raw_angle = np.degrees(np.arccos(np.clip(dot_product / (magnitude_AB * magnitude_BC), -1.0, 1.0)))

    # Return bending angle: 0° when fully extended (raw_angle = 180°), higher when flexed.
    return max(0, min(180, 180 - raw_angle))


def calculate_mcp_angle(wrist, mcp, pip):
    """
    Calculates the MCP bending angle using the wrist, MCP, and PIP landmarks.
    When the finger is straight, the computed raw angle should be 180° so that the bending is 0°.
    """
    return calculate_angle(wrist, mcp, pip)


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

            # Extract landmark coordinates (each is a tuple: (x, y, z))
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

            wrist = landmarks[0]  # Wrist (base reference for MCP calculation)
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

            for finger, (mcp_idx, pip_idx, dip_idx, tip_idx) in fingers.items():
                if landmarks:
                    # MCP angle: using wrist, MCP, and PIP.
                    mcp_angle = calculate_mcp_angle(wrist, landmarks[mcp_idx], landmarks[pip_idx])

                    # PIP angle: use the order (DIP, PIP, MCP) so that when extended, raw angle = 180° -> bending = 0°.
                    pip_angle = calculate_angle(landmarks[dip_idx], landmarks[pip_idx], landmarks[mcp_idx])

                    # DIP angle: use the order (Tip, DIP, PIP) similarly.
                    dip_angle = calculate_angle(landmarks[tip_idx], landmarks[dip_idx], landmarks[pip_idx])

                    angles_info.append((finger, mcp_angle, pip_angle, dip_angle))

    # Draw a sidebar for angles on the right side of the frame
    sidebar_x = frame.shape[1] - 350
    cv2.rectangle(frame, (sidebar_x, 0), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)

    y_offset = 50
    font_scale, thickness = 1.5, 3

    for finger, mcp_angle, pip_angle, dip_angle in angles_info:
        text_color = colors[finger]
        cv2.putText(frame, f"{finger}:", (sidebar_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                    thickness, cv2.LINE_AA)
        y_offset += 50
        cv2.putText(frame, f"MCP: {int(mcp_angle)}°", (sidebar_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    text_color, thickness, cv2.LINE_AA)
        y_offset += 50
        cv2.putText(frame, f"PIP: {int(pip_angle)}°", (sidebar_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    text_color, thickness, cv2.LINE_AA)
        y_offset += 50
        cv2.putText(frame, f"DIP: {int(dip_angle)}°", (sidebar_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    text_color, thickness, cv2.LINE_AA)
        y_offset += 60

    cv2.imshow('Hand Tracking with Corrected Joint References', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()