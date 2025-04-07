import cv2
import mediapipe as mp
import numpy as np

# MediaPipe hand detection settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start Camera
cap = cv2.VideoCapture(0)

# Upload PNG file
png_image = cv2.imread('himym.jpg', cv2.IMREAD_UNCHANGED)

if png_image is None:
    print("PNG file could not be loaded. Please make sure you are in the correct path.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

if png_image.shape[2] == 4:
    print("PNG transparency channel available.")
else:
    print("PNG file has no transparency channel.")

png_height, png_width = png_image.shape[:2]

prev_distance = None
zoom_factor = 1.0
zoom_factor_limit = 2.5  # Upper limit for zoom

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Hand detection
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Take the positions of the index fingers
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Normalize locations for better visibility
            thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            # Calculate the distance of the finger
            current_distance = np.linalg.norm(np.array([thumb_x, thumb_y]) - np.array([index_x, index_y]))

            # Resize PNG to that distance
            new_width = int(current_distance)
            new_height = int(png_height * (new_width / png_width))
            resized_png = cv2.resize(png_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # Place the middle of the PNG between the fingers
            h, w = frame.shape[:2]
            x_offset = int((thumb_x + index_x - new_width) / 2)
            y_offset = int((thumb_y + index_y - new_height) / 2)

            # Overlay using the PNG's alpha channel (transparency)
            if png_image.shape[2] == 4:  # Alfa kanalı varsa
                for c in range(0, 3):  # R, G, B kanalları
                    frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width, c] = \
                        frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width, c] * (1 - resized_png[:, :, 3] / 255.0) + \
                        resized_png[:, :, c] * (resized_png[:, :, 3] / 255.0)
            else:
                frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_png

            # Draw
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), -1)
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Show on screen
    cv2.imshow("Zoom Gesture", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Free up resources
cap.release()
cv2.destroyAllWindows()
