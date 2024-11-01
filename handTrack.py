import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture from the default webcam
cap = cv2.VideoCapture(0)

# Set up MediaPipe Hands
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        # and convert the color space from BGR to RGB
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        results = hands.process(image_rgb)

        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections on the original BGR image
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

                # Get landmark positions
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    h, w, _ = image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.putText(image, f'{idx}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show the processed video frame
        cv2.imshow('Hand Tracker', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
