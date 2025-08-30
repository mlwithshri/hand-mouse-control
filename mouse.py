import cv2
import mediapipe as mp
import pyautogui
import math

# Get screen size
screen_width, screen_height = pyautogui.size()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

click_threshold = 30  # pixel distance for click gesture

clicked = False  # to prevent repeated clicks

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    if not success:
        break

    # Flip image for natural interaction
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    # Get frame dimensions
    h, w, _ = image.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Index finger tip landmark (id 8)
            index_finger_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            x = int(index_finger_tip.x * w)
            y = int(index_finger_tip.y * h)

            # Map to screen coordinates
            screen_x = int(index_finger_tip.x * screen_width)
            screen_y = int(index_finger_tip.y * screen_height)

            # Move the mouse
            pyautogui.moveTo(screen_x, screen_y)
            # Convert to image coordinates for drawing and distance
            
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # calculate distance between thumb and finger
            distance=math.hypot(tx-x,ty-y)
            # condition for click
            if distance <click_threshold:
                if  not clicked:
                    pyautogui.click()
                    clicked=True
            else:
                clicked = False
                    

            # Optional: visualize
            cv2.circle(image, (x, y), 10, (255, 0, 255), -1)
            cv2.circle(image, (tx, ty), 10, (0, 255, 0), -1)

    cv2.imshow("Hand Cursor Control", image)
    if cv2.waitKey(1) & 0xFF == 27: # esc key
        break

cap.release()
cv2.destroyAllWindows()
