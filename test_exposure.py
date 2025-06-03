import cv2
import time

def test_brightness(device, brightness_value):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Failed to open {device}")
        return
    
    # Set brightness
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness_value)
    actual_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    print(f"Attempted to set brightness to {brightness_value}, actual value: {actual_brightness}")
    
    # Read and display frames for a few seconds
    start_time = time.time()
    while time.time() - start_time < 3:  # Show for 3 seconds
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f'Brightness: {brightness_value}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

# Test different brightness values
for brightness in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    test_brightness(2, brightness)