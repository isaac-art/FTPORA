import cv2
import time

def test_exposure(device, exposure_value):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Failed to open {device}")
        return
    
    # Set exposure
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
    actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    print(f"Attempted to set exposure to {exposure_value}, actual value: {actual_exposure}")
    
    # Read and display frames for a few seconds
    start_time = time.time()
    while time.time() - start_time < 3:  # Show for 5 seconds
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f'Exposure: {exposure_value}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

# Test different exposure values
for exposure in [-1, -4, -7, -10, -13]:
    test_exposure(2, exposure)