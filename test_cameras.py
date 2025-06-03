import cv2
import time
import subprocess
import numpy as np

def get_camera_devices():
    # Get v4l2 device information
    result = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True)
    devices = result.stdout
    
    # Parse the output to get video devices
    video_devices = []
    for line in devices.split('\n'):
        if line.strip().startswith('/dev/video'):
            video_devices.append(line.strip())
    
    return video_devices

def test_camera(device_path, frames=30):
    # Extract device number from path (e.g., /dev/video0 -> 0)
    device_index = int(device_path.split('video')[-1])
    
    # Try to open the camera
    cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Failed to open {device_path}")
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\nTesting {device_path}:")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    
    # Create window with device name
    window_name = f"Camera: {device_path}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    frame_count = 0
    while frame_count < frames:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame from {device_path}")
            break
        
        # Add text overlay with device info
        text = f"Device: {device_path}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count + 1}/{frames}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow(window_name, frame)
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        time.sleep(1/fps)  # Try to maintain proper frame rate
    
    # Clean up
    cap.release()
    cv2.destroyWindow(window_name)
    return True

def main():
    print("Scanning for cameras...")
    devices = get_camera_devices()
    
    if not devices:
        print("No cameras found!")
        return
    
    print(f"\nFound {len(devices)} video devices:")
    for device in devices:
        print(f"- {device}")
    
    print("\nTesting each camera for 30 frames...")
    print("Press 'q' to skip to next camera")
    
    for device in devices:
        if test_camera(device):
            print(f"Successfully tested {device}")
        else:
            print(f"Failed to test {device}")
        
        # Small delay between cameras
        time.sleep(1)
    
    print("\nCamera testing complete!")

if __name__ == "__main__":
    main() 