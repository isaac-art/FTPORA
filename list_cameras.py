import cv2
import time
import subprocess
import re

def get_camera_info():
    # Get v4l2 device information
    result = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True)
    devices = result.stdout
    
    # Parse the output to group devices by camera
    camera_groups = {}
    current_camera = None
    
    for line in devices.split('\n'):
        if not line.strip():
            continue
        if not line.startswith('\t'):
            current_camera = line.strip().rstrip(':')
            camera_groups[current_camera] = []
        elif current_camera and line.strip().startswith('/dev/video'):
            device = line.strip()
            camera_groups[current_camera].append(device)
    
    return camera_groups

def try_camera_with_backend(device_path, backend):
    # Extract device number from path (e.g., /dev/video0 -> 0)
    device_index = int(device_path.split('video')[-1])
    
    cap = cv2.VideoCapture(device_index, backend)
    if not cap.isOpened():
        return None
    
    # Give the camera time to initialize
    time.sleep(0.1)
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    cap.release()
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'backend': backend
    }

def list_available_cameras():
    # List of backends to try
    backends = [
        cv2.CAP_ANY,  # Auto-detect
        cv2.CAP_V4L2,  # Video4Linux2
        cv2.CAP_V4L,   # Video4Linux
        cv2.CAP_FFMPEG # FFMPEG
    ]
    
    backend_names = {
        cv2.CAP_ANY: "Auto-detect",
        cv2.CAP_V4L2: "Video4Linux2",
        cv2.CAP_V4L: "Video4Linux",
        cv2.CAP_FFMPEG: "FFMPEG"
    }
    
    print("Scanning for cameras...")
    print("=" * 50)
    
    # Get camera information from v4l2
    camera_groups = get_camera_info()
    
    for camera_name, devices in camera_groups.items():
        print(f"\nCamera: {camera_name}")
        print("=" * 50)
        
        for device in devices:
            print(f"\nTrying device: {device}")
            print("-" * 30)
            
            for backend in backends:
                print(f"Trying backend: {backend_names[backend]}")
                result = try_camera_with_backend(device, backend)
                
                if result:
                    print(f"✓ Camera found with {backend_names[backend]}:")
                    print(f"  Resolution: {result['width']}x{result['height']}")
                    print(f"  FPS: {result['fps']}")
                else:
                    print(f"✗ No camera found with {backend_names[backend]}")
            
            print("-" * 30)

if __name__ == "__main__":
    list_available_cameras() 