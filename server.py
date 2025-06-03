import sys
import cv2
import time
import math
import random
import uvicorn
import platform
import threading
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, Response, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse

#############################################
## FOR THE PURPOSES OF RATIONAL AMUSEMENTS ##
#############################################

class DiscObject:
    def __init__(self, image, radius, angle, speed):
        self.image = image  # Now expects RGBA image
        self.radius = radius  # Distance from center
        self.angle = angle    # Current angle in radians
        self.speed = speed    # Rotation speed in radians per frame
        self.size = image.shape[0]  # Assuming square image

    def update(self):
        self.angle += self.speed
        if self.angle >= 2 * math.pi:
            self.angle -= 2 * math.pi

    def get_position(self, center_x, center_y):
        x = center_x + int(self.radius * math.cos(self.angle))
        y = center_y + int(self.radius * math.sin(self.angle))
        return x, y

    def draw(self, canvas, center_x, center_y, color_index):
        x, y = self.get_position(center_x, center_y)
        rotation_angle = math.degrees(self.angle)
        rotation_matrix = cv2.getRotationMatrix2D((self.size//2, self.size//2), rotation_angle, 1.0)
        rotated_image = cv2.warpAffine(self.image, rotation_matrix, (self.size, self.size), 
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                      borderValue=(0, 0, 0, 0))
        x1 = x - self.size // 2
        y1 = y - self.size // 2
        x2 = x1 + self.size
        y2 = y1 + self.size
        if (x1 >= 0 and x2 < canvas.shape[1] and 
            y1 >= 0 and y2 < canvas.shape[0]):
            roi = canvas[y1:y2, x1:x2]
            alpha_mask = rotated_image[:, :, 3] / 255.0
            kernel = np.ones((5, 5), np.uint8)
            expanded_mask = cv2.dilate(alpha_mask, kernel, iterations=1)
            expanded_mask = np.expand_dims(expanded_mask, axis=-1)
            border_color = object_colors[color_index]
            roi = roi * (1 - expanded_mask) + border_color * expanded_mask
            alpha_mask = np.expand_dims(alpha_mask, axis=-1)
            roi = roi * (1 - alpha_mask) + rotated_image[:, :, :3] * alpha_mask
            canvas[y1:y2, x1:x2] = roi.astype(np.uint8)

# --- Globals for frame sharing ---
screen_one = np.zeros((1920, 1080, 3), dtype=np.uint8)  # Vertical orientation
screen_two = np.zeros((480, 480, 3), dtype=np.uint8)    # Keep as RGB for display
frame_lock = threading.Lock()

# --- FastAPI app setup ---
app = FastAPI()

# --- HTML templates ---
SCREEN_STREAM_HTML = '''
<!DOCTYPE html>
<html>
<head>
  <title>{title}</title>
  <style>
    body {{ background: black; margin: 0; }}
    img {{ display: block; margin: auto; max-width: 100vw; max-height: 100vh; }}
  </style>
</head>
<body>
  <img id="live" src="/{endpoint}_stream" />
  <script>
    // Fullscreen toggle on 'f' key
    document.addEventListener('keydown', (e) => {{
      if (e.key === 'f' || e.key === 'F') {{
        const img = document.getElementById('live');
        if (!document.fullscreenElement) {{
          img.requestFullscreen();
        }} else {{
          document.exitFullscreen();
        }}
      }}
    }});
  </script>
</body>
</html>
'''

@app.get("/frame")
async def get_frame(request: Request):
    print(f"{time.time()} /frame request from {request.client.host}")
    # Simulate frame timestamp (could be improved with actual frame timing)
    frame_timestamp = time.time()
    with frame_lock:
        screen_two_small = cv2.resize(screen_two, (240, 240))
        _, img = cv2.imencode('.jpg', screen_two_small)
        frame_data = img.tobytes()
    if frame_data is None:
        raise HTTPException(status_code=503, detail="No frames available")
    headers = {
        "X-Frame-Timestamp": str(frame_timestamp),
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
    }
    return Response(
        content=frame_data,
        media_type="image/jpeg",
        headers=headers
    ) 

# --- Endpoints for HTML pages ---
@app.get("/screen1", response_class=HTMLResponse)
def screen1_page():
    return SCREEN_STREAM_HTML.format(title="Screen 1", endpoint="screen1")

@app.get("/screen2", response_class=HTMLResponse)
def screen2_page():
    return SCREEN_STREAM_HTML.format(title="Screen 2", endpoint="screen2")

# --- Endpoints for image streaming ---
@app.get("/screen1_img")
def get_screen1_img():
    with frame_lock:
        _, img = cv2.imencode('.jpg', screen_one)
    return Response(content=img.tobytes(), media_type="image/jpeg")

@app.get("/screen2_img")
def get_screen2_img():
    with frame_lock:
        _, img = cv2.imencode('.jpg', screen_two)
    return Response(content=img.tobytes(), media_type="image/jpeg")

def mjpeg_generator(get_frame_func):
    while True:
        frame = get_frame_func()
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.get("/screen1_stream")
def screen1_stream():
    def get_frame():
        with frame_lock:
            return screen_one.copy()
    return StreamingResponse(mjpeg_generator(get_frame), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/screen2_stream")
def screen2_stream():
    def get_frame():
        with frame_lock:
            return screen_two.copy()
    return StreamingResponse(mjpeg_generator(get_frame), media_type='multipart/x-mixed-replace; boundary=frame')

# --- Image processing logic from main.py ---
def image_processing_loop():
    global screen_one, screen_two
    # Camera and model setup
    if platform.system() == "Linux":
        environment_camera = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Environment camera
        internal_camera = cv2.VideoCapture(2, cv2.CAP_V4L2)    # Internal camera
    else:
        environment_camera = cv2.VideoCapture(0)
        internal_camera = cv2.VideoCapture(1)

    # Check if both cameras are available
    if not environment_camera.isOpened():
        print("Error: Environment camera not available")
        sys.exit(1)
    if not internal_camera.isOpened():
        print("Error: Internal camera not available")
        sys.exit(1)

    error_count = 0
    error_threshold = 10
    model = YOLO("yolov8n-seg.pt")
    # Screen one state
    main_screen_mode = 1
    main_screen_interval = 100
    main_screen_counter = 0
    # Screen two state
    counter = 0
    interval = 10
    center_x = 240
    center_y = 240
    disc_radius = 200
    MAX_OBJECTS = 11
    disc_objects = []
    rotation_speed = 0.02
    global object_colors
    object_colors = [np.array([np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]) for _ in range(MAX_OBJECTS)]
    while True:
        ret_environment, frame_environment = environment_camera.read()
        if main_screen_mode == 1:
            ret_internal, frame_internal = internal_camera.read()
            if ret_internal:
                frame_internal = cv2.rotate(frame_internal, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame_internal = cv2.resize(frame_internal, (1080, 1920))
        if ret_environment:
            yolo_res = model(frame_environment, imgsz=320, verbose=False, classes=[i for i in range(80) if i != 0])
            annotated_frame = yolo_res[0].plot()
            objects = yolo_res[0].boxes.xyxy.cpu().numpy()
            if counter % interval == 0 and len(objects) > 0:
                counter = 0
                random_object = np.random.randint(0, len(objects))
                bounding_box = objects[random_object]
                x1, y1, x2, y2 = map(int, bounding_box)
                if hasattr(yolo_res[0], 'masks') and yolo_res[0].masks is not None:
                    mask = yolo_res[0].masks.data[random_object].cpu().numpy()  # shape: (mask_h, mask_w)
                    # Get the shape of the original frame
                    frame_h, frame_w = frame_environment.shape[:2]

                    # Scale mask to original frame size
                    mask_resized = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

                    # Crop mask and image to bounding box
                    mask_cropped = mask_resized[y1:y2, x1:x2]
                    object_image = frame_environment[y1:y2, x1:x2].copy()

                    # Create RGBA image
                    rgba_image = np.zeros((y2-y1, x2-x1, 4), dtype=np.uint8)
                    rgba_image[:, :, :3] = object_image
                    rgba_image[:, :, 3] = (mask_cropped * 255).astype(np.uint8)

                    # ENLARGE the segment (e.g., to 120x120)
                    enlarged_size = random.randint(80, 200)
                    rgba_image = cv2.resize(rgba_image, (enlarged_size, enlarged_size), interpolation=cv2.INTER_LINEAR)

                    radius = np.random.randint(50, disc_radius - 50)
                    angle = np.random.uniform(0, 2 * math.pi)
                    speed = rotation_speed * (1 + np.random.uniform(-0.2, 0.2))
                    disc_objects.append(DiscObject(rgba_image, radius, angle, speed))
                if len(disc_objects) == MAX_OBJECTS:
                    disc_objects.pop(0)
            counter += 1
            # Update and draw all objects
            with frame_lock:
                screen_two.fill(0)
                cv2.circle(screen_two, (center_x, center_y), disc_radius, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                for i, obj in enumerate(disc_objects):
                    obj.update()
                    obj.draw(screen_two, center_x, center_y, i)
                if main_screen_mode == 1 and 'ret_internal' in locals() and ret_internal:
                    screen_one[...] = frame_internal
                else:
                    annotated_frame = cv2.resize(annotated_frame, (1080, 960))
                    screen_one[0:960, 0:1080] = annotated_frame
                    screen_one[960:1920, 0:1080] = 0
                    resized_screen_two = cv2.resize(screen_two, (960, 960))
                    x_offset = 960 + (960 - 960) // 2
                    y_offset = (1080 - 960) // 2
                    try:
                        screen_one[x_offset:x_offset+960, y_offset:y_offset+960] = resized_screen_two
                    except ValueError as e:
                        print(f"Error during paste: {e}")
                        print(f"Attempted to paste {resized_screen_two.shape} into slice of shape {screen_one[x_offset:x_offset+420, y_offset:y_offset+420].shape}")
                    # Increase saturation and contrast of screen_two
                    screen_two_hsv = cv2.cvtColor(screen_two, cv2.COLOR_BGR2HSV)
                    screen_two_hsv[:, :, 1] = cv2.multiply(screen_two_hsv[:, :, 1], 1.5)  # Increase saturation
                    screen_two_enhanced = cv2.cvtColor(screen_two_hsv, cv2.COLOR_HSV2BGR)
                    screen_two_enhanced = cv2.convertScaleAbs(screen_two_enhanced, alpha=1.3, beta=10)  # Increase contrast
                    screen_two = screen_two_enhanced
                if main_screen_counter % main_screen_interval == 0:
                    main_screen_counter = 0
                    main_screen_mode = 1 - main_screen_mode
                main_screen_counter += 1
        else:
            print("Error reading frames")
            error_count += 1
            if error_count > error_threshold:
                print("Error threshold reached, exiting image processing loop.")
                break
        time.sleep(1/30)  # ~30 FPS
    environment_camera.release()
    internal_camera.release()
    print("Image processing loop ended")

# --- Startup event to launch processing thread ---
@app.on_event("startup")
def start_bg_thread():
    thread = threading.Thread(target=image_processing_loop, daemon=True)
    thread.start()

# --- Main entry point ---
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

