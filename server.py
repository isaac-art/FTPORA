import sys
import cv2
import time
import math
import random
import uvicorn
import platform
import threading
import subprocess
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, Response, Request, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse


from settings import settings
#############################################
## FOR THE PURPOSES OF RATIONAL AMUSEMENTS ##
#############################################

class DiscObject:
    def __init__(self, image, radius, angle, speed, fade_in_frames):
        self.image = image  # Now expects RGBA image
        self.radius = radius  # Distance from center
        self.angle = angle    # Current angle in radians
        self.speed = speed    # Rotation speed in radians per frame
        self.size = image.shape[0]  # Assuming square image
        self.opacity = 0.0  # Start fully transparent
        self.fade_in_frames = fade_in_frames
        self._fade_in_step = 1.0 / self.fade_in_frames
        self.border_color = np.array([np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)])

    def update(self):
        self.angle += self.speed
        if self.angle >= 2 * math.pi:
            self.angle -= 2 * math.pi
        # Fade in
        if self.opacity < 1.0:
            self.opacity = min(1.0, self.opacity + self._fade_in_step)

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
            # Apply fade-in opacity
            alpha_mask = alpha_mask * self.opacity
            kernel = np.ones((9, 9), np.uint8)
            expanded_mask = cv2.dilate(alpha_mask, kernel, iterations=2)
            expanded_mask = np.expand_dims(expanded_mask, axis=-1)
            roi = roi * (1 - expanded_mask) + self.border_color * expanded_mask
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

@app.get("/settings")
def get_settings_page():
    return HTMLResponse(content=open("settings.html").read())

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
    global screen_one, screen_two, frame_lock, disc_objects

    # Camera and model setup
    if platform.system() == "Linux":
        environment_camera = cv2.VideoCapture(2, cv2.CAP_V4L2)
        internal_camera = cv2.VideoCapture(1, cv2.CAP_V4L2)
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

    overlay = cv2.imread("stencil_two.png", cv2.IMREAD_UNCHANGED)
    if overlay is not None and overlay.shape[2] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
    alpha_overlay = overlay[:, :, 3] / 255.0  # Normalize alpha to [0,1]
    alpha_image = 1.0 - alpha_overlay

    overlay_tall = cv2.imread("stencil_tall.png", cv2.IMREAD_UNCHANGED)
    if overlay_tall is not None and overlay_tall.shape[2] == 3:
        overlay_tall = cv2.cvtColor(overlay_tall, cv2.COLOR_BGR2BGRA)
    alpha_overlay_tall = overlay_tall[:, :, 3] / 255.0  # Normalize alpha to [0,1]
    alpha_image_tall = 1.0 - alpha_overlay_tall

    sample_videos_loop = cv2.VideoCapture(settings['sample_video'])
    error_count = 0
    model = YOLO("yolov8n-seg.pt")
    # Screen one state
    main_screen_mode = 1
    main_screen_counter = 0
    # Screen two state
    draw_circle = True
    counter = 0
    center_x = 240
    center_y = 240
    disc_objects = []
    
    while True:
        # Read settings live for hot-reload
        disc_radius = settings['disc_radius']
        MAX_OBJECTS = settings['max_objects']
        object_size_min, object_size_max = settings['object_size_range']
        rotation_speed = settings['rotation_speed']
        disc_fade_in_frames = settings['fade_in_frames']
        toggle_classes = settings['toggle_classes']
        main_screen_interval = settings['main_screen_interval']
        interval = settings['object_add_interval']
        debug_mode = settings['debug_mode']
        error_threshold = settings['error_threshold']

        ret_environment, frame_environment = environment_camera.read()
        ret_sample_videos, frame_sample_videos = sample_videos_loop.read()
        if not ret_sample_videos:
            # loop
            sample_videos_loop.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_sample_videos, frame_sample_videos = sample_videos_loop.read()
        if main_screen_mode == 1:
            # show internal camera view
            ret_internal, frame_internal = internal_camera.read()
            if ret_internal:
                frame_internal = cv2.rotate(frame_internal, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame_internal = cv2.resize(frame_internal, (1080, 1920))
        elif main_screen_mode == 2:
            # show sample video
            frame_internal = cv2.resize(frame_sample_videos, (1080, 1920))
            ret_internal = True
        if ret_environment:
            if toggle_classes:
                yolo_res = model(frame_environment, imgsz=320, verbose=False, classes=[i for i in range(80) if i != 0])
            else:
                yolo_res = model(frame_environment, imgsz=320, verbose=False)
            annotated_frame = yolo_res[0].plot()
            objects = yolo_res[0].boxes.xyxy.cpu().numpy()
            if counter % interval == 0 and len(objects) > 0:
                counter = 0
                random_object = np.random.randint(0, len(objects))
                bounding_box = objects[random_object]
                x1, y1, x2, y2 = map(int, bounding_box)
                if hasattr(yolo_res[0], 'masks') and yolo_res[0].masks is not None:
                    mask = yolo_res[0].masks.data[random_object].cpu().numpy()
                    frame_h, frame_w = frame_environment.shape[:2]
                    mask_resized = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
                    mask_cropped = mask_resized[y1:y2, x1:x2]
                    object_image = frame_environment[y1:y2, x1:x2].copy()
                    rgba_image = np.zeros((y2-y1, x2-x1, 4), dtype=np.uint8)
                    rgba_image[:, :, :3] = object_image
                    rgba_image[:, :, 3] = (mask_cropped * 255).astype(np.uint8)
                    enlarged_size = random.randint(object_size_min, object_size_max)
                    rgba_image = cv2.resize(rgba_image, (enlarged_size, enlarged_size), interpolation=cv2.INTER_LINEAR)
                    radius = np.random.randint(50, disc_radius - 50)
                    angle = np.random.uniform(0, 2 * math.pi)
                    speed = rotation_speed * (1 + np.random.uniform(-0.2, 0.2))
                    disc_objects.append(DiscObject(rgba_image, radius, angle, speed, disc_fade_in_frames))
                if len(disc_objects) == MAX_OBJECTS:
                    disc_objects.pop(0)
            counter += 1
            # Update and draw all objects
            with frame_lock:
                screen_two.fill(0)
                if draw_circle: cv2.circle(screen_two, (center_x, center_y), disc_radius, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                for i, obj in enumerate(disc_objects):
                    obj.update()
                    obj.draw(screen_two, center_x, center_y, i)
                if main_screen_mode == 1 or main_screen_mode == 2:
                    if ret_internal:
                        screen_one[...] = frame_internal
                        screen_one = cv2.cvtColor(screen_one, cv2.COLOR_BGR2RGBA)
                        for c in range(3):  # For B, G, R channels
                            screen_one[:, :, c] = (alpha_image_tall * screen_one[:, :, c] + alpha_overlay_tall * overlay_tall[:, :, c]).astype(np.uint8)
                        screen_one = cv2.cvtColor(screen_one, cv2.COLOR_RGBA2BGR)
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
                    
                    screen_one = cv2.cvtColor(screen_one, cv2.COLOR_BGR2RGBA)
                    for c in range(3):  # For B, G, R channels
                        screen_one[:, :, c] = (alpha_image * screen_one[:, :, c] + alpha_overlay * overlay[:, :, c]).astype(np.uint8)
                    screen_one = cv2.cvtColor(screen_one, cv2.COLOR_RGBA2BGR)

                if main_screen_counter % main_screen_interval == 0:
                    main_screen_counter = 0
                    main_screen_mode += 1
                    if main_screen_mode > 2:
                        main_screen_mode = 0
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

# --- Processing thread management ---
processing_thread = None
processing_stop_event = threading.Event()

def stop_processing_loop():
    global processing_stop_event
    processing_stop_event.set()
    if processing_thread and processing_thread.is_alive():
        processing_thread.join(timeout=5)
    processing_stop_event.clear()

def start_processing_loop():
    global processing_thread
    processing_thread = threading.Thread(target=image_processing_loop, daemon=True)
    processing_thread.start()

@app.post('/api/reload_processing')
def reload_processing():
    stop_processing_loop()
    start_processing_loop()
    return {'ok': True, 'msg': 'Processing loop reloaded'}

@app.post('/api/clear_disc')
def clear_disc():
    global disc_objects
    disc_objects = []
    return {'ok': True, 'msg': 'Cleared Disc'}

@app.get('/api/sample_video')
def get_sample_video():
    return {'current': settings['sample_video']}

@app.post('/api/sample_video')
def set_sample_video(payload: dict = Body(...)):
    settings['sample_video'] = payload['path']
    return {'ok': True, 'current': settings['sample_video']}


# --- Disc & Object Settings ---
@app.get('/api/disc/radius')
def get_disc_radius():
    return {'current': settings['disc_radius']}

@app.post('/api/disc/radius')
def set_disc_radius(payload: dict = Body(...)):
    settings['disc_radius'] = payload['value']
    return {'ok': True, 'current': settings['disc_radius']}

@app.get('/api/disc/max_objects')
def get_max_objects():
    return {'current': settings['max_objects']}

@app.post('/api/disc/max_objects')
def set_max_objects(payload: dict = Body(...)):
    settings['max_objects'] = payload['value']
    return {'ok': True, 'current': settings['max_objects']}

@app.get('/api/disc/object_size_range')
def get_object_size_range():
    return {'current': settings['object_size_range']}

@app.post('/api/disc/object_size_range')
def set_object_size_range(payload: dict = Body(...)):
    settings['object_size_range'] = payload['range']
    return {'ok': True, 'current': settings['object_size_range']}

@app.get('/api/disc/rotation_speed')
def get_rotation_speed():
    return {'current': settings['rotation_speed']}

@app.post('/api/disc/rotation_speed')
def set_rotation_speed(payload: dict = Body(...)):
    settings['rotation_speed'] = payload['value']
    return {'ok': True, 'current': settings['rotation_speed']}

@app.get('/api/disc/fade_in_frames')
def get_fade_in_frames():
    return {'current': settings['fade_in_frames']}

@app.post('/api/disc/fade_in_frames')
def set_fade_in_frames(payload: dict = Body(...)):
    settings['fade_in_frames'] = payload['value']
    return {'ok': True, 'current': settings['fade_in_frames']}

# --- Detection & Model Settings ---
@app.get('/api/yolo/toggle_classes')
def get_toggle_classes():
    return {'current': settings['toggle_classes']}

@app.post('/api/yolo/toggle_classes')
def set_toggle_classes(payload: dict = Body(...)):
    settings['toggle_classes'] = bool(payload['value'])
    return {'ok': True, 'current': settings['toggle_classes']}

# --- Interval & Timing Settings ---
@app.get('/api/interval/main_screen')
def get_main_screen_interval():
    return {'current': settings['main_screen_interval']}

@app.post('/api/interval/main_screen')
def set_main_screen_interval(payload: dict = Body(...)):
    settings['main_screen_interval'] = payload['value']
    return {'ok': True, 'current': settings['main_screen_interval']}

@app.get('/api/interval/object_add')
def get_object_add_interval():
    return {'current': settings['object_add_interval']}

@app.post('/api/interval/object_add')
def set_object_add_interval(payload: dict = Body(...)):
    settings['object_add_interval'] = payload['value']
    return {'ok': True, 'current': settings['object_add_interval']}

# --- Debug & Developer Settings ---
@app.get('/api/debug/mode')
def get_debug_mode():
    return {'current': settings['debug_mode']}

@app.post('/api/debug/mode')
def set_debug_mode(payload: dict = Body(...)):
    settings['debug_mode'] = payload['value']
    return {'ok': True, 'current': settings['debug_mode']}

@app.get('/api/debug/stats')
def get_debug_stats():
    return {'current': settings['show_debug_stats']}

@app.post('/api/debug/stats')
def set_debug_stats(payload: dict = Body(...)):
    settings['show_debug_stats'] = payload['value']
    return {'ok': True, 'current': settings['show_debug_stats']}

@app.get('/api/debug/error_threshold')
def get_error_threshold():
    return {'current': settings['error_threshold']}

@app.post('/api/debug/error_threshold')
def set_error_threshold(payload: dict = Body(...)):
    settings['error_threshold'] = payload['value']
    return {'ok': True, 'current': settings['error_threshold']}

# --- General Settings ---
@app.get('/api/settings')
def get_all_settings():
    return {'settings': settings}

@app.post('/api/settings')
def set_all_settings(payload: dict = Body(...)):
    settings.update(payload['settings'])
    return {'ok': True, 'settings': settings}

# --- Startup event to launch processing thread ---
@app.on_event("startup")
def start_bg_thread():
    start_processing_loop()

# --- Main entry point ---
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000) #, reload=True)

