import os
import sys
import cv2 
import time
import numpy as np
from ultralytics import YOLO
import math

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
        
        # Calculate rotation angle in degrees (convert from radians)
        rotation_angle = math.degrees(self.angle)
        
        # Get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((self.size//2, self.size//2), rotation_angle, 1.0)
        # Rotate the image and its alpha channel
        rotated_image = cv2.warpAffine(self.image, rotation_matrix, (self.size, self.size), 
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                      borderValue=(0, 0, 0, 0))
        
        # Calculate the top-left corner of the image
        x1 = x - self.size // 2
        y1 = y - self.size // 2
        x2 = x1 + self.size
        y2 = y1 + self.size
        
        # Only draw if the object is within the canvas bounds
        if (x1 >= 0 and x2 < canvas.shape[1] and 
            y1 >= 0 and y2 < canvas.shape[0]):
            # Create a region of interest
            roi = canvas[y1:y2, x1:x2]
            # Create a mask for the alpha channel
            alpha_mask = rotated_image[:, :, 3] / 255.0
            
            # Create expanded mask for stroke effect
            kernel = np.ones((5, 5), np.uint8)
            expanded_mask = cv2.dilate(alpha_mask, kernel, iterations=1)
            expanded_mask = np.expand_dims(expanded_mask, axis=-1)
            
            # Use the predefined color for this object
            border_color = object_colors[color_index]
            
            # Apply colored stroke using expanded mask
            roi = roi * (1 - expanded_mask) + border_color * expanded_mask
            
            # Apply the original image on top using original alpha mask
            alpha_mask = np.expand_dims(alpha_mask, axis=-1)
            roi = roi * (1 - alpha_mask) + rotated_image[:, :, :3] * alpha_mask
            
            canvas[y1:y2, x1:x2] = roi.astype(np.uint8)
        

##############################

fullscreen = False

screen_one = np.zeros((1920, 1080, 3), dtype=np.uint8)  # Vertical orientation
screen_two = np.zeros((480, 480, 3), dtype=np.uint8)  # Keep as RGB for display

if fullscreen:
    cv2.namedWindow("Screen1", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Screen1", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("Screen1", 0, 0)
    cv2.namedWindow("Screen2", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Screen2", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("Screen2", 1080, 0)
else:
    cv2.namedWindow("Screen1", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("Screen2", cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow("Screen1", 0, 0)
    cv2.moveWindow("Screen2", 1080, 0)

def open_camera(device_path):
    device_index = int(device_path.split('video')[-1])
    cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {device_path}")
    return cap

def main():
    # Open both cameras
    internal_cam = open_camera('/dev/video0')
    env_cam = open_camera('/dev/video2')
    
    # Create windows
    cv2.namedWindow('Internal Camera', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Environment Camera', cv2.WINDOW_NORMAL)
    
    print("Cameras opened successfully!")
    print("Press 'q' to quit")
    
    try:
        while True:
            # Read frames from both cameras
            ret1, frame1 = internal_cam.read()
            ret2, frame2 = env_cam.read()
            
            if not ret1 or not ret2:
                print("Failed to read frames from one or both cameras")
                break
            
            # Add labels to frames
            cv2.putText(frame1, "Internal Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame2, "Environment Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frames
            cv2.imshow('Internal Camera', frame1)
            cv2.imshow('Environment Camera', frame2)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Small delay to maintain frame rate
            time.sleep(0.01)
            
    finally:
        # Clean up
        internal_cam.release()
        env_cam.release()
        cv2.destroyAllWindows()
        print("Cameras released")

if __name__ == "__main__":
    main()

#SCREEN ONE
main_screen_mode = 1 # 0 - yolo annotation top and screen two content bottom, 1 camera fill screen
main_screen_interval = 100 # frames between screen mode changes
main_screen_counter = 0 # frames since last screen mode change

# SCREEN TWO
counter = 0
interval = 10 # frames between object creation
center_x = 240
center_y = 240
radius = 200
cv2.circle(screen_two, (center_x, center_y), radius, (255, 255, 255), 2)

# Disc parameters
MAX_OBJECTS = 10 + 1
disc_objects = []
disc_radius = 200
rotation_speed = 0.02  # radians per frame

# Generate a list of random colors for object borders
object_colors = [np.array([np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]) 
                for _ in range(MAX_OBJECTS)]

# Draw the disc outline
cv2.circle(screen_two, (center_x, center_y), disc_radius, (255, 255, 255), 2)

while True:
    ret_environment, frame_environment = environment_camera.read()
    if main_screen_mode == 1:
        ret_internal, frame_internal = internal_camera.read()
        frame_internal = cv2.rotate(frame_internal, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_internal = cv2.resize(frame_internal, (1080, 1920))

    if ret_environment:
        yolo_res = model(frame_environment, imgsz=320, verbose=False)
        annotated_frame = yolo_res[0].plot()
        objects = yolo_res[0].boxes.xyxy.cpu().numpy()

        if counter % interval == 0 and len(objects) > 0:
            #print("adding new object")
            counter = 0
            random_object = np.random.randint(0, len(objects))
            bounding_box = objects[random_object]
            x1, y1, x2, y2 = map(int, bounding_box)
            
            # Get the segmentation mask for this object
            if hasattr(yolo_res[0], 'masks') and yolo_res[0].masks is not None:
                mask = yolo_res[0].masks.data[random_object].cpu().numpy()
                # Resize mask to match bounding box
                mask = cv2.resize(mask, (x2-x1, y2-y1))
                
                # Extract object from the same frame we got the mask from
                object_image = frame_environment[y1:y2, x1:x2].copy()
                
                # Create RGBA image
                rgba_image = np.zeros((y2-y1, x2-x1, 4), dtype=np.uint8)
                rgba_image[:, :, :3] = object_image  # RGB channels
                rgba_image[:, :, 3] = (mask * 255).astype(np.uint8)  # Alpha channel
                
                # Resize to our target size while preserving alpha channel
                rgba_image = cv2.resize(rgba_image, (80, 80))
                
                # Create new disc object with random radius and angle
                radius = np.random.randint(50, disc_radius - 50)
                angle = np.random.uniform(0, 2 * math.pi)
                speed = rotation_speed * (1 + np.random.uniform(-0.2, 0.2))  # Slight speed variation
                disc_objects.append(DiscObject(rgba_image, radius, angle, speed))

            if len(disc_objects) == MAX_OBJECTS:
                disc_objects.pop(0)
        counter += 1

        # Update and draw all objects
        screen_two.fill(0)  # Clear screen
        cv2.circle(screen_two, (center_x, center_y), disc_radius, (255, 255, 255), 2)
        
        for i, obj in enumerate(disc_objects):
            obj.update()
            obj.draw(screen_two, center_x, center_y, i)

        if main_screen_mode == 1 and ret_internal:
            screen_one = frame_internal
        else:
            # Vertical layout: annotated frame on top, disc view on bottom
            annotated_frame = cv2.resize(annotated_frame, (1080, 960))
            screen_one[0:960, 0:1080] = annotated_frame
            
            # Clear bottom half and draw the disc view
            screen_one[960:1920, 0:1080] = 0

            # Resize screen_two to fit bottom half
            resized_screen_two = cv2.resize(screen_two, (960, 960))
            
            # Calculate offsets for bottom half
            x_offset = 960 + (960 - 960) // 2  # Center vertically in bottom half
            y_offset = (1080 - 960) // 2       # Center horizontally
            
            try:
                screen_one[x_offset:x_offset+960, y_offset:y_offset+960] = resized_screen_two
            except ValueError as e:
                print(f"Error during paste: {e}")
                print(f"Attempted to paste {resized_screen_two.shape} into slice of shape {screen_one[x_offset:x_offset+420, y_offset:y_offset+420].shape}")
            
            
        if main_screen_counter % main_screen_interval == 0:
            #print("Switching main screen mode")
            main_screen_counter = 0
            main_screen_mode = 1 - main_screen_mode
        main_screen_counter += 1
        
    else:
        print("Error reading frames")
        error_count += 1
        if error_count > error_threshold: raise Exception("Error threshold reached")

    cv2.imshow("Screen1", screen_one)
    cv2.imshow("Screen2", screen_two)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

environment_camera.release()
internal_camera.release()
cv2.destroyAllWindows()
print("Program ended")