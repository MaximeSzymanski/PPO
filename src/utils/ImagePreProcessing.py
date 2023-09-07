import cv2

def resize_frame(frame):
    frame = frame[35:,:]
    # Convert to grayscale
    # Resize
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame