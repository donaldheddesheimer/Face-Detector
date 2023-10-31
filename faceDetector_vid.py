import cv2
data = cv2.CascadeClassifier('faceDetector_ai/haarcascade_frontalface_default.xml')
key = cv2.waitKey(1)

# turns on your webcam by the nth webcam, 0 is default
webcam = cv2.VideoCapture(0)

# loop to keep drawing rectangles during video
while True:
    frame_success, frame = webcam.read() # .read() outputs (success<bool>, img<obj>)
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    coords = data.detectMultiScale(grayscale_img)
    for (x,y,w,h) in coords:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0) ,1)
         
    cv2.imshow('Face Detector', frame)
    cv2.waitKey(1) # waits 1 ms
    
    # makes sure to end detection after Q key press
    if(key==81 or key==113):
        break # currently doesn't work rip
    
webcam.release()
