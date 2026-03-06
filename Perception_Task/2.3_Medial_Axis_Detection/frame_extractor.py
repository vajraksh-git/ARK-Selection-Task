import cv2

for id in (1,2,3):
    vid = cv2.VideoCapture(f"./videos/{id}.mp4")

    count, success = 0, True
    while success:  
        success, image = vid.read() # Read frame
        if success: 
            cv2.imwrite(f"./extracted_frames/{id}/frame{count}.jpg", image) # Save frame
            count += 1

    vid.release()



