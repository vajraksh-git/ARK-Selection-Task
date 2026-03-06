import numpy as np
import cv2

# fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

fgbg = cv2.createBackgroundSubtractorMOG2(history=800, varThreshold=40, detectShadows=False)



#--------image cleaning--------
def img_cleaning(img):
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)
    
    return opening



def hough_line(img , theta , threshold):
    y_idx , x_idx = np.nonzero(img)

    rho_max = int(np.sqrt(img.shape[0]**2 + img.shape[1]**2))
    accumulator = np.zeros((2*int(rho_max), int(180/theta)) , dtype=np.int32)
    
    theta_values = np.arange(0, 180, theta)
    sin_values = np.sin(np.deg2rad(theta_values))
    cos_values = np.cos(np.deg2rad(theta_values))

    rho_values = x_idx[:, np.newaxis] * cos_values + y_idx[:, np.newaxis] * sin_values
    rho_idx = np.round(rho_values + rho_max).astype(int)


    for t_idx in range(len(theta_values)):
        np.add.at(accumulator[:, t_idx], rho_idx[:, t_idx], 1)
   

    hough_points = np.argwhere(accumulator > threshold)  
    print(hough_points.shape)
    out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for point in hough_points:
        rho = point[0]-rho_max
        theta0 = point[1] * theta
        a = np.cos(theta0 * np.pi/180)
        b = np.sin(theta0 * np.pi/180)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(b))
        y1 = int(y0 + 1000*(-a))
        x2 = int(x0 - 1000*(b))
        y2 = int(y0 - 1000*(-a))

        cv2.line(out_img, (x1,y1), (x2,y2), (255,0,0), 5)

    arranged_points = np.sort(hough_points[:, 0])
    edge1 =    None
    
    return out_img

    

#----bg subtraction--------
count = 0   
id = 1
while(True):
    frame = cv2.imread(f"./extracted_frames/{id}/frame{count}.jpg")
    if frame is None:
        break
    fgmask = fgbg.apply(frame , learningRate=1.5e-2)
    cleaned_img = img_cleaning(fgmask)

    sobelx = cv2.Sobel(src = cleaned_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobely = cv2.Sobel(src = cleaned_img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)

    magnitude = cv2.magnitude(sobelx, sobely)
    edges = np.uint8(magnitude)

   
    cv2.imshow("frame", frame)
    cv2.imshow("edges", edges)

    line=hough_line(edges, 1, 180)
    cv2.imshow("line", line)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    count += 1

    



cv2.destroyAllWindows()



