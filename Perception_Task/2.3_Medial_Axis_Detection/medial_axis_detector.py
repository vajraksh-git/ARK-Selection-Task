import numpy as np
import cv2

fgbg = cv2.createBackgroundSubtractorMOG2(history=800, varThreshold=40, detectShadows=False)

# CHANGE THIS ID TO 1, 2, OR 3 TO TEST ON DIFFERENT VIDEOS
id = 1   

def img_cleaning(img):
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)
    return opening

def hough_line(img, theta, threshold, frame):
    y_idx, x_idx = np.nonzero(img)
    rho_max = int(np.sqrt(img.shape[0]**2 + img.shape[1]**2))
    accumulator = np.zeros((2*rho_max, int(180/theta)), dtype=np.int32)
    
    theta_values = np.arange(0, 180, theta)
    sin_values = np.sin(np.deg2rad(theta_values))
    cos_values = np.cos(np.deg2rad(theta_values))

    rho_values = x_idx[:, np.newaxis] * cos_values + y_idx[:, np.newaxis] * sin_values
    rho_idx = np.round(rho_values + rho_max).astype(int)

    for t_idx in range(len(theta_values)):
        np.add.at(accumulator[:, t_idx], rho_idx[:, t_idx], 1)

    lines =[]
    acc_copy = np.copy(accumulator)
    for _ in range(2):
        idx = np.argmax(acc_copy)
        r_idx, t_idx = np.unravel_index(idx, acc_copy.shape)
        if acc_copy[r_idx, t_idx] > threshold:
            lines.append((r_idx - rho_max, t_idx * theta))
            r_min, r_max = max(0, r_idx - 50), min(acc_copy.shape[0], r_idx + 50)
            t_min, t_max = max(0, t_idx - 20), min(acc_copy.shape[1], t_idx + 20)
            acc_copy[r_min:r_max, t_min:t_max] = 0

    for rho, theta0 in lines:
        a = np.cos(theta0 * np.pi / 180)
        b = np.sin(theta0 * np.pi / 180)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * b)
        y1 = int(y0 + 1000 * -a)
        x2 = int(x0 - 1000 * b)
        y2 = int(y0 - 1000 * -a)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if len(lines) == 2:
        y1_pt, y2_pt = 0, img.shape[0]
        pts =[]
        for rho, theta0 in lines:
            cos_t = np.cos(theta0 * np.pi / 180)
            sin_t = np.sin(theta0 * np.pi / 180)
            if abs(cos_t) > 1e-6:
                x1_pt = int((rho - y1_pt * sin_t) / cos_t)
                x2_pt = int((rho - y2_pt * sin_t) / cos_t)
                pts.append(((x1_pt, y1_pt), (x2_pt, y2_pt)))
        
        if len(pts) == 2:
            mid_x1 = (pts[0][0][0] + pts[1][0][0]) // 2
            mid_x2 = (pts[0][1][0] + pts[1][1][0]) // 2
            cv2.line(frame, (mid_x1, y1_pt), (mid_x2, y2_pt), (0, 0, 255), 5)

    return frame

count = 0   

while True:
    frame = cv2.imread(f"./extracted_frames/{id}/frame{count}.jpg")
    if frame is None:
        break
        
    fgmask = fgbg.apply(frame, learningRate=1.5e-2)
    cleaned_img = img_cleaning(fgmask)

    sobelx = cv2.Sobel(src=cleaned_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobely = cv2.Sobel(src=cleaned_img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)

    magnitude = cv2.magnitude(sobelx, sobely)
    edges = np.uint8(magnitude)

    result_frame = hough_line(edges, 1, 180, frame.copy())

    cv2.imshow("fgmask", fgmask)
    cv2.imshow("edges", edges)
    cv2.imshow("Medial Axis Video", result_frame)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    count += 1

cv2.destroyAllWindows()