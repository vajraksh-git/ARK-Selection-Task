import numpy as np
import cv2

def calculate_snr(image):
    mean = np.mean(image)
    std = np.std(image)
    if std == 0:
        return float('inf')
    return 20 * np.log10(mean / std)

img_path_1 = 'iron_man_noisy.jpg'
img_1 = cv2.imread(img_path_1 , cv2.IMREAD_UNCHANGED)

def img_cleaning(img):
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    blured_img = cv2.medianBlur(img, 1)
    closed = cv2.morphologyEx(blured_img, cv2.MORPH_CLOSE, kernel_close)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # sharpen_kernel = np.array([[-1, -1, -1],
    #                             [-1,  9, -1],
    #                             [-1, -1, -1]])
                               
    
    # unblurred_img = cv2.filter2D(opened, -1, sharpen_kernel)


    return opened

print(img_1.shape)
cleaned_img = img_cleaning(img_1)

snr_before = calculate_snr(img_1)
snr_after = calculate_snr(cleaned_img)

print(f"SNR Before: {snr_before:.2f} dB")
print(f"SNR After: {snr_after:.2f} dB")

cv2.imshow(f"noisy_{img_path_1}", img_1)
cv2.imshow(f"cleaned_{img_path_1}", cleaned_img)

cv2.imwrite(f"cleaned_{img_path_1}", cleaned_img)

cv2.waitKey(0)
cv2.destroyAllWindows()