import numpy as np
import cv2  
import os
import argparse
from matplotlib import pyplot as plt
import pandas as pd

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin1(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return image_bin

def invert(image):
    return 255 - image

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
    plt.show()

def image_clahe(image_hs):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image_hs)

def label2rgb(image, markers):
    cmap = plt.get_cmap('jet', len(np.unique(markers)))
    return cmap(markers)

def getToadAndBoo(img, hsv):
    lower_white = np.array([0,0,120])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    #display_image(mask_white)

    contours, _ = cv2.findContours(mask_white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_barcode = []

    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour)
        width, height = size
        if width > 23 and width < 90 and height >35 and height < 100:
            contours_barcode.append(contour)
    
    print(len(contours_barcode))
    cv2.drawContours(img, contours_barcode, -1, (255,0,0), 1)
    display_image(img)
    

def getBlackBobomb(hsv):
    lower_black = np.array([0,0,0])
    upper_black = np.array([180, 100, 80])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((3, 3), np.uint8)
    mask_black = cv2.dilate(mask_black, kernel, iterations=2)
    mask_black = cv2.erode(mask_black, kernel, iterations=1)
    display_image(mask_black)

def getRedBobomb(hsv):
    lower_red_1 = np.array([0, 30, 30])
    upper_red_1 = np.array([15, 255, 255])

    lower_red_2 = np.array([165, 30, 30])
    upper_red_2 = np.array([180, 255, 255])

    mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)
    display_image(mask_red)

def getBlueCaps(hsv):
    lower_blue = np.array([80, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((3, 3), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    display_image(mask_blue)

def removeBackground(img, hsv):
    lower_green = np.array([30,40,40])
    upper_green = np.array([90,255,255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    mask_inv[620:, :] = 0
    mask_inv[0:150, :] = 0

    result = cv2.bitwise_and(img, img, mask=mask_inv)

    kernel = np.ones((3, 3), np.uint8)
    result = cv2.erode(result, kernel)
    #display_image(result)
    return result

def calculate_mae(predicted_counts, true_counts):
    errors = np.abs(np.array(predicted_counts) - np.array(true_counts))
    print(true_counts)
    print(predicted_counts)
    print(errors)
    mae = np.mean(errors)
    return mae

def read_true_counts(csv_path):
    df = pd.read_csv(csv_path)
    true_counts = {os.path.splitext(filename)[0]: count for filename, count in zip(df['picture'], df['toad_boo_bobomb'])}
    return true_counts



def process_image(image_path):
    img = load_image(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #display_image(hsv)

    #getToadAndBoo(hsv)    
    #getBlackBobomb(hsv)
    #getRedBobomb(hsv)
    #getBlueCaps(hsv)
    img_without_background = removeBackground(img, hsv)
    hsv_without_background = cv2.cvtColor(img_without_background, cv2.COLOR_BGR2HSV)
    #display_image(hsv_without_background)

    #getToadAndBoo(img, hsv_without_background)
    # contours, _ = cv2.findContours(mask_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # contours_barcode = []

    # for contour in contours:
    #     for dots in contour:
    #         x, y = dots[0]
    #         if y>650:
    #             continue
    #         center, size, angle = cv2.minAreaRect(contour)
    #         height, width = size
    #         if width > 10 and width < 100 and height >20 and height < 200:
    #             contours_barcode.append(contour)
    
    # cv2.drawContours(img, contours_barcode, -1, (255,0,0), 1)
    #display_image(img)

    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images from a dataset folder.')
    parser.add_argument('dataset_folder', type=str, help='Path to the folder containing the dataset images')
    args = parser.parse_args()

    csv_path = os.path.join(args.dataset_folder, "object_count.csv")
    true_counts = read_true_counts(csv_path)

    detected_counts = []
    true_counts_list = []

    for filename in os.listdir(args.dataset_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(filename)
            image_path = os.path.join(args.dataset_folder, filename)
            print(f"Processing {image_path}")
            detected_count = process_image(image_path)
            detected_counts.append(detected_count)
            base_name = os.path.splitext(filename)[0]
            true_count = true_counts.get(base_name, 0)
            true_counts_list.append(true_count)
    
    mae = calculate_mae(detected_counts, true_counts_list)
    print(f"Mean Absolute Error (MAE): {mae}")
