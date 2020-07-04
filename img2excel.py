import shutil
from math import sqrt

import cv2
import imutils
import numpy as np
import pandas as pd
from google.cloud.vision import enums
from google.cloud.vision import types

try:
    from PIL import Image
except ImportError:
    import Image
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path to json file containing api credentials"


def detect_text(images_array):
    """Detects text in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    # with io.open(path, 'rb') as image_file:
    #   content = image_file.read()
    features = [
        types.Feature(type=enums.Feature.Type.TEXT_DETECTION)
    ]

    requests = []
    for filename in images_array:
        with open(filename, 'rb') as image_file:
            image = types.Image(
                content=image_file.read())
        request = types.AnnotateImageRequest(
            image=image, features=features)
        requests.append(request)

    response_vision = client.batch_annotate_images(requests)

    # response = client.text_detection(image=image)
    # texts = response.text_annotations
    # print('Texts:')
    if response_vision:
        return response_vision

    # for text in texts:
    # print('\n"{}"'.format(text.description))

    # vertices = (['({},{})'.format(vertex.x, vertex.y)
    # for vertex in text.bounding_poly.vertices])

    # print('bounds: {}'.format(','.join(vertices)))

    if response_vision.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response_vision.error.message))


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def show_img(img_show, title):
    # plt.imshow(img_show, cmap='none')
    # plt.title(title)
    # plt.show()
    # cv2.waitKey(0)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    imS = cv2.resize(img_show, (1577, 708))  # Resize image
    cv2.imshow(title, imS)  # Show image
    cv2.waitKey(0)


def resize(img_resize, height=800):
    rat = height / img_resize.shape[0]
    return cv2.resize(img_resize, (int(rat * img_resize.shape[1]), height))


def angle(pt1, pt2, pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return int(dx1 * dx2 + dy1 * dy2) / sqrt(int(dx1 * dx1 + dy1 * dy1) * int(dx2 * dx2 + dy2 * dy2) + 1e-10)


def sort_contours(cntrs, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(cr) for cr in cntrs]

    (cntrs, boundingBoxes) = zip(*sorted(zip(cntrs, boundingBoxes),
                                         key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes


def delete_files_in_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


# read your file
file = r'table.jpg'
img = cv2.imread(file, 0)
img = resize(img)

img_bin = cv2.medianBlur(img, 3)
# thresholding the image to a binary image
img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, 13)

cv2.fastNlMeansDenoising(img_bin, img_bin, 15, 7, 21)

# inverting the image
img_bin = 255 - img_bin
img_bin = cv2.GaussianBlur(img_bin, (3, 3), 0)
# Plotting the image to see the output
#show_img(img_bin, "Binary Image")

edges = auto_canny(img_bin)

kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3))
edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)
# edges = cv2.dilate(edges, cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(2, 2)))
# edges = img_bin

#show_img(edges, "Canny Image")

bitxor = cv2.bitwise_xor(img, edges)
bitnot = cv2.bitwise_not(bitxor)

contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(contours)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:100]

i = 0
screenCnt = []
for c in cnts:
    # approximate the contour
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(hull, 0.028 * peri, True)
    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    exit
    if len(approx) == 4 and 45000 > cv2.contourArea(approx) > 2500 and cv2.isContourConvex(approx):
        s = 0
        for i in range(4):
            # find minimum angle between joint
            # edges (maximum of cosine)
            if i >= 2:
                t = abs(angle(approx[i], approx[i - 2], approx[i - 1]))
                if s < t:
                    s = t
        if s < 0.3:
            screenCnt.append(approx)

sorted_ctrs = sorted(screenCnt, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * img.shape[1])

cv2.drawContours(img_bin, sorted_ctrs, -1, (0, 255, 0), 2)
#show_img(img, "Contoured Image")

# Create list box to store all boxes in
box = []
# Get position (x,y), width and height for every contour and show the contour on image
for c in sorted_ctrs:
    x, y, w, h = cv2.boundingRect(c)
    # image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    box.append([x, y, w, h])

# Creating a list of heights for all detected boxes
heights = [box[i][3] for i in range(len(box))]

#print("Heights")
#print(heights)

# Get mean of heights
mean = np.mean(heights)

# Creating two lists to define row and column in which cell is located
row = []
column = []
j = 0

# Sorting the boxes to their respective row and column
for i in range(len(box)):

    if i == 0:
        column.append(box[i])
        previous = box[i]

    else:
        if box[i][1] <= previous[1] + mean / 2:
            column.append(box[i])
            previous = box[i]

            if i == len(box) - 1:
                row.append(column)

        else:
            row.append(column)
            column = []
            previous = box[i]
            column.append(box[i])

#print(column)
#print(row)

# calculating maximum number of cells
countcol = 0
for i in range(len(row)):
    countcol = len(row[i])
    if countcol > countcol:
        countcol = countcol

#print("Col Count")
#print(countcol)

# Retrieving the center of each column
center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]

center = np.array(center)
center.sort()
#print("Center")
#print(center)
# Regarding the distance to the columns center, the boxes are arranged in respective order

finalboxes = []
for i in range(len(row)):
    lis = []
    for k in range(countcol):
        lis.append([])
    for j in range(len(row[i])):
        diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
        minimum = min(diff)
        indexing = list(diff).index(minimum)
        lis[indexing].append(row[i][j])
    finalboxes.append(lis)

# from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
outer = []
n = 0
images = []
for i in range(len(finalboxes)):
    for j in range(len(finalboxes[i])):
        inner = ''
        if len(finalboxes[i][j]) == 0:
            outer.append(' ')
        else:
            for k in range(len(finalboxes[i][j])):
                y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                             finalboxes[i][j][k][3]
                finalimg = img[x:x + h, y:y + w]
                n = n + 1
                cv2.imwrite("temp/" + str(n) + ".jpg", finalimg)
                images.append("temp/" + str(n) + ".jpg")

all_images_split = np.array_split(images, len(images) / np.math.ceil(len(images) / 16))
# print(images_split)

cv2.destroyAllWindows()

print("Getting the contents of the table using Google Vision API...")

text_responses = []
for images_split in all_images_split:
    response = detect_text(images_split)
    # print(response)
    for annotation_response in response.responses:
        text_a = annotation_response.text_annotations
        if len(text_a) > 0:
            text_responses.append(text_a[0].description)
        else:
            text_responses.append("")

outer = []
n = 0
for i in range(len(finalboxes)):
    for j in range(len(finalboxes[i])):
        inner = ''
        if len(finalboxes[i][j]) == 0:
            outer.append(' ')
        else:
            for k in range(len(finalboxes[i][j])):
                inner = inner + " " + text_responses[n]
                n = n + 1
            outer.append(inner)

# Creating a dataframe of the generated OCR list
arr = np.array(outer)
dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
print(dataframe)
data = dataframe.style.set_properties(align="left")
# Converting it in a excel-file
data.to_excel("output.xlsx")

print("Excel sheet generated: output.xlsx")

delete_files_in_folder("temp")
