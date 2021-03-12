import cv2
import numpy as np
import pytesseract
import os

# function for cropping each detection and saving as new image
def crop_objects(image, xmin, ymin, xmax, ymax):
    # crop detection from image (take an additional 5 pixels around all edges)
    cropped_img = image[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
    # cropped_img = img[y-3:y+h+3, x+3:x+w-3]
    # cropped_img = cv2.resize(cropped_img,None,fx=3,fy=3)
    cv2.imshow("image",cropped_img)
    cv2.waitKey(500)
    return cropped_img

def write_image(image, path, class_name, cropped=False):
    
    if not os.path.exists(path):
        os.makedirs(path)

    label = str(class_name).replace(' ', '_')
    img_name = 'detected_' + label + '.png'
    
    if cropped:
        img_name = 'cropped_' + label + '.png'
    
    img_path = os.path.join(path, img_name)
    cv2.imwrite(img_path, image)

def recognize(image):
    # point to license plate image (works well with custom crop function)
    img = image
    img = np.array(img, dtype=np.uint8)

    resize = cv2.resize(img, None, fx = 3, fy = 3)
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)

    # perform otsu thresh (using binary inverse since opencv contours work better with white text)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # apply dilation 
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)

    # find contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # create copy of image
    im2 = gray.copy()

    plate_num = ""
    height, width = im2.shape

    for contour in sorted_contours:
        try:
            x,y,w,h = cv2.boundingRect(contour)
            
            if height / float(h) > 6: continue
            if h / float(w) < 1.5: continue
            if width / float(w) > 25: continue
            area = h * w
            # if area < 100: continue
            
            rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (255,255,255),2)
            roi = thresh[y-5:y+h+5, x-5:x+w+5]
            # roi = cv2.bitwise_not(roi)
            # roi = cv2.medianBlur(roi, 5)

            text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 10 --oem 3')
            text = text.replace("\n","")
            text = text.replace("\x0c","")

            plate_num += text
            cv2.putText(im2, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 5)
        except Exception as e:
            print(e)
            continue
    
    return plate_num
    # print(plate_num)
    # cv2.imshow("Character's Segmented", im2)
    # cv2.waitKey(500)
    # cv2.destroyAllWindows()
