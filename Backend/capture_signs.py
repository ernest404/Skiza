import numpy as np
import os
import cv2
import sqlite3


# create data directory, where images are to be stored after capture
if not os.path.exists('signs'):
    os.mkdir('signs')
    
# Create database to store gesture labels
if not os.path.exists('sign_language_db.db'):
    conn = sqlite3.connect('sign_language_db.db')
    querry_1 = 'CREATE TABLE sign(id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, name TEXT NOT NULL)'
    conn.execute(querry_1)
    conn.commit()
    
# request for the sign id and name just before capturing
sign_id = input('Enter sign id: ')
sign_name = input('Enter sign name:')

# Store sign id and name to the database
conn = sqlite3.connect('sign_language_db.db')
querry_2 = "INSERT INTO sign(id, name) VALUES({}, \'{}\')".format(sign_id, sign_name)

# Exception handling when the gesture already exists in the database
try:
    conn.execute(querry_2)
except sqlite3.IntegrityError:
    choice = input("Sign id already exists.Do you want to make changes ? (y/n)")
    if choice.lower() == 'y':
        querry_3 = "UPDATE sign SET name = \'{}\' WHERE id = {}".format(sign_name, sign_id)
        conn.execute(querry_3)
    else:
        print("Aborting...")
conn.commit()

def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

# Capture 1200 images for each sign
if not os.path.exists("signs/"+str(sign_id)):
		os.mkdir("signs/"+str(sign_id))
# Initialize the camera, and create object
cam = cv2.VideoCapture(0)

hist = get_hand_hist()
# Set the dimensions for the thresh window
x, y, w, h = 300, 100, 300, 300
pic_no = 0
start_capturing = False
frames = 0

while True:
    img = cam.read()[1]
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (640, 480))
    imgCrop = img[y:y+h, x:x+w]
    imgHSV = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    cv2.filter2D(dst,-1,disc,dst)
    blur = cv2.GaussianBlur(dst, (11,11), 0)#Blurring to reduce noise
    blur = cv2.medianBlur(blur, 15)
    thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thresh = cv2.merge((thresh,thresh,thresh))
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    threshclr = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)

    
    
    # Find counters/complete structures in the image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    
    # If there exists more than one complete structure set parameters below.
    if len(contours) > 0:
        
        # if there more than one contours select the largest 
        contour = max(contours, key = cv2.contourArea)
        
        # start capturing after 50 frames to give time for user to get ready and If the area of largest contour is more than 10,000
        if cv2.contourArea(contour) > 10000 and frames > 50:
            
            # crop the preprocessed image with the bounding rectangle of the main contour.
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            pic_no += 1 #keep a count of the image
            save_img = thresh[y1:y1+h1, x1:x1+w1]#crop image by the bounding rctangle
            
            if w1 > h1:
                save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))#landscape
            elif h1 > w1:
                save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))#portriat
            # change size of the image to 50 by 50 for storage.
            save_img = cv2.resize(save_img, (50, 50))
            
            # Randomly flip images as they are captured to create variance.
            rand = np.random.randint(0, 10)
            if rand % 2 == 0:
                save_img = cv2.flip(save_img, 1)     
            # Insert text to indicate to user when capturing each frame
            cv2.putText(img, "Capturing...", (320, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (10, 10, 155))       
            # save preprocessed image in the signs folder
            cv2.imwrite("signs/"+str(sign_id)+"/"+str(pic_no)+".jpg", save_img)

        #Draw up capture area on the camera feed
    cv2.rectangle(img, (x,y), (x+w, y+h), (120,25,10), 2)
    
    #Insert text to show the number of images captured
    cv2.putText(img, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
    
    #Display the capture window, name it Webcam
    cv2.imshow("Data capture", img)
    
    # Press c to start capturing
    if cv2.waitKey(1) == ord('c'):
        if start_capturing == False:
            start_capturing = True
        else:
            start_capturing = False
            frames = 0
    if start_capturing == True:
        frames += 1           
    if (pic_no == 1500) or cv2.waitKey(1) == ord('q'):
        break
