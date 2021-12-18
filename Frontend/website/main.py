from flask import  Flask, Response, render_template
# from flask_login import login_user, login_required, logout_user, current_user
import cv2, pickle, sqlite3, pyttsx3
import numpy as np
from keras.models import load_model
from threading import Thread
app = Flask(__name__)

# Initialization of text to speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 120)

#Function to convert text to speech
def say_text(text):
    while engine._inLoop:
        pass
    engine.say(text)
    engine.runAndWait()
    
# Load the trained model
model = load_model('Frontend/model/cnn_model_keras2.h5')

#Function that makes frames larger
def rescale_frame(frame, wpercent=1.2, hpercent=1.15):
    width = int(frame.shape[1] * wpercent)
    height = int(frame.shape[0] * hpercent)
    
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

#gets us the handhist, so that we can extract the hand from background.
def get_hand_hist():
	with open("Frontend/hist", "rb") as f:
		hist = pickle.load(f)
	return hist
# def static_frame():
    

def detect_sign():
    
    text = "" #initialize text
    word = "" #initialize word as empty string
    
    count_same_frame = 0#intialize same frame count to 0
       
    cam = cv2.VideoCapture(0)
        
    hist = get_hand_hist()
    x, y, w, h = 370, 100, 250, 250
    
    while True:
        img = cam.read()[1]
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        cv2.filter2D(dst,-1,disc,dst)
        blur = cv2.GaussianBlur(dst, (11,11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        thresh = cv2.merge((thresh,thresh,thresh))
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        thresh = thresh[y:y+h, x:x+w]
        
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        
        old_text = text
        
        #While counters exists we only want to predict from the largest one which will the hand.
        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                save_img = thresh[y1:y1+h1, x1:x1+w1]
                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
                    
                # preprocess live feed for prediction
                save_img = cv2.resize(save_img, (50,50))# resize image to the image size we used to train the model.
                save_img = np.array(save_img, dtype = np.float32)# convert image to an array
                Processed_img = np.reshape(save_img, (1, 50, 50, 1))# reshape array,get the processed image

                # predict
                pred_probab = model.predict(Processed_img)[0]# get the probalilities for the different models
                pred_probab_highest = max(pred_probab)
                pred_class = list(pred_probab).index(pred_probab_highest)# get the predicted class, one with the highes probability
                #if probabilty is greater the 85 fetch sign name from database

                if pred_probab_highest*100 > 90:
                    #fetch predicted class from the database
                    conn = sqlite3.connect("Frontend/sign_language_db.db")
                    cur = conn.cursor()#create a cursor object
                    querry = "SELECT name FROM sign WHERE id="+str(pred_class)
                    for row in cur.execute(querry):
                        text = row[0]
                        
                if old_text == text:
                    count_same_frame += 1
                else:
                    count_same_frame = 0

                if count_same_frame > 8:
                    # if len(text) == 1:
                    #     Thread(target=say_text, args=(text, )).start()
                    word = word + text
                    
                    count_same_frame = 0

            elif cv2.contourArea(contour) < 1000:
                if word != '':

                    Thread(target=say_text, args=(word, )).start()
                text = ""
                word = ""
            else:
                if word != '':

                    Thread(target=say_text, args=(word, )).start()
                text = ""
                word = ""
        cv2.putText(img,text, (470, 80), cv2.FONT_HERSHEY_TRIPLEX, 1.4, (255, 255, 255 ))
        cv2.putText(img, word, (100, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        
        buffer = cv2.imencode('.jpg', rescale_frame(img))[1]
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # keypress = cv2.waitKey(1)
        # if keypress == ord('q'):
        #     break
    
@app.route('/video_feed')
def video_feed():
    return Response(detect_sign(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/tutorials')
def tutorials():
    return render_template("tutorials.html")

# @app.route('/admin')
# @login_required
# def admin():
#     return render_template("admin.html", user=current_user, text = text)


if __name__ == '__main__':
    app.run(debug = True)


