import cv2
import time
import os
import uuid
import shutil
import pymongo
import numpy as np
import random
from flask import (
    Flask,
    jsonify,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for, Response
)

import threading
from camera import VideoCam



client = pymongo.MongoClient('mongodb://127.0.0.1:27017/')
mydb = client['ojo']

userInfo=mydb.userInfo
camInfo=mydb.camInfo

username1 = ""
password1 = ""
alert=False

cam = {

    "_id": uuid.uuid4().hex,
    "ip": "0",
    "enabled": "false"

}

class Camera:
    def addCamera(self,request):
        cam = {
            "_id":uuid.uuid4().hex,
            "ip": request.form.get('ip1'),
            "enabled": "true"
        }

        camInfo.insert_one(cam)
        return True

    def removeCamera(self, request):
        cam = {

            "_id": uuid.uuid4().hex,
            "ip": request.form.get('ip1'),
            "enabled": "false"

        }
        camInfo.remove(cam)
        return jsonify(cam), 200

    def updateCamera(self, query,camStatus):
        camUpdate = {

            "$set":{
            "enabled": camStatus
            }

        }
        camInfo.update_one(query, camUpdate)


        return jsonify(camUpdate), 200



class Member:

    def start_session(self,member):
        session['logged_in']=True


        return jsonify(member),200

    def signIn(self,username,password):
        member = {
            "username":username,
            "password": password,
        }

        if(userInfo.find_one(member)):
            return True
        else: return False



    def addMember(self,request):
        member = {
            "_id":uuid.uuid4().hex,
            "name": request.form.get('name'),
            "email": request.form.get('email'),
            "username": request.form.get('username'),
            "password": request.form.get('password'),
            "role": request.form.get('role')
        }
        userInfo.insert_one(member)

        return jsonify(member), 200

    def removeMember(self, request):
        member = {

            "email": request.form.get('email'),
            "username": request.form.get('username')

        }
        userInfo.remove(member)

        return jsonify(member), 200

    def updateInfo(self, query,request):
        member = {

            "$set":{
            "name": request.form.get('name'),
            "email": request.form.get('email'),
            "username": request.form.get('username'),
            "password": request.form.get('password')
            }


        }

        userInfo.update_one(query, member)


        return jsonify(member), 200

    def updateRole(self, userName,rolename):
        role = {
            "$set":{
            "role": rolename
            }
        }
        query = {
                "username": userName
        }

        userInfo.update_one(query,  role)


        return jsonify(role), 200

app = Flask(__name__)
app.secret_key = 'somesecretkeythatonlyishouldknow'
APP_ROOT=os.path.dirname(os.path.abspath(__file__))

@app.before_request
def before_request():
    g.user = None
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if(x['_id']==session['user_id']):
                user=x
                g.user=user



@app.route('/login', methods=['GET', 'POST'])
def login():
    session.pop('user_id', None)
    if request.method == 'POST':
        session.pop('user_id', None)

        username1 = request.form['username']
        password1 = request.form['password']

        userData=userInfo.find_one({'username':username1})
        if(userData):
            if(userData['password']==password1):
                session['user_id'] = userData['_id']
                return redirect(url_for('dashboard'))
            return render_template('login.html')

        return render_template('login.html')


    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))


def gen():

    cap = cv2.VideoCapture(0)

    model = cv2.dnn.readNet('yolov3_training_final.weights',
                            'yolov3_training.cfg')

    classes = []
    with open('classes.txt', 'r') as f:
        classes = f.read().splitlines()
    tm=0
    while True:
        ret, image = cap.read()
        if(tm==1):
            tm=0
            img = image

            height, width, _ = img.shape

            blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), swapRB=True, crop=False)
            model.setInput(blob)
            output_layer_names = model.getUnconnectedOutLayersNames()
            ########Forward Pass################
            layer_output = model.forward(output_layer_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in layer_output:
                for prediction in output:
                    scores = prediction[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if (confidence > 0.6):
                        center_x = int(prediction[0] * width)
                        center_y = int(prediction[1] * height)
                        w = int(prediction[2] * width)
                        h = int(prediction[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.4)
            colors = np.random.uniform(0, 255, size=(len(indexes), 3))
            if (len(indexes) != 0):
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])

                    confidence = str(round(confidences[i], 2))
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img[:, :, 0] = 0
                    img[:, :, 1] = 0


                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    #cv2.putText(img, label, (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            image=img

        ret, jpeg = cv2.imencode('.jpg', image)
        data = []
        data.append(jpeg.tobytes())
        frame = data[0]
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
        tm+=1

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



    cap.release()
    cv2.destroyWindow()


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def main():
    return redirect(url_for('dashboard'))


@app.route('/test')
def test():
    return render_template("test.html")



@app.route('/evaluate')
def evaluate():
    if not g.user:
        return redirect(url_for('login'))
    target = os.path.join(APP_ROOT, "static/detectVideo/")
    target1 = os.path.join(APP_ROOT, "static/detectImage/")

    if os.path.exists(target):
        shutil.rmtree(target)
    os.makedirs(target)

    if os.path.exists(target1):
        shutil.rmtree(target1)
    os.makedirs(target1)

    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x
    #if request.method == 'POST':
    #    render_template('evaluate.html', user=user)

    return render_template('evaluate.html',user=user)


def convert_avi_to_mp4(avi_file_path,output_name):
    os.popen(
        "ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(
            input=avi_file_path, output=output_name))
    return True



@app.route('/result',methods=['POST'])
def result():
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x

    target=os.path.join(APP_ROOT,"static/detectVideo/")
    target1 = os.path.join(APP_ROOT, "static/detectImage/")


    if os.path.exists(target):
        shutil.rmtree(target)
    os.makedirs(target)

    if not os.path.isdir(target):
       os.mkdir(target)

    if os.path.exists(target1):
        shutil.rmtree(target1)
    os.makedirs(target1)

    if not os.path.isdir(target1):
        os.mkdir(target1)

    filename=""
    for file in request.files.getlist("file"):

        filename=file.filename
        dest="/".join([target,filename])
        detectedDest = "/".join([target, filename+"Detected"])
        print(dest)
        file.save(dest)

    model = cv2.dnn.readNet('yolov3_training_final.weights',
                            'yolov3_training.cfg')

    classes = []
    with open('classes.txt', 'r') as f:
        classes = f.read().splitlines()

    cap = cv2.VideoCapture(dest)


    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)

    #result = cv2.VideoWriter('static/detect.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)

    iteration=0
    while True:
        ret, image = cap.read()
        if(iteration==15):
            r1=random.random()
            r2=random.random()
            r3=random.uniform(7, 19)
            r4=random.randint(1, 10)

            iteration=0
            if ret == False:
                break
            img = image

            height, width, _ = img.shape

            blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), swapRB=True, crop=False)
            model.setInput(blob)
            output_layer_names = model.getUnconnectedOutLayersNames()
            ########Forward Pass################
            layer_output = model.forward(output_layer_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in layer_output:

                for prediction in output:
                    scores = prediction[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if (confidence > 0.3):
                        center_x = int(prediction[0] * width)
                        center_y = int(prediction[1] * height)
                        w = int(prediction[2] * width)
                        h = int(prediction[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
            if (len(indexes) != 0):
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0,0 , 255), 2)

                    cv2.imwrite(os.path.join(target1, str(r1)+str(r2)+str(r3)+str(r4)+"detectImage"+str(i)+".jpg"), img)

            image = img
            #result.write(image)

        iteration+=1

    cap.release()
    #result.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    #convert_avi_to_mp4('static/detect.avi', 'static/newDetect')
    images = os.listdir(os.path.join(app.static_folder, "detectImage"))

    return render_template("result.html",user=user,images=images,filename=filename)


@app.route('/dashboard')
def dashboard():
    if not g.user:
        return redirect(url_for('login'))
    cursor = userInfo.find({})

    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x

    cursor1 = camInfo.find({})
    for x in cursor1:
        if (x['name'] == "webcam"):
            cam = x

    if user['role'] == "manager":
        return render_template('dashboard.html', user=user,cam=cam)
    else:
        return render_template('staff.html', user=user,cam=cam)


@app.route('/addStaff')
def addStaff():
    if not g.user:
        return redirect(url_for('login'))
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x
    return render_template('addStaff.html',user=user)



@app.route('/removeMember')
def removeMember():
    if not g.user:
        return redirect(url_for('login'))
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x
    return render_template('removeMember.html',user=user)


@app.route('/updateDetails')
def updateDetails():
    if not g.user:
        return redirect(url_for('login'))
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x
                return render_template('updateDetails.html', user=user)

    return redirect(login)


@app.route('/manageRoles')
def manageRoles():
    if not g.user:
        return redirect(url_for('login'))
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x
                return render_template('manageRoles.html', user=user)

    return redirect(login)


@app.route('/userConfiguration')
def userConfiguration():
    if not g.user:
        return redirect(url_for('login'))
    cursor = userInfo.find({})
    if 'user_id' in session:
        for x in cursor:
            if (x['username'] == g.user['username']):
                user = x
    return render_template('addRemoveUser.html',user=user)


#CRUD ROUTES

@app.route('/added',methods=["POST"])
def added():
    cursor = userInfo.find({})
    #if 'user_id' in session:
    for x in cursor:
        if ((x['username'] == request.form.get('username'))):
            return redirect(addStaff)
        if ((x['email'] == request.form.get('email'))):
            return redirect(addStaff)

    mem = Member()
    rep = mem.addMember(request)
    return rep
    #res=Member().addMember()
    #return render_template('added.html',res=res)



@app.route('/remove',methods=["POST"])
def remove():
    cursor = userInfo.find({})
    # if 'user_id' in session:
    for x in cursor:
        if ((x['username'] == request.form.get('username'))):
            mem = Member()
            rep = mem.removeMember(request)
            return rep


    #res=Member().addMember()
    #return render_template('added.html',res=res)
    return redirect(removeMember)


@app.route('/enableCam',methods=["POST"])
def enableCam():

    cursor1 = camInfo.find({})
    for x in cursor1:
        if (x['name'] == "webcam"):
            cam = x
    quer={
        "name":"webcam"
    }

    c1=Camera()
    enable="true"
    rep=c1.updateCamera(quer, enable)

    return rep



@app.route('/disableCam',methods=["POST"])
def disableCam():

    cursor1 = camInfo.find({})
    for x in cursor1:
        if (x['name'] == "webcam"):
            cam = x
    quer={
        "name":"webcam"
    }

    c1=Camera()
    enable="false"
    rep=c1.updateCamera(quer, enable)

    return rep



@app.route('/updateit',methods=["POST"])
def updateit():
    cursor = userInfo.find({})

    if(request.form.get('username')!=g.user['username']):
        for x in cursor:
            if (x['username'] == request.form.get('username')):
                return redirect(updateDetails)

    elif(request.form.get('email')!=g.user['email']):
        for x in cursor:
            if (x['email'] == request.form.get('email')):
                return redirect(updateDetails)

    queryUsername = {
        'username': request.form.get('username')
    }
    mem = Member()
    rep = mem.updateInfo(queryUsername, request)
    #res=Member().addMember()
    #return render_template('added.html',res=res)
    return rep



@app.route('/roleUpdate',methods=["POST"])
def roleUpdate():

    cursor = userInfo.find({})
    for x in cursor:
        if ((x['username'] == request.form.get('username'))):
            mem = Member()
            rep = mem.updateRole(request.form.get("username"), request.form.get("role"))

            return rep

    return redirect(manageRoles)


