# '''
# 	Contoh Deloyment untuk Domain Computer Vision (CV)
# 	Orbit Future Academy - AI Mastery - KM Batch 3
# 	Tim Deployment
# 	2022
# '''

# # =[Modules dan Packages]========================

# from flask import Flask,render_template,request,jsonify
# from werkzeug.utils import secure_filename
# import pandas as pd
# import numpy as np
# import os
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
# Flatten, Dense, Activation, Dropout,LeakyReLU
# from PIL import Image
# from fungsi import make_model

# # =[Variabel Global]=============================

# app = Flask(__name__, static_url_path='/static')

# app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
# app.config['UPLOAD_EXTENSIONS']  = ['.jpg','.JPG']
# app.config['UPLOAD_PATH']        = './static/images/uploads/'

# model = None

# NUM_CLASSES = 10
# cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", 
#                    "dog", "frog", "horse", "ship", "truck"]

# # =[Routing]=====================================

# # [Routing untuk Halaman Utama atau Home]
# @app.route("/")
# def beranda():
# 	return render_template('index.html')

# # [Routing untuk API]	
# @app.route("/api/deteksi",methods=['POST'])
# def apiDeteksi():
# 	# Set nilai default untuk hasil prediksi dan gambar yang diprediksi
# 	hasil_prediksi  = '(none)'
# 	gambar_prediksi = '(none)'

# 	# Get File Gambar yg telah diupload pengguna
# 	uploaded_file = request.files['file']
# 	filename      = secure_filename(uploaded_file.filename)
	
# 	# Periksa apakah ada file yg dipilih untuk diupload
# 	if filename != '':
	
# 		# Set/mendapatkan extension dan path dari file yg diupload
# 		file_ext        = os.path.splitext(filename)[1]
# 		gambar_prediksi = '/static/images/uploads/' + filename
		
# 		# Periksa apakah extension file yg diupload sesuai (jpg)
# 		if file_ext in app.config['UPLOAD_EXTENSIONS']:
			
# 			# Simpan Gambar
# 			uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
			
# 			# Memuat Gambar
# 			test_image         = Image.open('.' + gambar_prediksi)
			
# 			# Mengubah Ukuran Gambar
# 			test_image_resized = test_image.resize((32, 32))
			
# 			# Konversi Gambar ke Array
# 			image_array        = np.array(test_image_resized)
# 			test_image_x       = (image_array / 255) - 0.5
# 			test_image_x       = np.array([image_array])
			
# 			# Prediksi Gambar
# 			y_pred_test_single         = model.predict_proba(test_image_x)
# 			y_pred_test_classes_single = np.argmax(y_pred_test_single, axis=1)
			
# 			hasil_prediksi = cifar10_classes[y_pred_test_classes_single[0]]
			
# 			# Return hasil prediksi dengan format JSON
# 			return jsonify({
# 				"prediksi": hasil_prediksi,
# 				"gambar_prediksi" : gambar_prediksi
# 			})
# 		else:
# 			# Return hasil prediksi dengan format JSON
# 			gambar_prediksi = '(none)'
# 			return jsonify({
# 				"prediksi": hasil_prediksi,
# 				"gambar_prediksi" : gambar_prediksi
# 			})

# # =[Main]========================================		

# if __name__ == '__main__':
	
# 	# Load model yang telah ditraining
# 	model = make_model()
# 	model.load_weights("model_cifar10_cnn_tf.h5")

# 	# Run Flask di localhost 
# 	app.run(host="localhost", port=5000, debug=True)
	
	

































# from flask import Flask, render_template, request
# import cv2
# from keras.models import load_model
# import numpy as np

# app = Flask(__name__)

# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/after', methods=['GET', 'POST'])
# def after():
#     img = request.files['file1']

#     img.save('static/file.jpg')

#     ####################################
#     img1 = cv2.imread('static/file.jpg')
#     gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
#     faces = cascade.detectMultiScale(gray, 1.1, 3)

#     for x,y,w,h in faces:
#         cv2.rectangle(img1, (x,y), (x+w, y+h), (0,255,0), 2)

#         cropped = img1[y:y+h, x:x+w]

#     cv2.imwrite('static/after.jpg', img1)

#     try:
#         cv2.imwrite('static/cropped.jpg', cropped)

#     except:
#         pass

#     #####################################

#     try:
#         image = cv2.imread('static/cropped.jpg', 0)
#     except:
#         image = cv2.imread('static/file.jpg', 0)

#     image = cv2.resize(image, (48,48))

#     image = image/255.0

#     image = np.reshape(image, (1,48,48,1))

#     model = load_model('model_3.h5')

#     prediction = model.predict(image)

#     label_map =   ['Anger','Neutral' , 'Fear', 'Happy', 'Sad', 'Surprise']

#     prediction = np.argmax(prediction)

#     final_prediction = label_map[prediction]

#     return render_template('after.html', data=final_prediction)

# if __name__ == "__main__":
#     app.run(debug=True)







































# from flask import Flask, render_template, request
# import cv2
# import numpy as np 
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

# @app.route('/')
# def index():
# 	return render_template('index.html')

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
# 	image = request.files['select_file']

# 	image.save('static/file.jpg')

# 	image = cv2.imread('static/file.jpg')

# 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
	
# 	faces = cascade.detectMultiScale(gray, 1.1, 3)

# 	for x,y,w,h in faces:
# 		cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

# 		cropped = image[y:y+h, x:x+w]


# 	cv2.imwrite('static/after.jpg', image)
# 	try:
# 		cv2.imwrite('static/cropped.jpg', cropped)

# 	except:
# 		pass



# 	try:
# 		img = cv2.imread('static/cropped.jpg', 0)

# 	except:
# 		img = cv2.imread('static/file.jpg', 0)

# 	img = cv2.resize(img, (48,48))
# 	img = img/255

# 	img = img.reshape(1,48,48,1)

# 	model = load_model('model.h5')

# 	pred = model.predict(img)


# 	label_map = ['Anger','Neutral' , 'Fear', 'Happy', 'Sad', 'Surprise']
# 	pred = np.argmax(pred)
# 	final_pred = label_map[pred]


# 	return render_template('predict.html', data=final_pred)


# if __name__ == "__main__":
# 	app.run(debug=True)






































from flask import Flask, render_template, request
import cv2
from keras.models import load_model
import numpy as np

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index1')
def index1():
    return render_template('index1.html')

@app.route('/marah')
def marah():
    return render_template('marah.html')
    
@app.route('/senang')
def senang():
    return render_template('senang.html')


@app.route('/after', methods=['GET', 'POST'])
def after():
    img = request.files['file1']

    img.save('static/hasil/file.jpg')

    ####################################
    img1 = cv2.imread('static/hasil/file.jpg')
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 3)

    for x,y,w,h in faces:
        cv2.rectangle(img1, (x,y), (x+w, y+h), (0,255,0), 2)

        cropped = img1[y:y+h, x:x+w]

    cv2.imwrite('static/hasil/after.jpg', img1)

    try:
        cv2.imwrite('static/hasil/cropped.jpg', cropped)

    except:
        pass

    #####################################

    try:
        image = cv2.imread('static/hasil/cropped.jpg', 0)
    except:
        image = cv2.imread('static/hasil/file.jpg', 0)

    image = cv2.resize(image, (48,48))

    image = image/255.0

    image = np.reshape(image, (1,48,48,1))

    model = load_model('model-acc-60.h5')

    prediction = model.predict(image)

    # label_map =   ['Anger','Neutral' , 'Fear', 'Happy', 'Sad', 'Surprise']

    label_map =   ['Marah', 'Bosan', 'Malas', 'Senang', 'Sedih', 'Terkejut']

    prediction = np.argmax(prediction)

    final_prediction = label_map[prediction]

    return render_template('after.html', data=final_prediction)

if __name__ == "__main__":
    app.run(debug=True)