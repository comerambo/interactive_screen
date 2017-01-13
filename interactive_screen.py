import StringIO
from math import sin,cos
import angus
import cv2
import numpy as np
import datetime
import pytz
import base64
import zlib
import subprocess



conn = angus.connect()


def decode_output(sound, filename):
	sound = base64.b64decode(sound)
	sound = zlib.decompress(sound)
	with open(filename, "wb") as f:
		f.write(sound)

# # def face_reco (frame, name, album):
# 	 service2 = conn.services.get_service("face_recognition", version=1)
# 	 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 	 ret, buff = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
# 	 buff = StringIO.StringIO(np.array(buff).tostring())
# 	 job = service.process({"image": buff, "album" : album})
# 	 res = job.result
# 	 for face in res['faces']:
# 		 x, y, dx, dy = face['roi']
# 		 cv2.rectangle(frame, (x, y), (x+dx, y+dy), (0,0,255))
# 	 if face ['names'][0]['confidence']> 0.5:
# 	 if len(face['names']) > 0 :
# 	 name = face['names'][0]['key']
# 	 cv2.putText(frame, "Name = {}".format(name), (x, y),
# 	 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))



def say(le_truc_a_dire):
	service = conn.services.get_service('text_to_speech', version=1)
	job = service.process({'text': le_truc_a_dire, 'lang' : "fr-FR"})
	decode_output(job.result["sound"], "output.wav")
	subprocess.call(["/usr/bin/aplay", "./output.wav"])




def display_picture(big_picture, hauteur, largeur, logo, px, py, size_x, size_y):

	if (py > 0 and size_y+py < hauteur) and (px > 0 and size_x+px < largeur):
		roi2 = big_picture[py:py+size_y, px:px+size_x]

		img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY).copy()

		ret2, mask2 = cv2.threshold(img2gray, 230, 255, cv2.THRESH_BINARY)

		mask_inv2 = cv2.bitwise_not(mask2)

		img1_bg = cv2.bitwise_and(roi2, roi2, mask = mask2)
		logo_fg = cv2.bitwise_and(logo, logo, mask = mask_inv2)

		dst = cv2.add(logo_fg, img1_bg)

		big_picture[py:py+size_y, px:px+size_x] = dst

	return big_picture


def toto(stream_index):
	camera = cv2.VideoCapture(stream_index)
	camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
	camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
	camera.set(cv2.cv.CV_CAP_PROP_FPS, 10)

	hauteur = 480
	largeur = 640
	say_come = True

	if not camera.isOpened():
		print("Cannot open stream of index {}".format(stream_index))
		exit(1)

	print("Input stream is of resolution: {} x {}".format(camera.get(3), camera.get(4)))

	conn = angus.connect()
	service = conn.services.get_service("scene_analysis", version=1)
	service.enable_session()

	service2 = conn.services.get_service("face_recognition", version=1)

	w1_s1 = conn.blobs.create(open("/home/comerambaud/Desktop/interactive_screen/facem1.jpeg", 'rb'))
	w2_s1 = conn.blobs.create(open("/home/comerambaud/Desktop/interactive_screen/facef1.jpeg", 'rb'))
	w3_s1 = conn.blobs.create(open("/home/comerambaud/Desktop/interactive_screen/facem2.jpeg", 'rb'))
	# w9_s1 = conn.blobs.create(open("/home/comerambaud/Desktop/interactive_screen/Screenshotphto.png", 'rb'))
	album = {'Type1': [w1_s1], 'Meuf2': [w2_s1], 'Type2': [w3_s1]}
#, 'Aurelien': [w4_s1], 'Gwennael': [w5_s1], 'Tom': [w6_s1], 'Jean': [w7_s1], 'Marion': [w8_s1]}
	service2.enable_session({"album" : album})

	premiere_fois_mec = True
	premiere_fois_meuf = False

	already_seen={}

	time = datetime.datetime.now()

	gr_une_fois = True
	se_une_fois = True

	vx = 30
	vy = 30
	bx = 100
	by = 100

	bax = 50
	bay = 100
	vbax = 5
	vbay = 5

	img10 = cv2.imread('/home/comerambaud/Desktop/interactive_screen/hpballai.png')
	img10 = cv2.resize(img10, None, fx=0.5, fy=0.5)
	size_y_hp_ballai, size_x_hp_ballai, _ = img10.shape

	while camera.isOpened():
		ret, frame = camera.read()
		if not ret:
			break

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		ret, buff = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 80])
		buff = StringIO.StringIO(np.array(buff).tostring())

		buff2 = StringIO.StringIO(buff.getvalue())



		t = datetime.datetime.now(pytz.utc)
		job = service.process({"image": buff,
							   "timestamp" : t.isoformat(),
							   "camera_position": "facing",
							   "sensitivity": {
								   "appearance": 0.7,
								   "disappearance": 0.7,
								   "age_estimated": 0.4,
								   "gender_estimated": 0.5,
								   "focus_locked": 0.9,
								   "emotion_detected": 0.4,
								   "direction_estimated": 0.8
							   }
		})
		res = job.result

		job2 = service2.process({"image": buff2})
		res2 = job2.result
		# print res2
		if 'faces' in res2:
			for face in res2['faces']:
				x, y, dx, dy = face['roi']
				# cv2.rectangle(frame, (x, y), (x+dx, y+dy), (0,0,255))
				if face ['names'][0]['confidence']> 0.5:
					if len(face['names']) > 0 :

						name = face['names'][0]['key']
						if name in already_seen:
							already_seen[name] = True
						else:
							 already_seen[name] = False

						for key, val in already_seen.iteritems():
							if val == False:
								if name == 'Type1':
									if say_come == True:
										say ('hey salut Type1')
										say_come = False
									else:
										if val == False:
											say("Bonjour "+ name)

		if "error" in res:
			print(res["error"])
		else:
			if "events" in res:
				for event in res["events"]:
					value = res["entities"][event["entity_id"]][event["key"]]
					print("{}| {}, {}".format(event["type"], event["key"], value))

			if len (res["entities"]) == 0:

				img9 = cv2.imread('/home/comerambaud/Desktop/interactive_screen/vif_d_or.jpg')
				img9 = cv2.resize(img9, None, fx=0.02, fy=0.02)
				size_y_vif_d_or, size_x_vif_d_or, _ = img9.shape

				bx = bx + vx
				by = by + vy

				if bx + size_x_vif_d_or >= largeur or bx <= 0:
					vx = -vx

				if by + size_y_vif_d_or >= hauteur or by <= 0:
 					vy = -vy



				bax = bax + vbax
				bay = bay + vbay

				if bax + size_x_hp_ballai >= largeur or bax <= 0:
					vbax = -vbax
					img10 = cv2.flip(img10, 1)

				if bay + size_y_hp_ballai >= hauteur or bay <= 0:
 					vbay = -vbay
				#
				if bx+ size_x_vif_d_or >= bax and bx <= bax+ size_x_hp_ballai and by+ size_y_vif_d_or >= bay and by<= bay + size_y_hp_ballai and bx + size_x_vif_d_or >= bax :
					cv2.putText(frame,"ATTRAPE",
							(170, 75), cv2.FONT_HERSHEY_SIMPLEX,
							2.8, (0, 0, 255))

				frame = display_picture(frame, hauteur, largeur, img10, bax, bay, size_x_hp_ballai, size_y_hp_ballai)
				frame = display_picture(frame, hauteur, largeur, img9, bx, by, size_x_vif_d_or, size_y_vif_d_or)


			else:
				for key, val in res["entities"].iteritems():


					x, y, dx, dy = map(int, val["face_roi"])
					age = val['age']
					gender = val['gender']
					#emotion = val['emotion_anger' or 'emotion_surprise' or 'emotion_sadness' or 'emotion_neutral' or 'emotion_happiness' or 'emotion_smiling_degree' or 'emotion_confidence']
					if val['emotion_anger'] > 0.5:
						emotion = "anger"
					elif val['emotion_surprise'] > 0.5:
						emotion = "surprise"
					elif val['emotion_sadness'] > 0.5:
						emotion = "sadness"
					elif val['emotion_neutral'] > 0.5:
						emotion = "neutral"
					elif val['emotion_happiness'] > 0.5:
						emotion = "happiness"
					elif val['emotion_smiling_degree'] > 0.5:
						emotion = "smiling"
					elif val['emotion_confidence'] < 0.09:
						emotion = "je ne sais pas"
					else:
						emotion = "waiting..."

					face_eye = val ['face_eye']
					face_nose = val ['face_nose']
					face_mouth = val ['face_mouth']
					#print face_nose

					# cv2.circle(frame, (face_eye[0][0], face_eye[0][1]), 3, (0, 255, 0))
					# cv2.circle(frame, (face_eye[1][0], face_eye[1][1]), 3, (0, 255, 0))
					# cv2.circle(frame,(face_nose[0], face_nose[1]), 3, (0,255,0))
					# cv2.circle(frame,(face_mouth[1], face_mouth [0]),3,(0,255,0))
					# cv2.rectangle(frame, (x, y), (x+dx, y+dy), (0, 255, 0), 2)
					# cv2.putText(frame, "(age, gender, emotion) = ({:.1f}, {}, {})".format(age, gender, emotion),
					# 		(x, y), cv2.FONT_HERSHEY_SIMPLEX,
					# 		0.8, (255, 255, 255))


					face_roi = val ["face_roi"]


					size_x =int(face_roi[2])
					size_y =int(face_roi[3])
					size_x_chap = int (face_roi[2]*1.5)
					size_y_chap = int (face_roi[3]*1.5)
					# size_x_vif_d_or = int (face_roi [2])
					# size_y_vif_d_or = int (face_roi [3])
					# img1 = frame
					img2 = cv2.imread('/home/comerambaud/Desktop/interactive_screen/gryffondor.jpeg')
					img3 = cv2.imread('/home/comerambaud/Desktop/interactive_screen/serpentare.jpeg')
					img4 = cv2.imread('/home/comerambaud/Desktop/interactive_screen/choixpeau.jpeg')
					img6 = cv2.imread('/home/comerambaud/Desktop/interactive_screen/r.jpeg')
					img5 = cv2.imread('/home/comerambaud/Desktop/interactive_screen/l2.jpeg')
					# img7 = cv2.imread('/home/comerambaud/Downloads/mex.jpeg')
					img8 = cv2.imread('/home/comerambaud/Desktop/interactive_screen/lunettehp.png')

					img2 = cv2.resize(img2, (size_x, size_y))
					img3 = cv2.resize(img3, (size_x, size_y))
					img4 = cv2.resize(img4, (size_x_chap, size_y_chap))
					img5 = cv2.resize(img5, (size_x, size_y))
					img6 = cv2.resize(img6, (size_x, size_y))
					# img7 = cv2.resize(img7, (size_x, size_y))
					img8 = cv2.resize(img8, (size_x, size_y))


					pxm =int(face_roi[0] + face_roi[2])
					pym =int(face_roi[1] - face_roi[3])

					pxc = int(face_roi[0] - size_x_chap/4 + 10)
					pyc = int(face_roi[1] - size_y_chap)

					pxr = int(face_roi[0])
					pyr = int(face_roi[1] + face_roi[1]/8)

					pxa = int(face_roi[0] + 10)
					pya = int(face_roi[1] - face_roi[1]/9)

					pxs = int (face_roi[0])
					pys = int (face_roi[1] - face_roi[3])

					pxe = int (face_roi[0] + 10)
					pye = int (face_roi[1] - 30)


					emotion_happiness = val ['emotion_happiness']
					gender = val ['gender']
					if emotion_happiness > 0.6:
						frame = display_picture(frame, hauteur, largeur, img2, pxm, pym, size_x, size_y)
						frame = display_picture(frame, hauteur, largeur, img8, pxe, pye, size_x, size_y)
						if gr_une_fois == True:
							say ('Bravo, tu es a Gryffondor')
							gr_une_fois = False
					else:
						frame = display_picture(frame, hauteur, largeur, img3, pxm, pym, size_x, size_y)
						frame = display_picture(frame, hauteur, largeur, img5, pxa, pya, size_x, size_y)
						if se_une_fois == True:
							say ('Bravo tu es a serpentar')
							se_une_fois = False

					frame = display_picture(frame, hauteur, largeur, img4, pxc, pyc, size_x_chap, size_y_chap)

					if val ['gender'] == "male" :

						#### a debuguer
						if premiere_fois_mec == True:
							say("hey, salut mecton")
							premiere_fois_mec = False
					else:
						frame = display_picture(frame, hauteur, largeur, img6, pxr, pyr, size_x, size_y)
						if premiere_fois_meuf == True:
							say("hey, salut meuf")
							premiere_fois_meuf = False
					# frame = display_picture(frame, hauteur, largeur, img7, pxc, pyc, size_x, size_y)


				# # if (face_roi[0] < 0 and face_roi[1] < 0):
				# 	roi = img9[pyv:pyv+rows, pxv:pxv+cols]
				#
				# 	# Now create a mask of logo and create its inverse mask also
				# 	img9gray = cv2.cvtColor(img9,cv2.COLOR_BGR2GRAY).copy()
				# 	ret9, mask9 = cv2.threshold(img9gray, 230, 255, cv2.THRESH_BINARY)
				# 	mask_inv9 = cv2.bitwise_not(mask4)
				#
				# 	# Now black-out the area of logo in ROI
				# 	img1_bg = cv2.bitwise_and(roi,roi,mask = mask9)
				# 	# Take only region of logo from logo image.
				# 	img9_fg = cv2.bitwise_and(img9,img9,mask = mask_inv9)
				#
				# 	# Put logo in ROI and modify the main image
				# 	dst = cv2.add(img9_fg, img1_bg)
				# 	img1[py:py+rows, px:px+cols] = dst




					nose = val ['face_nose']
					nose = (nose[0], nose[1])
					eyel = val ['face_eye'][0]
					eyel = (eyel[0], eyel[1])
					eyer = val ['face_eye'][1]
					eyer = (eyer[0], eyer[1])

					psi = val ['head'][2]
					theta = - val ['head'][0]
					phi = val ['head'][1]

					### head orientation
					length = 150
					xvec = int(length*(sin(phi)*sin(psi) - cos(phi)*sin(theta)*cos(psi)))
					yvec = int(- length*(sin(phi)*cos(psi) - cos(phi)*sin(theta)*sin(psi)))
					#cv2.line(frame, nose, (nose[0]+xvec, nose[1]+yvec), (0, 140, 255), 3)

					psi = 0
					theta = -val['gaze'][0]
					phi = val['gaze'][1]

					### gaze orientation
					length = 150
					xvec = int(length*(sin(phi)*sin(psi) - cos(phi)*sin(theta)*cos(psi)))
					yvec = int(- length*(sin(phi)*cos(psi) - cos(phi)*sin(theta)*sin(psi)))
					# cv2.line(frame, eyel, (eyel[0]+xvec, eyel[1]+yvec), (0, 140, 0), 3)

					xvec = int(length*(sin(phi)*sin(psi) - cos(phi)*sin(theta)*cos(psi)))
					yvec = int(- length*(sin(phi)*cos(psi) - cos(phi)*sin(theta)*sin(psi)))
					# cv2.line(frame, eyer, (eyer[0]+xvec, eyer[1]+yvec), (0, 140, 0), 3)
					# print xvec, yvec

					current_time = datetime.datetime.now()
					# print current_time
					if current_time - time >= datetime.timedelta(seconds=5) and abs(xvec) > 30 and abs(yvec) > 30:
						time = datetime.datetime.now()
					# print "Time :", time
						say("Regarde moi bordel")


		cv2.imshow("test", frame)

		if cv2.waitKey(1) & 0xFF == 27:
			 break

	service.disable_session()
	service2.disable_session()

	camera.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	print "c'est parti mon kiki"
	toto(0)
	print "c'est fini"
