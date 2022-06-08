import numpy as np
from imutils.video import FPS
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest
from functions.mailer import Mailer
from functions import config
import time, schedule, csv
import smtplib, ssl
from scipy.spatial import distance as dist
from collections import OrderedDict


timer0 = time.time()

class trackable_object:
	def __init__(self, objectID, centroid):
		# Defining the new centriods from the old and storing the ID
		self.objectID = objectID
		self.centroids = [centroid]
		# Checking if the object is counted or not
		self.counted = False


class CentroidTracker:
	def __init__(self, vanish=50, max_distance=50):
	
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.vanish = vanish
		self.max_distance = max_distance

	def register(self, centroid):

		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):

		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):

		if len(rects) == 0:

			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				if self.disappeared[objectID] > self.vanish:
					self.deregister(objectID)
			return self.objects

		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		for (i, (startX, startY, endX, endY)) in enumerate(rects):

			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])
		else:

			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			rows = D.min(axis=1).argsort()
			cols = D.argmin(axis=1)[rows]
			usedRows = set()
			usedCols = set()

			for (row, col) in zip(rows, cols):
		
				if row in usedRows or col in usedCols:
					continue

				if D[row, col] > self.max_distance:
					continue

				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				usedRows.add(row)
				usedCols.add(col)
				
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)
			if D.shape[0] >= D.shape[1]:

				for row in unusedRows:
	
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					if self.disappeared[objectID] > self.vanish:
						self.deregister(objectID)
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		return self.objects

class Mailer:

    def __init__(self):

        self.EMAIL = "saisandeep876@gmail.com"
        self.PASS = ""
        self.PORT = 465
        self.server = smtplib.SMTP_SSL('smtp.gmail.com', self.PORT)

    def send(self, mail):
        self.server = smtplib.SMTP_SSL('smtp.gmail.com', self.PORT)
        self.server.login(self.EMAIL, self.PASS)
        # message to be sent
        SUBJECT = 'More number of people inside'
        TEXT = f'Maximum limit exceeded in the building!'
        message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT)

        # sending the mail
        self.server.sendmail(self.EMAIL, mail, message)
        self.server.quit()



def run():

	# arguments parser
	a_parser = argparse.ArgumentParser()
	a_parser.add_argument("-p", "--prototxt", required=False,
		help="Caffe 'deploy' prototxt file path")
	a_parser.add_argument("-m", "--model", required=True,
		help="pre-trained model's path")
	a_parser.add_argument("-i", "--input", type=str,
		help="input video file path")
	a_parser.add_argument("-o", "--output", type=str,
		help="output video file path")
	# setting the mark_confidence to  0.35
	a_parser.add_argument("-c", "--mark_confidence", type=float, default=0.35,
		help="Filtering poor detections with the lowest possible probability")
	a_parser.add_argument("-s", "--skip-frames", type=int, default=30,
		help="No of frames that are skipped")
	args = vars(a_parser.parse_args())

	# Classes that our Single Shot Detector is trained
	CLASSES = ["background", "cow", "bicycle", "tvmonitor", "horse", "sofa", "sheep", "car", "cat", "boat", "aeroplane", "diningtable", "dog", "chair", "motorbike", "person", "pottedplant", "bus",
		"bottle", "train", "bird"]

	# Load the model
	model = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	# Playing the video
	print("Getting the video started")
	source = cv2.VideoCapture(args["input"])
	editor = None

	# we are setting the frame dimentions as of now
	W = None
	H = None

	# Defining the Centroid tracker and trackable objects
	ct = CentroidTracker(vanish=50, max_distance=55)
	trackerobjects_list = []
	objects_track = {}

	# Defining the frames and the position of objects i.e, up or down
	frames = 0
	down_frames = 0
	up_frames = 0
	x = []
	up_clear_frames=[]
	down_clear_frames=[]

	# defining and starting the FPS
	fps = FPS().start()

	while True:
		# reading all the frames in order
		frame = source.read()
		frame = frame[1] if args.get("input", False) else frame

		# breaking the video
		if args["input"] is not None and frame is None:
			break

		# Converting the BGR frames to RGB frames for deep learning librabries
		frame = imutils.resize(frame, width = 500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		if W is None or H is None:
			(H, W) = frame.shape[:2]

		if args["output"] is not None and editor is None:
			o_video = cv2.VideoWriter_fourcc(*"mp4v")
			editor = cv2.VideoWriter(args["output"], o_video, 30, (W, H), True)

		# defining  the detector Status

		Detector = "Stand_by"
		detector_list = []

		# Defining the object detection method 
		if frames % args["skip_frames"] == 0:
			
			# Defining the new status of the detector
			Detector = "Identifying"
			trackerobjects_list = []

			# Defining the mark from the frame
			mark = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			model.setInput(mark)
			detections = model.forward()

			# Finding the mark_confidence
			for i in np.arange(0, detections.shape[2]):
				
				mark_confidence = detections[0, 0, i, 2]

				# Removing the weak mark_confidence marks of the objects

				if mark_confidence > args["mark_confidence"]:
					
					idx = int(detections[0, 0, i, 1])

					if CLASSES[idx] != "person":
						continue

					# Finding the (x,y) coordinates for the Objects that is the people
					coordinates = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(Begin_X, Begin_Y, Finish_X, Finish_Y) = coordinates.astype("int")


					# Setting a rectangle box around the object and tracking it
					tracker = dlib.correlation_tracker()
					rectangle = dlib.rectangle(Begin_X, Begin_Y, Finish_X, Finish_Y)
					tracker.start_track(rgb, rectangle)

					# Appending the tracker to list
					trackerobjects_list.append(tracker)

		else:

			for tracker in trackerobjects_list:
				# Defining the new Detector of the tracker to Tracking
				Detector = "Tracking"

				# Updating the tracker
				tracker.update(rgb)
				tracker_position = tracker.get_position()

				Begin_X = int(tracker_position.left())
				Begin_Y = int(tracker_position.top())
				Finish_X = int(tracker_position.right())
				Finish_Y = int(tracker_position.bottom())

				# Appending the coordinates to the list
				detector_list.append((Begin_X, Begin_Y, Finish_X, Finish_Y))

		# Defining a line for a reference 

		cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
		cv2.putText(frame, " -- Reference Line -- ", (10, H - ((i * 20) + 200)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

		# Using the Centroid Tracker to compute the centroid coordinates from the old objects
		objects = ct.update(detector_list)

		for (objectID, centroid) in objects.items():
			# checking that the object is in the list 
			to = objects_track.get(objectID, None)

			# Creating new object tracker and appending it to the list
			if to is None:
				to = trackable_object(objectID, centroid)

			else:
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				# Counting Objects
				if not to.counted:
					# The value is negative, it means the object is moving up otherwise down
					if direction < 0 and centroid[1] < H // 2:
						up_frames += 1
						up_clear_frames.append(up_frames)
						to.counted = True

					elif direction > 0 and centroid[1] > H // 2:
						down_frames += 1
						down_clear_frames.append(down_frames)
						
						# If the count of people are grater than threshold then alert the owner
						if sum(x) >= config.Threshold:
							cv2.putText(frame, "-ALERT: limit exceeded-", (10, frame.shape[0] - 80),
								cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
							if config.ALERT:
								print("Preparing to send the email alert.")
								Mailer().send(config.MAIL)
								print("The Alert had been sent")

						to.counted = True
						
					x = []
					# total people inside
					x.append(len(down_clear_frames)-len(up_clear_frames))
					


			# storing the trackable object 
			objects_track[objectID] = to

			# Defining ID and a rectangle box
			text = "ID:{}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

		# Printing the frames and the ID
		info = [
		("Out", up_frames),
		("In", down_frames),
		("Detector", Detector),
		]
        # Printing the count
		info2 = [
		("People count inside", x),
		]

        # Displaying the output
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

		for (i, (k, v)) in enumerate(info2):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

		# Saving the output count in the excel sheet 
		if config.Log:
			d_t = [datetime.datetime.now()]
			d = [d_t, down_clear_frames, up_clear_frames, x]
			export_data = zip_longest(*d, fillvalue = '')

			with open('Log.csv', 'w', newline='') as myfile:
				wr = csv.editor(myfile, quoting=csv.QUOTE_ALL)
				wr.editorow(("End Time", "In", "Out", "Total Inside"))
				wr.editorows(export_data)
				
		# Writing to the file, if not written
		if editor is not None:
			editor.write(frame)

		# Output frame
		cv2.imshow("Monitor/Analyze in Real-Time", frame)
		key = cv2.waitKey(1) & 0xFF

		# In case to break out press s
		if key == ord("s"):
			break

		# Update the FPS
		frames += 1
		fps.update()

		if config.Timer:
			# Intitialzing the video to stop analyzing after 30000s.
			timer = time.time()
			count_sec=(timer-timer0)
			if count_sec > 30000:
				break

	# Display FPS and Time 
	fps.stop()
	print(" Duration: {:.2f}".format(fps.elapsed()))
	print(" FPS: {:.2f}".format(fps.fps()))

	

	# Intialize closing of all the opened windows
	source.release()
	cv2.destroyAllWindows()

if config.Scheduler:
	while 1:
		schedule.run_pending()

else:
	run()
