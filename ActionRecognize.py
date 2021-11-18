import tensorflow as tf
import cv2
from video import mp_drawing, mp_holistic, mediapipe_detection, draw_styled_landmarks
from tensorflow import keras
import os
import numpy as np
import video
import LearningModel
from LearningModel import actions
import CollectionData

model = LearningModel.model_fn()
model.load_weights('action.h5')

sequence = []
sentence = []
threshold = 0.4
dummy = np.zeros((5, 30, 1662))

res = model.predict(dummy)[0]

cap = cv2.VideoCapture(0) # get number 0 webcam
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read() # read image frome webcam

        image, results = mediapipe_detection(frame, holistic)
        #print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        #prediction
        keypoints = video.extract_keypoints(results)
        sequence.insert(0, keypoints)
        sequence = sequence[:30]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])

        if res[np.argmax(res)] > threshold:
            if len(sentence) > 0:
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
            else:
                sentence.append(actions[np.argmax(res)])
        if len(sentence) > 5:
            sentence = sentence[-5:]
        
        # append infomation to cv2
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        
        cv2.imshow('OpenCV', image) # print image to screen

        # close cv2
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()