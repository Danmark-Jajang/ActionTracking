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
from socket import *

class Queue():
    def __init__(self):
        self.queue = []
        self.size = 0
        self.MAX_SIZE = 10
        self.front = 0
        
    def __len__(self):
        return self.size
    
    def isEmpty(self):
        return self.size == 0
    
    def isFull(self):
        return self.size >= self.MAX_SIZE
    
    def enqueue(self, data):
        if self.isFull():
            del self.queue[0]
            self.queue.append(data)
        else:
            self.queue.append(data)
            self.size += 1
    
    def peek(self):
        return self.queue[0]
    
    def equals(self):
        chk_data = self.peek()
        for i in self.queue[1:]:
            if chk_data != i:
                return False
        return True

    def get_list(self):
        return self.queue

##################################################################
# you must run this program after run java client!!              #
##################################################################
def ActionRecognize():
    model = LearningModel.model_fn()
    model.load_weights('faction.h5')

    outputData = Queue()
    for _ in range(outputData.MAX_SIZE):
        outputData.enqueue('None')
    
    sequence = []
    sentence = ['None']
    threshold = 0.7
    dummy = np.zeros((5, 30, 1662))

    res = model.predict(dummy)[0]
    
    # communicate java client
    clientSocket = socket(AF_INET, SOCK_STREAM)
    try:
        clientSocket.connect(('localhost', 59892))
    except Exception as e:
        print(e)
    

    cap = cv2.VideoCapture(0) # get number 0 webcam
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read() # read image frome webcam

            image, results = mediapipe_detection(frame, holistic)

            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            #prediction
            keypoints = video.extract_keypoints(results)
            sequence.insert(0, keypoints)
            sequence = sequence[:30]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]

            if res[np.argmax(res)] > threshold:
                outputData.enqueue(actions[np.argmax(res)])
                '''
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
                '''
            
            if outputData.equals():
                printData = outputData.peek()
                if printData != 'None' and sentence[-1]!=printData:
                    sentence.append(printData)
                    print(printData)
                    clientSocket.sendall(bytes(printData+'\n', 'UTF-8'))
            
            # sentence length limit to 5        
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
    clientSocket.close()
    
if __name__=='__main__':
    ActionRecognize()