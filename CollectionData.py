import cv2
import mediapipe as mp
import numpy as np
import os
import video
from video import mp_drawing, mp_holistic

# 'None', 'My', 'Name', 'You', 'Where', 'What'
actions = np.array(['None', 'Name', 'My', 'Kim', 'Cooking', 'Bye', '1', '2'])

def CollectionData():
    DATA_PATH = os.path.join('MP_Data')
    # 우리가 탐지할 모션의 이름의 배열

    # 30개의 프레임
    no_sequences = 30

    # 한 비디오당 탐지할 길이
    sequence_length = 30
    for action in actions:
        for sequence in range(no_sequences):
            try:
                # DATA_PATH 하위폴더로 각 Action명을 가지고 그 하위폴더로 각 시퀀스(프레임)을 만듦
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass
        
    cap = cv2.VideoCapture(0) # 0번째(첫 번째) 웹캠
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
        
        for action in actions: # 각 모션별로 looping
            for sequence in range(no_sequences): # 각 시퀀스당
                for frame_number in range(sequence_length): # 각 프레임(30 frame)

                    ret, frame = cap.read() # 웹캠으로부터 이미지를 읽음

                    image, results = video.mediapipe_detection(frame, holistic)
                    # print(results)

                    # landmarks 그리기
                    video.draw_styled_landmarks(image, results)
                    
                    
                    if frame_number == 0: # First frame
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} motion / Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV', image) # 화면에 완전히 처리된 이미지 출력
                        cv2.waitKey(2000) # 모션 인식 준비를 위한 2초 대기
                    else:
                        cv2.putText(image, 'Collecting frames for {} motion / Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV', image) # 화면에 완전히 처리된 이미지 출력
                        
                    keypoints = video.extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_number))
                    np.save(npy_path, keypoints)

                    # 웹캠 닫기
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
    cap.release()
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    CollectionData()