from video import video_run
from ActionRecognize import ActionRecognize
from CollectionData import CollectionData
from LearningModel import load_data, start_learning

if __name__=="__main__":
    print("Select Mode(1: Collect Data / 2: Run Webcam / 3: ActionRecognize / 4: Learning Model)>>")
    num = input()