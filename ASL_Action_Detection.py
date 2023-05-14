import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

actions = np.array(['right', 'left', 'up', 'down', 'hit']) #actions
label_map = {label:num for num, label in enumerate(actions)}


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss="categorical_crossentropy", metrics=['categorical_accuracy'])
model.load_weights('action.h5')

mp_hands = mp.solutions.hands #hands model
mp_drawing = mp.solutions.drawing_utils

def mediapie_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image, results

def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                mp_drawing.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)

def extract_keypoints(results):
    all_hand_landmarks = np.array([[results.multi_hand_landmarks[0].landmark[n].x, 
                  results.multi_hand_landmarks[0].landmark[n].y,
                  results.multi_hand_landmarks[0].landmark[n].z] 
                  for n in range(len(results.multi_hand_landmarks[0].landmark))]).flatten() if results.multi_hand_landmarks else np.zeros(21 * 3)
    return all_hand_landmarks

colors = [(144, 238, 144), (230, 216, 173), (224, 225, 225), (227, 195, 203), (0, 165, 255)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
    
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

threshold = 0.4
sequence = []
res =[]


cap = cv2.VideoCapture(0)

#mediapipe model
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        
        #Run detection
        image, results = mediapie_detection(frame, hands)
        
        #Draw Detections
        draw_landmarks(image, results)
        
        #Prediction
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            
        image = prob_viz(res, actions, image, colors)
            

        cv2.imshow("Frame", image)
        
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()