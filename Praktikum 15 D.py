import pickle
from unittest import result

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('D:\\SEMESTER 6\\PRAKTIKUM 15\\Documents\\phyton\\buku\\pelatihan\\model.p', 'rb'))
model=model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(max_num_hands=1)
labels_dict = {0: 'tutup', 1: 'buka'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,mp_drawing_style.get_default_hand_landmarks_style(),mp_drawing_style.get_default_hand_connections_style())
            for lm in hand_landmarks.landmark:
                x = lm.x
                y = lm.y
                x_.append(x)
                y_.append(y)

            for i in range(len(x_)):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            cv2.putText(frame, predicted_character, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,3),3)

        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()