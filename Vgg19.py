from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np
import cv2
from keras.models import load_model

model = load_model('st.h5')
image_labels = {
    'combat':0,
    'humanitarianaid':3,
    'militaryvehicles':4,
    'fire':2,
    'destroyedbuilding':1 
}

image_path = 'train/combat/45.jpeg'
img_arr = cv2.imread(image_path)
img_arr = cv2.resize(img_arr, (224, 224))  
img_arr = np.reshape(img_arr, [1, 224, 224, 3])

y_pred = model.predict(img_arr)
y_pred = np.argmax(y_pred, axis=1)


predicted_label = y_pred[0]
print(list(image_labels.keys())[list(image_labels.values()).index(predicted_label)])

