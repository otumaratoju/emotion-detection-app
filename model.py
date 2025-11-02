import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers.legacy import Adam

class EmotionDetector:
    def __init__(self, model_path='emotion_model.h5'):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.model = self._create_model()
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
            except:
                print("Error loading model, using default...")
        else:
            print("No pre-trained model found, using default model...")
    
    def _create_model(self):
        """Create the CNN model architecture"""
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _train_model(self):
        """Train the model with the FER2013 dataset"""
        # Note: In a production environment, you should load and use the actual FER2013 dataset
        # For this example, we're using dummy data
        X_train = np.random.randn(100, 48, 48, 1)
        y_train = np.random.randint(0, 7, size=(100,))
        y_train = tf.keras.utils.to_categorical(y_train, 7)
        
        self.model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.2)
    
    def preprocess_image(self, image_path):
        """Preprocess image for emotion detection"""
        # Read image and convert to grayscale
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        # Process the first face found
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        
        return face
    
    def detect_emotion(self, image_path):
        """Detect emotion from an image file"""
        face = self.preprocess_image(image_path)
        if face is None:
            return {'error': 'No face detected'}
        
        predictions = self.model.predict(face)
        emotion_index = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_index])
        
        return {
            'emotion': self.emotions[emotion_index],
            'confidence': confidence
        }
    
    def detect_emotion_from_array(self, face_array):
        """Detect emotion from a numpy array (for video stream)"""
        # Preprocess the face array
        face = face_array.astype('float32') / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        
        predictions = self.model.predict(face)
        emotion_index = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_index])
        
        return {
            'emotion': self.emotions[emotion_index],
            'confidence': confidence
        }