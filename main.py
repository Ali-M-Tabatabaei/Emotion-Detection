from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten 
from keras.optimizers import Adam 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.optimizers.schedules import ExponentialDecay 
import cv2 
from keras.models import model_from_json 
import numpy as np
import matplotlib.pyplot as plt 


class Detector:
    def __init__(self,train_data, test_data) -> None:
        self.train_data_gen = ImageDataGenerator(rescale=1./255) 
        self.validation_data_gen = ImageDataGenerator(rescale=1./255) 
        self.train_generator = self.train_data_gen.flow_from_directory( 
	        train_data, 
	        target_size=(48, 48), 
	        batch_size=64, 
	        color_mode="grayscale", 
	        class_mode='categorical') 
        
        self.validation_generator = self.validation_data_gen.flow_from_directory( 
        test_data, 
        target_size=(48, 48), 
        batch_size=64, 
        color_mode="grayscale", 
        class_mode='categorical')
        self.emotion_model = None
        self.emotion_model_info = None
        self.emotion_dict = emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",  
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"} 
    
    def model_creation(self):
        self.emotion_model = Sequential() 
        self.emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',  
                                input_shape=(48, 48, 1))) 
        self.emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) 
        self.emotion_model.add(MaxPooling2D(pool_size=(2, 2))) 
        self.emotion_model.add(Dropout(0.25)) 

        self.emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu')) 
        self.emotion_model.add(MaxPooling2D(pool_size=(2, 2))) 
        self.emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu')) 
        self.emotion_model.add(MaxPooling2D(pool_size=(2, 2))) 
        self.emotion_model.add(Dropout(0.25)) 
        
        self.emotion_model.add(Flatten()) 
        self.emotion_model.add(Dense(1024, activation='relu')) 
        self.emotion_model.add(Dropout(0.5)) 
        self.emotion_model.add(Dense(7, activation='softmax')) 
        
        self.emotion_model.summary() 
        
        cv2.ocl.setUseOpenCL(False) 
        
        initial_learning_rate = 0.0001
        lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000,  
                                    decay_rate=0.96) 
        
        optimizer = Adam(learning_rate=lr_schedule) 
        
        self.emotion_model.compile(loss='categorical_crossentropy', optimizer=optimizer,  
                            metrics=['accuracy'])
    
    def train_model(self):
        self.emotion_model_info = self.emotion_model.fit( 
        self.train_generator, 
        steps_per_epoch=28709 // 64, 
        epochs=30, 
        validation_data=self.validation_generator, 
        validation_steps=7178 // 64)
        self.emotion_model.evaluate(self.validation_generator)
        
    def visualization(self):
        accuracy = self.emotion_model_info.history['accuracy'] 
        val_accuracy = self.emotion_model_info.history['val_accuracy'] 
        loss = self.emotion_model_info.history['loss'] 
        val_loss = self.emotion_model_info.history['val_loss']
        # Accuracy graph 
        plt.subplot(1, 2, 1) 
        plt.plot(accuracy, label='accuracy') 
        plt.plot(val_accuracy, label='val accuracy') 
        plt.title('Accuracy Graph') 
        plt.xlabel('Epochs') 
        plt.ylabel('Accuracy') 
        plt.legend() 
        
        # loss graph 
        plt.subplot(1, 2, 2) 
        plt.plot(loss, label='loss') 
        plt.plot(val_loss, label='val loss') 
        plt.title('Loss Graph') 
        plt.xlabel('Epochs') 
        plt.ylabel('Loss') 
        plt.legend() 
        
        plt.show() 
    
    
    def save_model(self):
        model_json = self.emotion_model.to_json() 
        with open("emotion_model.json", "w") as json_file: 
            json_file.write(model_json) 
        
    
    def validation(self):
        cap = cv2.VideoCapture(0) 
        while True: 
            # Find haar cascade to draw bounding box around face 
            ret, frame = cap.read() 
            frame = cv2.resize(frame, (1280, 720)) 
            if not ret: 
                print(ret) 
            # Create a face detector 
            face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
        
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
            # detect faces available on camera 
            num_faces = face_detector.detectMultiScale(gray_frame,  
                                                    scaleFactor=1.3, minNeighbors=5) 
        
            # take each face available on the camera and Preprocess it 
            for (x, y, w, h) in num_faces: 
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4) 
                roi_gray_frame = gray_frame[y:y + h, x:x + w] 
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame,  
                                                                        (48, 48)), -1), 0) 
                # predict the emotions 
                emotion_prediction = self.emotion_model.predict(cropped_img) 
                maxindex = int(np.argmax(emotion_prediction)) 
                cv2.putText(frame, self.emotion_dict[maxindex], (x+5, y-20),  
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) 
        
            cv2.imshow('Emotion Detection', frame) 
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        
        cap.release() 
        cv2.destroyAllWindows()
        
        # save trained model weight in .h5 file 
        self.emotion_model.save_weights('emotion_model.h5')
                
        
if __name__ == "__main__":
        detector = Detector("./archive/train", "./archive/test")
        detector.model_creation()
        detector.train_model()
        detector.visualization()
        detector.save_model()
        detector.validation()