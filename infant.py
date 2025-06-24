
import numpy as np
import cv2
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import dlib
import librosa
import warnings
warnings.filterwarnings('ignore')

class InfantEmotionDetector:
    def __init__(self):
        """
        Initialize the Infant Emotion Detection System using SVM
        """
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = None  # Load with dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.svm_model = None
        self.emotion_labels = ['calm', 'happy', 'sad', 'angry', 'surprised', 'fear']
        
    def extract_facial_features(self, image):
        """
        Extract facial landmark features from infant face
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_detector(gray)
            
            if len(faces) == 0:
                return None
                
            # Get the largest face (assuming it's the infant)
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Extract facial landmarks (requires dlib predictor model)
            if self.landmark_predictor is not None:
                landmarks = self.landmark_predictor(gray, face)
                
                # Convert landmarks to numpy array
                points = np.array([[p.x, p.y] for p in landmarks.parts()])
                
                # Calculate geometric features
                features = self._calculate_geometric_features(points)
                
            else:
                # Basic geometric features from face rectangle
                features = self._basic_face_features(face, gray.shape)
                
            return features
            
        except Exception as e:
            print(f"Error in facial feature extraction: {e}")
            return None
    
    def _calculate_geometric_features(self, landmarks):
        """
        Calculate geometric features from facial landmarks
        """
        features = []
        
        # Eye aspect ratio (EAR) - indicates eye openness
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        features.extend([left_ear, right_ear, (left_ear + right_ear) / 2])
        
        # Mouth aspect ratio (MAR) - indicates mouth openness
        mouth = landmarks[48:68]
        mar = self._mouth_aspect_ratio(mouth)
        features.append(mar)
        
        # Eyebrow position relative to eyes
        left_eyebrow = landmarks[17:22]
        right_eyebrow = landmarks[22:27]
        
        left_brow_height = np.mean(left_eyebrow[:, 1]) - np.mean(left_eye[:, 1])
        right_brow_height = np.mean(right_eyebrow[:, 1]) - np.mean(right_eye[:, 1])
        features.extend([left_brow_height, right_brow_height])
        
        # Mouth curvature (smile detection)
        mouth_corners = [landmarks[48], landmarks[54]]  # Left and right mouth corners
        mouth_center = landmarks[51]  # Upper lip center
        
        curvature = (mouth_corners[0][1] + mouth_corners[1][1]) / 2 - mouth_center[1]
        features.append(curvature)
        
        # Face symmetry
        face_center_x = np.mean(landmarks[:, 0])
        left_side = landmarks[landmarks[:, 0] < face_center_x]
        right_side = landmarks[landmarks[:, 0] >= face_center_x]
        
        symmetry = np.abs(np.mean(left_side[:, 0]) - np.mean(right_side[:, 0]))
        features.append(symmetry)
        
        return np.array(features)
    
    def _basic_face_features(self, face_rect, image_shape):
        """
        Extract basic features when landmark detection is not available
        """
        features = []
        
        # Face dimensions relative to image
        face_width = face_rect.width() / image_shape[1]
        face_height = face_rect.height() / image_shape[0]
        face_area = face_width * face_height
        
        # Face position
        face_center_x = (face_rect.left() + face_rect.width()/2) / image_shape[1]
        face_center_y = (face_rect.top() + face_rect.height()/2) / image_shape[0]
        
        features.extend([face_width, face_height, face_area, face_center_x, face_center_y])
        
        return np.array(features)
    
    def _eye_aspect_ratio(self, eye_points):
        """
        Calculate eye aspect ratio
        """
        # Vertical eye distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal eye distance
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def _mouth_aspect_ratio(self, mouth_points):
        """
        Calculate mouth aspect ratio
        """
        # Vertical mouth distances
        A = np.linalg.norm(mouth_points[2] - mouth_points[10])  # 50-58
        B = np.linalg.norm(mouth_points[4] - mouth_points[8])   # 52-56
        
        # Horizontal mouth distance
        C = np.linalg.norm(mouth_points[0] - mouth_points[6])   # 48-54
        
        # Mouth aspect ratio
        mar = (A + B) / (2.0 * C)
        return mar
    
    def extract_audio_features(self, audio_file):
        """
        Extract audio features from infant vocalizations/crying
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_file, sr=22050)
            
            # Extract audio features
            features = []
            
            # Spectral features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))
            
            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            features.extend([pitch_mean, pitch_std])
            
            # Energy features
            rms = librosa.feature.rms(y=y)[0]
            features.extend([np.mean(rms), np.std(rms)])
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.extend([np.mean(zcr), np.std(zcr)])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error in audio feature extraction: {e}")
            return None
    
    def prepare_training_data(self, image_paths, audio_paths, labels):
        """
        Prepare training data from images, audio files, and labels
        """
        features_list = []
        valid_labels = []
        
        for i, (img_path, audio_path, label) in enumerate(zip(image_paths, audio_paths, labels)):
            try:
                # Extract visual features
                image = cv2.imread(img_path)
                if image is None:
                    continue
                    
                visual_features = self.extract_facial_features(image)
                if visual_features is None:
                    continue
                
                # Extract audio features (if available)
                audio_features = None
                if audio_path and audio_path != '':
                    audio_features = self.extract_audio_features(audio_path)
                
                # Combine features
                if audio_features is not None:
                    combined_features = np.concatenate([visual_features, audio_features])
                else:
                    combined_features = visual_features
                
                features_list.append(combined_features)
                valid_labels.append(label)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        return np.array(features_list), np.array(valid_labels)
    
    def train_model(self, X, y, test_size=0.2):
        """
        Train the SVM model for infant emotion detection
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Create pipeline with scaling and SVM
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(random_state=42))
        ])
        
        # Hyperparameter tuning
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__kernel': ['rbf', 'poly', 'linear'],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
        
        print("Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        self.svm_model = grid_search.best_estimator_
        
        # Evaluate model
        y_pred = self.svm_model.predict(X_test)
        
        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Best Cross-validation Score: {grid_search.best_score_:.4f}")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        
        # Detailed evaluation
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_
        ))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return self.svm_model
    
    def predict_emotion(self, image, audio_file=None):
        """
        Predict emotion from a single image and optional audio file
        """
        if self.svm_model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Extract features
        visual_features = self.extract_facial_features(image)
        if visual_features is None:
            return None, 0.0
        
        # Extract audio features if available
        audio_features = None
        if audio_file:
            audio_features = self.extract_audio_features(audio_file)
        
        # Combine features
        if audio_features is not None:
            features = np.concatenate([visual_features, audio_features])
        else:
            features = visual_features
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Predict
        prediction = self.svm_model.predict(features)[0]
        confidence = np.max(self.svm_model.predict_proba(features)[0])
        
        # Decode label
        emotion = self.label_encoder.inverse_transform([prediction])[0]
        
        return emotion, confidence
    
    def real_time_emotion_detection(self, camera_id=0):
        """
        Real-time emotion detection using webcam
        """
        if self.svm_model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        cap = cv2.VideoCapture(camera_id)
        
        print("Starting real-time emotion detection. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict emotion
            emotion, confidence = self.predict_emotion(frame)
            
            if emotion is not None:
                # Display result on frame
                text = f"Emotion: {emotion} ({confidence:.2f})"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Infant Emotion Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Example usage
def main():
    """
    Example usage of the Infant Emotion Detection System
    """
    # Initialize detector
    detector = InfantEmotionDetector()
    
    # Example training data (replace with actual data paths)
    image_paths = ['infant1.jpg', 'infant2.jpg', 'infant3.jpg']  # Add actual paths
    audio_paths = ['cry1.wav', '', 'laugh1.wav']  # Optional audio files
    labels = ['happy', 'sad', 'happy']  # Corresponding emotion labels
    
    # Prepare training data
    print("Preparing training data...")
    # X, y = detector.prepare_training_data(image_paths, audio_paths, labels)
    
    # For demonstration, create dummy data
    X = np.random.random((100, 15))  # 100 samples, 15 features
    y = np.random.choice(['calm', 'happy', 'sad', 'angry', ’fear’, ‘surprised’], 100)
    
    # Train model
    print("Training SVM model...")
    detector.train_model(X, y)
    
    # Example prediction
    # test_image = cv2.imread('test_infant.jpg')
    # emotion, confidence = detector.predict_emotion(test_image)
    # print(f"Predicted emotion: {emotion} with confidence: {confidence:.2f}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()


