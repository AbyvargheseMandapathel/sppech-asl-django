import cv2
import os
import numpy as np
from django.shortcuts import render, redirect
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

image_folder = 'images'  # Path to the folder where training images will be saved
model_path = 'trained_model.h5'  # Path to save the trained model file

from django.http import JsonResponse

def save_recorded_video(request):
    if request.method == 'POST':
        # Retrieve the title and video file from the form data
        text_label = request.POST.get('title')
        video_file = request.FILES.get('video')

        # Save the video file with the provided text label
        video_path = os.path.join(image_folder, f'{text_label}.webm')
        with open(video_path, 'wb') as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        # Perform any additional processing or saving logic here
        # ...

        return JsonResponse({'message': 'Video saved successfully.'})

    return JsonResponse({'message': 'Invalid request.'})

def home(request):
    return render(request, 'home.html')

def record_data(request):
    # Create the image folder if it doesn't exist
    os.makedirs(image_folder, exist_ok=True)

    # Check if the user submitted the form with a title
    if request.method == 'POST':
        text_label = request.POST.get('title')

        # Perform recording logic here
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend

        # Check if video capture is successful
        if not cap.isOpened():
            error_message = 'Failed to open video capture.'
            return render(request, 'record.html', {'error_message': error_message})

        # Load the hand detection model
        hand_cascade = cv2.CascadeClassifier('static/haar_cascades/hand_cascade.xml')

        # Load the trained model
        model = load_model(model_path)

        recording = True  # Flag to control the recording loop

        while recording:
            ret, frame = cap.read()

            # Check if frame capture is successful
            if not ret:
                error_message = 'Failed to capture frame.'
                return render(request, 'record.html', {'error_message': error_message})

            # Convert the frame to grayscale for hand detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Perform hand detection
            hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around the detected hands
            for (x, y, w, h) in hands:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Preprocess the hand region for prediction
                hand_roi = gray[y:y+h, x:x+w]
                hand_roi = cv2.resize(hand_roi, (28, 28))
                hand_roi = hand_roi.reshape(1, 784) / 255.0

                # Perform prediction using the trained model
                predicted_label = model.predict(hand_roi)
                predicted_class = np.argmax(predicted_label)

                # Display the predicted class as text
                cv2.putText(frame, str(predicted_class), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display the captured frame with hand detection and predictions
            cv2.imshow('Recording', frame)

            # Press 'q' to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                recording = False

            # Press 's' to save the current frame as training data
            if cv2.waitKey(1) & 0xFF == ord('s'):
                # Save the frame as an image with the provided text label
                image_path = os.path.join(image_folder, f'{text_label}.jpg')
                cv2.imwrite(image_path, frame)

        # Release the capture and close the window
        cap.release()
        cv2.destroyAllWindows()

        return redirect('train')

    return render(request, 'record.html', {'error_message': ''})

def train_data(request):
    # Check if the user submitted the form to start training
    if request.method == 'POST':
        # Prepare the training data
        images = []
        labels = []

        for filename in os.listdir(image_folder):
            if filename.endswith('.jpg'):
                image_path = os.path.join(image_folder, filename)
                image = cv2.imread(image_path)
                images.append(image)

                # Extract the text label from the filename
                label = os.path.splitext(filename)[0]
                labels.append(label)

        # Convert the images and labels to NumPy arrays
        images = np.array(images)
        labels = np.array(labels)

        # Perform training data logic here
        # ...

        # Create a simple model for demonstration purposes
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=784))
        model.add(Dense(10, activation='softmax'))

        # Compile and train the model
        model.compile(optimizer=Adam(), loss='categorical_crossentropy')
        # ...

        # Save the trained model
        model.save(model_path)

        return redirect('home')

    return render(request, 'train.html')
