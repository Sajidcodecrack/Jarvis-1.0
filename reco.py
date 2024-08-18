import cv2
import pyttsx3

# Initialize the recognizer and text-to-speech engine
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
recognizer = cv2.face.LBPHFaceRecognizer_create()
engine = pyttsx3.init()

# Load the training data (your face)
recognizer.read("trainer.yml")


# Define a function to greet the user
def greet_user(greeting):
    engine.say(greeting)
    engine.runAndWait()


# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, w, h in faces:
        roi_gray = gray[y : y + h, x : x + w]
        id_, confidence = recognizer.predict(roi_gray)

        if confidence < 70:  # Adjust this threshold as needed
            greet_user("Welcome back, sajid sir!")
        # elif confidence >=50:
        #     greet_user("Goodbye, sajid sir have a good day!")
        else:
            greet_user("Too much noise move to a quiet place ")

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
