import cv2
import face_recognition

# Load known images
known_image_1 = face_recognition.load_image_file("Mark.jpg")
known_image_2 = face_recognition.load_image_file("Mask.jpg")

# Encode the loaded images
known_encoding_1 = face_recognition.face_encodings(known_image_1)[0]
known_encoding_2 = face_recognition.face_encodings(known_image_2)[0]

# Store encodings and their names
known_encodings = [known_encoding_1, known_encoding_2]
known_names = ["Mark", "ElonMask"]

# Load the image you want to check
image_to_check = face_recognition.load_image_file("markandmask.jpg")

# Find all face locations and their encodings in the image
face_locations = face_recognition.face_locations(image_to_check)
face_encodings = face_recognition.face_encodings(image_to_check, face_locations)

# Convert image to BGR color for OpenCV
image_to_check = cv2.cvtColor(image_to_check, cv2.COLOR_RGB2BGR)

# Iterate through each detected face
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # Compare the detected face encoding with the known encodings
    matches = face_recognition.compare_faces(known_encodings, face_encoding)

    name = "Unknown"

    # Find the known face with the smallest distance
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_match_index = face_distances.argmin()

    if matches[best_match_index]:
        name = known_names[best_match_index]

    # Draw a rectangle around the face
    cv2.rectangle(image_to_check, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with a name below the face
    cv2.rectangle(image_to_check, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    cv2.putText(image_to_check, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

# Display the image with the detected and recognized faces
cv2.imshow("Image", image_to_check)
cv2.waitKey(0)
cv2.destroyAllWindows()

