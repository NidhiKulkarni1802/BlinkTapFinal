import cv2

# Start the webcam
camera = cv2.VideoCapture(0)

# Check whether the webcam opened successfully
if not camera.isOpened():
    print("ERROR: Could not access the webcam.")
    exit()

print("Webcam started successfully.")
print("Press Q to quit.")

while True:
    # Capture a frame
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read frame.")
        break

    # Display the webcam frame
    cv2.imshow("BlinkTap - Camera Test", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam
camera.release()
cv2.destroyAllWindows()

print("Webcam test completed.")