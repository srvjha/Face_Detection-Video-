import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold the image to make the hand white and the background black
    _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find the contour with the largest area, which should be the hand
    if len(contours) > 0:
        hand_contour = max(contours, key=cv2.contourArea)

        # create a bounding box around the hand contour
        x, y, w, h = cv2.boundingRect(hand_contour)

        # select the region of interest (ROI) corresponding to the hand
        roi = frame[y:y+h, x:x+w]

        # check if the ROI has a valid width and height
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            # display the ROI with a rectangle around it
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('Hand Detection', roi)

    cv2.imshow('Video Feed', frame)

    # exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
