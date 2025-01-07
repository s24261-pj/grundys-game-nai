import cv2 as cv
from ultralytics import YOLO

model = YOLO("./runs/weights/best.pt")

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()

    results = model.predict(frame, conf=0.5)
    annotated_frame = results[0].plot()
    cv.imshow("Detected Frame", annotated_frame)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
