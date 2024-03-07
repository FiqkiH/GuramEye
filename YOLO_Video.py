from ultralytics import YOLO
import cv2


def preprocess_frame(frame):
    # Resize the frame to 640x640
    resized_frame = cv2.resize(frame, (640, 640))
    return resized_frame


def video_detection(path_x):
    video_capture = path_x
    # Webcam Object
    cap = cv2.VideoCapture(video_capture)

    # load model
    model = YOLO("static/best_gurameye.pt")

    while True:
        success, img = cap.read()
        if not success:
            break

        object_count = 0

        # preprocess
        img = preprocess_frame(img)

        # prediksi
        results = model(img, stream=True)

        # menggambarkan hasil prediksi(bounding box dan nilai)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # hanya deteksi objek dengan conf score > 4
                if box.conf[0]*100/100 > 0.4:
                    x1, y1, x2, y2 = box.xyxy[0]  # mendapatkan nilai koordinat
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # mengubah tensor ke int
                    # print(x1, y1, x2, y2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)  # menggambarkan bbox

                    object_count += 1
        count_text = f'Jumlah: {object_count}'
        cv2.putText(img, count_text, (img.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2, cv2.LINE_AA)

        if object_count > 1:
            # Draw a small red square with "OVERCROWD!"
            cv2.rectangle(img, (10, 10), (230, 60), (0, 0, 255), -1)
            cv2.putText(img, "OVERCROWD!", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        yield img


cv2.destroyAllWindows()
