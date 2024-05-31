import cv2
import os

# this method will resize all images to 224x224 pixels
# also it uses a face detector to center the image around the first face it detects
def resize_and_center_face(folder, target_size=(224, 224)):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            # get  first face (assuming there's only one face per image)
            x, y, w, h = faces[0]

            # calculate center of the face
            center_x = x + w // 2
            center_y = y + h // 2

            # calculate crop boundaries to center the face
            crop_x = max(0, center_x - target_size[0] // 2)
            crop_y = max(0, center_y - target_size[1] // 2)
            crop_x_end = min(img.shape[1], crop_x + target_size[0])
            crop_y_end = min(img.shape[0], crop_y + target_size[1])

            # crop and resize the image
            cropped_img = img[crop_y:crop_y_end, crop_x:crop_x_end]
            resized_img = cv2.resize(cropped_img, target_size)

            # cave the image
            cv2.imwrite(filepath, resized_img)


if __name__ == "__main__":
    folder = "us"
    resize_and_center_face(folder)
