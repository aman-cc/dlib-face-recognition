from face_encoding_script import face_encodings
import cv2

if __name__ == '__main__':
    img_path = 'face.jpg'
    img = cv2.imread(img_path)
    face_loc = (0, img.shape[1], img.shape[0], 0)

    # encoding = face_encodings(img)
    encoding_full_frame = face_encodings(img, [face_loc])
    print(f"Encoding shape is {encoding_full_frame[0].shape}")