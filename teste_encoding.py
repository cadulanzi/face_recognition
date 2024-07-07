import dlib
import numpy as np
import os
from PIL import Image
import cv2

# Carregar modelos do dlib
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

TOLERANCE = 0.6

# Função para obter encodings faciais a partir de uma imagem
def get_face_encodings(image):
    detected_faces = face_detector(image, 1)
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]

# Função para comparar encodings faciais
def compare_face_encodings(known_faces, face):
    return (np.linalg.norm(known_faces - face, axis=1) <= TOLERANCE)

# Função para encontrar a correspondência de um rosto
def find_match(known_faces, names, face):
    matches = compare_face_encodings(known_faces, face)
    count = 0
    for match in matches:
        if match:
            return names[count]
        count += 1
    return 'Not Found'

# Carregar imagens conhecidas
def load_known_faces(image_dir):
    known_face_encodings = []
    known_face_names = []

    for file_name in os.listdir(image_dir):
        if file_name.endswith(('jpg', 'jpeg', 'png')):
            image_path = os.path.join(image_dir, file_name)
            print(f"Processing file: {image_path}")

            try:
                # Carregar e converter a imagem para RGB usando PIL
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    img_array = np.array(img)

                encodings = get_face_encodings(img_array)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(file_name.split('.')[0])
                else:
                    print(f"No face encodings found in: {image_path}")

            except Exception as e:
                print(f"Error processing file {image_path}: {e}")
                continue

    return known_face_encodings, known_face_names

# Reconhecer rostos em uma imagem de teste
def recognize_faces_in_image(image_path, known_face_encodings, known_face_names):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img_array = np.array(img)
        
        face_locations = face_detector(img_array, 1)
        face_encodings = get_face_encodings(img_array)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = find_match(known_face_encodings, known_face_names, face_encoding)
            
            # Desenhar uma caixa ao redor do rosto
            cv2.rectangle(img_array, (left, top), (right, bottom), (0, 0, 255), 2)
            # Desenhar o nome abaixo do rosto
            cv2.rectangle(img_array, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img_array, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        # Converter de volta para BGR para exibir com OpenCV
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cv2.imshow('Image', img_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error processing test image {image_path}: {e}")

# Carregar imagens conhecidas
image_dir = 'db/wanted_img'
known_face_encodings, known_face_names = load_known_faces(image_dir)
print(f"Learned encoding for {len(known_face_encodings)} images.")

# Testar reconhecimento facial em uma nova imagem
test_image_path = 'db/wanted_img/obama.jpg'  # Substitua pelo caminho da imagem de teste
recognize_faces_in_image(test_image_path, known_face_encodings, known_face_names)
