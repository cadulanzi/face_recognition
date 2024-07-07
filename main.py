import requests
import boto3
import cv2
import os
import face_recognition
import numpy as np
from multiprocessing import Process, Queue, current_process
import csv
import subprocess
import logging
import sys

from config import Config

# Inicializar a configuração
config = Config()

# Defina os caminhos dos diretórios
base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
db_wanted_list = os.path.join(base_path, "db/wanted_list")
db_wanted_img = os.path.join(base_path, "db/wanted_img")
os.makedirs(db_wanted_img, exist_ok=True)
os.makedirs(db_wanted_list, exist_ok=True)

def fetch_data(base_url, endpoint, headers):
    url = f"{base_url}{endpoint}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch data from {url}")

def save_wanted_list(wanted_list, csv_file_path):
    if wanted_list:
        fieldnames = list(wanted_list[0].keys())
    else:
        fieldnames = ["name", "gender", "path", "nickname", "race", "age", "socialNumber", "notes", "birthDate", "id", "token", "url"]

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for wanted in wanted_list:
            writer.writerow(wanted)

def sync_s3(bucket, contract_id, destination, aws_access_key_id, aws_secret_access_key):
    cmd_sync = [
        "aws", "s3", "sync", f"s3://{bucket}/{contract_id}", destination
    ]

    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    env["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    result = subprocess.run(cmd_sync, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        logging.error(f"Error syncing S3 files: {result.stderr}")
    else:
        logging.info("Files synced successfully")

def fetch_wanted_list():
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': config.token_facial_api,
        'Authorization': f'Bearer {config.token_facial_api}'
    }
    return fetch_data(config.url_facial_api, "/v2/wanted", headers)

def download_images(wanted_list):
    csv_file_path = os.path.join(db_wanted_list, "wanted_list.csv")
    save_wanted_list(wanted_list, csv_file_path)

    for person in wanted_list:
        image_url = person['url']
        image_data = requests.get(image_url).content
        image_name = person['path'].split('/')[-1]
        image_path = os.path.join(db_wanted_img, image_name)
        with open(image_path, 'wb') as file:
            file.write(image_data)
        print(f"Downloaded and saved {image_name} to {db_wanted_img}")

def get_camera_frames(camera_info, frame_queue, use_webcam=False):
    if use_webcam:
        cap = cv2.VideoCapture(0)  # 0 is the default index for the built-in webcam
    else:
        user = camera_info['user']
        password = camera_info['password']
        ip = camera_info['ip']
        rtsp_url = f"rtsp://{user}:{password}@{ip}"
        cap = cv2.VideoCapture(rtsp_url)
    
    while True:
        ret, frame = cap.read()
        if ret:
            frame_queue.put(frame)
        else:
            break
    cap.release()


def recognize_faces(frame_queue, known_face_encodings, known_face_names):
    process_this_frame = True

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Only process every other frame of video to save time
            if process_this_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    for root, dirs, files in os.walk(db_wanted_img):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(root, file)
                image = face_recognition.load_image_file(image_path)
                rgb_small_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                encodings = face_recognition.face_encodings(rgb_small_frame)
                if encodings:  # Ensure encodings is not empty
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(file.split('.')[0])  # Use the file name as the person's name
    return known_face_encodings, known_face_names


def main():
    try:
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': config.token_facial_api
        }

        # wanted_list = fetch_wanted_list()
        # download_images(wanted_list)
        known_face_encodings, known_face_names = load_known_faces()

        # camera_list = fetch_data(config.url_facial_api, "/camera/list", headers)
        camera_list = []
        frame_queue = Queue()

        processes = []
        
        # Adicione uma câmera fictícia para a webcam
        webcam_info = {
            'user': '',
            'password': '',
            'ip': ''
        }
        
        # Adiciona a webcam à lista de processos de captura de câmera
        p = Process(target=get_camera_frames, args=(webcam_info, frame_queue, True))
        p.start()
        processes.append(p)

        # Processos para câmeras IP
        for camera_info in camera_list:
            p = Process(target=get_camera_frames, args=(camera_info, frame_queue))
            p.start()
            processes.append(p)

        for _ in range(len(camera_list) + 1):  # Inclui a webcam
            p = Process(target=recognize_faces, args=(frame_queue, known_face_encodings, known_face_names))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
