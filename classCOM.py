import mediapipe as mp
import streamlit as st
import tempfile
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Clase encargada de la detección del centro de masa
class CenterOfMassDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    # Función para calcular el centro de un segmento
    def segment_center(self, point1, point2):
        return [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2, (point1[2] + point2[2]) / 2]

    # Función para calcular el centro de masa (CM)
    def calculate_center_of_mass(self, landmarks, peso_persona):
        # Relación de masas por segmentos del cuerpo (valores aproximados)
        segment_mass_ratios = {
            'head': 0.08,
            'torso': 0.5,
            'upper_arm': 0.03,
            'lower_arm': 0.02,
            'thigh': 0.1,
            'lower_leg': 0.05
        }

        # Extraer las coordenadas relevantes de los landmarks
        head_center = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
                       landmarks[self.mp_pose.PoseLandmark.NOSE.value].y,
                       landmarks[self.mp_pose.PoseLandmark.NOSE.value].z]

        torso_center = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].z]

        upper_arm_center = self.segment_center(
            [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z],
            [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
        )

        thigh_center = self.segment_center(
            [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].z],
            [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
        )

        # Calcular el eje X como el punto medio entre las dos caderas
        hip_right = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].z]

        hip_left = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].z]

        cm_x = (hip_right[0] + hip_left[0]) / 2

        # Centros y masas de los segmentos corporales
        segment_centers = [
            (head_center, segment_mass_ratios['head'] * peso_persona),
            (torso_center, segment_mass_ratios['torso'] * peso_persona),
            (upper_arm_center, segment_mass_ratios['upper_arm'] * peso_persona),
            (thigh_center, segment_mass_ratios['thigh'] * peso_persona)
        ]

        # Calcular el centro de masa en el eje Y y Z, manteniendo el nuevo eje X
        total_mass = sum(mass for _, mass in segment_centers)
        cm_y = sum(center[1] * mass for center, mass in segment_centers) / total_mass
        cm_z = sum(center[2] * mass for center, mass in segment_centers) / total_mass

        return cm_x, cm_y, cm_z

    # Procesa el video y detecta el esqueleto junto con el centro de masa
    def process_video(self, uploaded_file, peso_persona):
        # Crear un archivo temporal para almacenar el video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Abre el video usando av
        video = av.open(tfile.name)
        stframe = st.empty()

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            for frame in video.decode(video=0):
                # Convertir el frame a un array de numpy
                image = np.array(frame.to_image())

                # Procesa la imagen para detectar poses
                results = pose.process(image)

                # Dibujar el esqueleto y el centro de masa en la imagen
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Dibujar el esqueleto en la imagen
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                    # Calcular el centro de masa
                    cm_x, cm_y, cm_z = self.calculate_center_of_mass(landmarks, peso_persona)

                    # Dibujar el centro de masa en la imagen
                    height, width, _ = image.shape
                    cm_x_px = int(cm_x * width)
                    cm_y_px = int(cm_y * height)

                    # Convertir la imagen a PIL para dibujar
                    img_pil = Image.fromarray(image)
                    draw = ImageDraw.Draw(img_pil)
                    draw.ellipse((cm_x_px - 5, cm_y_px - 5, cm_x_px + 5, cm_y_px + 5), fill=(255, 0, 0))

                    # Dibujar las coordenadas X, Y, Z junto al punto rojo
                    text = f"X: {cm_x:.2f}, Y: {cm_y:.2f}, Z: {cm_z:.2f}"
                    draw.text((cm_x_px + 10, cm_y_px - 10), text, fill=(255, 255, 255))

                    # Mostrar la imagen con esqueleto, centro de masa y coordenadas
                    stframe.image(img_pil, use_column_width=True)




