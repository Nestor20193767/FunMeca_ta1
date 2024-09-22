import mediapipe as mp
import streamlit as st
import tempfile
import av
import numpy as np
import cv2
import os

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

    # Procesa el video en tiempo real desde la cámara
    def process_camera(self, peso_persona):
        cap = cv2.VideoCapture(0)  # Abrir la cámara
        with self.mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.75) as pose:
            while cap.isOpened():
                success, image_np = cap.read()  # Leer frame de la cámara
                if not success:
                    print("No se pudo acceder a la cámara.")
                    break

                # Procesa la imagen para detectar poses
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                results = pose.process(image_np)

                # Dibujar el esqueleto y el centro de masa en la imagen
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Dibujar el esqueleto en la imagen
                    self.mp_drawing.draw_landmarks(image_np, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                    # Calcular el centro de masa
                    cm_x, cm_y, cm_z = self.calculate_center_of_mass(landmarks, peso_persona)

                    # Dibujar el centro de masa en la imagen
                    height, width, _ = image_np.shape
                    cm_x_px = int(cm_x * width)
                    cm_y_px = int(cm_y * height)

                    # Dibujar un círculo rojo en el centro de masa
                    cv2.circle(image_np, (cm_x_px, cm_y_px), 5, (255, 0, 0), -1)

                    # Dibujar las coordenadas X, Y, Z junto al punto rojo
                    cv2.putText(image_np, f"X: {cm_x:.2f}, Y: {cm_y:.2f}, Z: {cm_z:.2f}",
                                (cm_x_px + 10, cm_y_px - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Mostrar el frame procesado
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                cv2.imshow('Centro de Masa en Tiempo Real', image_np)

                if cv2.waitKey(1) & 0xFF == ord('q'):  # Presionar 'q' para salir
                    break

        cap.release()
        cv2.destroyAllWindows()

class AppFrontend:
    def __init__(self, detector):
        self.detector = detector
        self.st = st

    def run_page_1(self):
        self.st.title("Detección del Centro de Masa - Video Subido")
        uploaded_file = self.st.file_uploader("Sube un video", type=['mp4', 'mov', 'avi'])
        if uploaded_file is not None:
            peso_persona = self.st.number_input("Ingresa el peso de la persona (kg):", min_value=0.0, step=0.1)
            if peso_persona > 0:
                if self.st.button("Procesar Video"):
                    with st.spinner("Procesando video..."):
                        processed_video_path = self.detector.process_video(uploaded_file, peso_persona)
                    self.st.success("Procesamiento completado.")

                    # Leer el video procesado
                    with open(processed_video_path, 'rb') as video_file:
                        video_bytes = video_file.read()

                    # Mostrar el video procesado
                    self.st.video(video_bytes)

                    # Agregar botón de descarga
                    self.st.download_button(
                        label="Descargar Video Procesado",
                        data=video_bytes,
                        file_name="video_procesado.mp4",
                        mime="video/mp4"
                    )

                    # Eliminar el archivo de video procesado después de usarlo
                    os.remove(processed_video_path)

    def run_page_real_time(self):
        self.st.title("Detección del Centro de Masa en Tiempo Real")
        peso_persona = self.st.number_input("Ingresa el peso de la persona (kg):", min_value=0.0, step=0.1)
        if peso_persona > 0:
            if self.st.button("Iniciar Detección en Tiempo Real"):
                self.detector.process_camera(peso_persona)

def main():
    # Instancia la clase detectora de centro de masa
    detector = CenterOfMassDetector()

    # Instancia la clase de frontend
    frontend = AppFrontend(detector)

    # Crea una barra lateral con un selector de páginas
    page = frontend.st.sidebar.selectbox(
        "Selecciona la página",
        ("Detección del Centro de Masa - Video Subido", "Detección del Centro de Masa en Tiempo Real")
    )

    # Ejecutar diferentes páginas
    if page == "Detección del Centro de Masa - Video Subido":
        frontend.run_page_1()
    elif page == "Detección del Centro de Masa en Tiempo Real":
        frontend.run_page_real_time()

if __name__ == "__main__":
    main()

