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

class AppFrontend:
    def __init__(self, detector):
        self.detector = detector
        self.st = st

    def run_page_1(self):
        self.st.title("Detección del Centro de Masa")
        uploaded_file = self.st.file_uploader("Sube un video", type=['mp4', 'mov', 'avi'])
        if uploaded_file is not None:
            peso_persona = self.st.number_input("Ingresa el peso de la persona (kg):", min_value=0.0, step=0.1)
            if peso_persona > 0:
                self.st.text("Procesando video...")
                self.detector.process_video(uploaded_file, peso_persona)
                self.st.text("Procesamiento completado.")

    def run_page_2(self):
        self.st.title("Cálculo del Centro de Masa")
        self.st.header("Descripción del Cálculo del Centro de Masa")

        # Modelo Fisiológico: Descripción
        self.st.write("""
        Para calcular el centro de masa de una persona, hemos utilizado el **modelo segmental del cuerpo humano**, 
        basado en el trabajo de **Dempster (1955)**. Este modelo asigna una proporción de la masa total del cuerpo 
        a distintos segmentos, como la cabeza, el torso, los brazos y las piernas. Estas proporciones son aproximadas 
        y se basan en estudios biomecánicos.
        """)

        # Fórmula del Centro de Masa en LaTeX
        self.st.subheader("Fórmula del Centro de Masa")
        self.st.latex(r"""
        CM = \frac{\sum_{i=1}^{n} m_i \cdot r_i}{\sum_{i=1}^{n} m_i}
        """)
        self.st.write("""
        Donde:
        - \( CM \) es la coordenada del centro de masa.
        - \( m_i \) es la masa del segmento corporal \( i \).
        - \( r_i \) es la posición (coordenada) del centro del segmento corporal \( i \).
        - La suma se realiza sobre todos los segmentos del cuerpo.
        """)

        # Proporciones de Masa por Segmento
        self.st.subheader("Proporciones de Masa por Segmento Corporal")
        self.st.write("""
        A continuación, mostramos las proporciones de masa de los diferentes segmentos del cuerpo, 
        que se utilizan para calcular el centro de masa ponderado:
        """)
        self.st.write("""
        - **Cabeza**: 8% de la masa total.
        - **Torso**: 50% de la masa total.
        - **Brazo superior**: 3% de la masa total (por cada brazo).
        - **Antebrazo**: 2% de la masa total (por cada antebrazo).
        - **Muslo**: 10% de la masa total (por cada muslo).
        - **Pierna inferior**: 5% de la masa total (por cada pierna).
        """)

        # Ilustración del Cálculo
        self.st.write("""
        El cálculo se realiza considerando las coordenadas 3D de los puntos clave del esqueleto, como la nariz, caderas, hombros, rodillas, etc. 
        Se calcula la posición media de los segmentos corporales y se ponderan estas posiciones con las masas correspondientes.
        """)

        # Paper utilizado
        self.st.write("""
        El paper que se uso para poder calcular las porciones segmentales del cuerpo humano es: 
        \n
        [1]   https://www.sciencedirect.com/science/article/pii/0021929094900353
        \n
        [2]   https://deepblue.lib.umich.edu/bitstream/handle/2027.42/4540/bab9715.0001.001.pdf?sequence=5&isAllowed=y
        """)

    def run_page_n(self):
        self.st.title("Tarea academica 1")
        self.st.write("""
        Este programada fue creado para la Tarea academica 1 del curos de Fundamento mecanico de los biomateriales
        dictado en la Pontifica Universidad Catolica del Perú.
        """)

def main():
    # Instancia la clase detectora de centro de masa
    detector = CenterOfMassDetector()

    # Instancia la clase de frontend
    frontend = AppFrontend(detector)

    # Crea una barra lateral con un selector de páginas
    page = frontend.st.sidebar.selectbox(
        "Selecciona la página",
        ("Detección del Centro de Masa", "Cálculo del Centro de Masa", "Tarea academica 1")
    )

    # Ejecutar diferentes páginas
    if page == "Detección del Centro de Masa":
        frontend.run_page_1()
    elif page == "Cálculo del Centro de Masa":
        frontend.run_page_2()
    elif page == "Tarea academica 1":
        frontend.run_page_n()

if __name__ == "__main__":
    main()