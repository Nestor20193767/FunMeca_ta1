import mediapipe as mp
import streamlit as st
import tempfile
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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

    # Procesa el video y detecta el esqueleto junto con el centro de masa
    def process_video(self, uploaded_file, peso_persona):
        # Crear archivos temporales para el video de entrada y salida
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile_in.write(uploaded_file.read())
        tfile_in.close()

        tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile_out.close()

        # Abre el video de entrada usando av
        input_container = av.open(tfile_in.name)
        # Prepara el contenedor de salida usando av
        output_container = av.open(tfile_out.name, mode='w')

        # Obtén el stream de video y su configuración
        input_stream = input_container.streams.video[0]
        codec_name = input_stream.codec_context.name

        # Configura el stream de salida
        output_stream = output_container.add_stream(codec_name, rate=input_stream.average_rate)
        output_stream.width = input_stream.width
        output_stream.height = input_stream.height
        output_stream.pix_fmt = 'yuv420p'

        with self.mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.85) as pose:
            for frame in input_container.decode(video=0):
                # Convertir el frame a un array de numpy
                image = frame.to_image()
                image_np = np.array(image)

                # Procesa la imagen para detectar poses
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

                # Convertir el frame procesado a AV frame
                processed_frame = av.VideoFrame.from_ndarray(image_np, format='rgb24')
                for packet in output_stream.encode(processed_frame):
                    output_container.mux(packet)

            # Finalizar el stream de salida
            for packet in output_stream.encode():
                output_container.mux(packet)

            # Cerrar los contenedores
            output_container.close()
            input_container.close()

        # Eliminar el archivo de entrada temporal
        os.remove(tfile_in.name)

        return tfile_out.name

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
        El paper que se usó para calcular las porciones segmentales del cuerpo humano es: 
        \n
        [1]   https://www.sciencedirect.com/science/article/pii/0021929094900353
        \n
        [2]   https://deepblue.lib.umich.edu/bitstream/handle/2027.42/4540/bab9715.0001.001.pdf?sequence=5&isAllowed=y
        \n
        [3]   https://www.youtube.com/watch?v=OJh9bdrOLQw
        """)
        

    def run_page_n(self):
        self.st.title("Manual de Uso")
        self.st.write("""
        Bienvenido al manual de uso de la aplicación de detección del centro de masa, diseñada para el curso de 
        **Fundamento Mecánico de los Biomateriales** de la **Pontificia Universidad Católica del Perú**.
        """)
        
        self.st.write("""
        ### ¿Cómo funciona la aplicación?
        
        1. **Subir un video**: En la página principal "Detección del Centro de Masa", puedes subir un video en formatos 
        compatibles como MP4, MOV o AVI. Asegúrate de que el video muestre claramente el cuerpo de la persona cuyas 
        coordenadas serán analizadas.
        
        2. **Ingresar el peso**: Después de subir el video, deberás ingresar el peso de la persona. Este valor se usa 
        para calcular de manera precisa el centro de masa, basándonos en un modelo segmental del cuerpo humano.
    
        3. **Procesar video**: Al hacer clic en el botón "Procesar Video", la aplicación detecta el esqueleto de la 
        persona en cada cuadro del video y calcula el centro de masa. La detección se realiza utilizando la 
        biblioteca **Mediapipe**, que genera las coordenadas 3D de los puntos clave del cuerpo (cabeza, torso, brazos, 
        piernas, etc.).
        
        4. **Resultado visual**: Una vez procesado, podrás ver el video con el esqueleto y el centro de masa dibujados 
        en cada cuadro. El centro de masa se representa como un punto rojo en el video, acompañado de sus coordenadas.
    
        5. **Descargar el video procesado**: Finalmente, puedes descargar el video procesado directamente a tu dispositivo 
        para futuras referencias o análisis.
    
        ### ¿Para qué se utiliza?
        
        Esta aplicación ha sido creada con fines académicos y de investigación, permitiendo a los estudiantes y 
        profesionales visualizar el comportamiento dinámico del centro de masa en diversas actividades físicas. 
        Está diseñada como parte del curso de Fundamento Mecánico de los Biomateriales de la PUCP.
        """)


def main():
    # Instancia la clase detectora de centro de masa
    detector = CenterOfMassDetector()

    # Instancia la clase de frontend
    frontend = AppFrontend(detector)

    # Crea una barra lateral con un selector de páginas
    page = frontend.st.sidebar.selectbox(
        "Selecciona la página",
        ("Detección del Centro de Masa", "Cálculo del Centro de Masa", "Manual de Uso")
    )

    # Ejecutar diferentes páginas
    if page == "Detección del Centro de Masa":
        frontend.run_page_1()
    elif page == "Cálculo del Centro de Masa":
        frontend.run_page_2()
    elif page == "Manual de Uso":
        frontend.run_page_n()

if __name__ == "__main__":
    main()

