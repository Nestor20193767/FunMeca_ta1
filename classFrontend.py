import streamlit as st

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
