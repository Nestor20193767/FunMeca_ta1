from classCOM import CenterOfMassDetector
from classFrontend import AppFrontend

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

