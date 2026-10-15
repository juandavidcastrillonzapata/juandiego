import os
import sys
import time
import threading

import cv2
import numpy as np
import face_recognition

ARCHIVO_ROSTRO = "mi_rostro.npy"
UMBRAL_DISTANCIA = 0.5
FRAMES_PARA_CONFIRMAR = 5
SEGUNDOS_MENU = 20

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")
NUMERO_A_LLAMAR = os.getenv("NUMERO_A_LLAMAR")

LUCES_WEBHOOK_URL = os.getenv("LUCES_WEBHOOK_URL", "")

alarma_activa = threading.Event()


def llamar_policia():
    print("[POLICIA] Solicitando ayuda...")
    if not (TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM and NUMERO_A_LLAMAR):
        print("[POLICIA] (simulado) Configura Twilio para hacer la llamada real.")
        return
    try:
        from twilio.rest import Client
        cliente = Client(TWILIO_SID, TWILIO_TOKEN)
        cliente.calls.create(
            to=NUMERO_A_LLAMAR,
            from_=TWILIO_FROM,
            twiml="<Response><Say language='es-MX'>Alerta de seguridad. "
                  "Se necesita ayuda en este domicilio.</Say></Response>",
        )
        print("[POLICIA] Llamada realizada.")
    except Exception as e:
        print(f"[POLICIA] Error al llamar: {e}")


def _sonar_alarma():
    try:
        import winsound
        while alarma_activa.is_set():
            winsound.Beep(2000, 400)
            winsound.Beep(1200, 400)
    except ImportError:
        while alarma_activa.is_set():
            print("\a", end="", flush=True)
            time.sleep(0.5)


def alternar_alarma():
    if alarma_activa.is_set():
        alarma_activa.clear()
        print("[ALARMA] Desactivada.")
    else:
        alarma_activa.set()
        threading.Thread(target=_sonar_alarma, daemon=True).start()
        print("[ALARMA] Activada.")


def encender_luces():
    print("[LUCES] Encendiendo...")
    if not LUCES_WEBHOOK_URL:
        print("[LUCES] (simulado) Define LUCES_WEBHOOK_URL para controlar luces reales.")
        return
    try:
        import requests
        r = requests.post(LUCES_WEBHOOK_URL, timeout=5)
        print(f"[LUCES] Respuesta: {r.status_code}")
    except Exception as e:
        print(f"[LUCES] Error: {e}")


ACCIONES = {
    ord("1"): ("1) Llamar a la policia", llamar_policia),
    ord("2"): ("2) Activar/desactivar alarma", alternar_alarma),
    ord("3"): ("3) Encender luces", encender_luces),
}


def rostro_mas_grande(ubicaciones):
    return max(ubicaciones, key=lambda u: (u[2] - u[0]) * (u[1] - u[3]))


def confirmar_rostro(frame, ubicacion):
    arriba, derecha, abajo, izquierda = ubicacion
    vista = frame.copy()
    cv2.rectangle(vista, (izquierda, arriba), (derecha, abajo), (0, 200, 0), 2)
    cv2.rectangle(vista, (0, 0), (vista.shape[1], 45), (0, 0, 0), -1)
    cv2.putText(vista, "G: guardar | N: no guardar | Q: salir", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Registrar rostro", vista)
    while True:
        tecla = chr(cv2.waitKey(0) & 0xFF).lower()
        if tecla == "g":
            return "guardar"
        if tecla == "n":
            return "no_guardar"
        if tecla == "q":
            return "salir"


def registrar():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("No se pudo abrir la camara.")
    print("Mira a la camara. ESPACIO = tomar foto, q = salir.")

    mensaje = ""
    mensaje_hasta = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        vista = frame.copy()
        cv2.putText(vista, "ESPACIO: tomar foto | q: salir", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if time.time() < mensaje_hasta:
            cv2.putText(vista, mensaje, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Registrar rostro", vista)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord(" "):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ubicaciones = face_recognition.face_locations(rgb)
            if not ubicaciones:
                mensaje = "No se detecto ningun rostro"
                mensaje_hasta = time.time() + 2
                continue

            ubicacion = rostro_mas_grande(ubicaciones)
            codigo = face_recognition.face_encodings(rgb, [ubicacion])[0]

            decision = confirmar_rostro(frame, ubicacion)
            if decision == "guardar":
                np.save(ARCHIVO_ROSTRO, codigo)
                print("Rostro guardado.")
                break
            if decision == "salir":
                print("Saliste sin guardar.")
                break
            print("Foto descartada. Toma otra.")

    cap.release()
    cv2.destroyAllWindows()


def iniciar():
    if not os.path.exists(ARCHIVO_ROSTRO):
        sys.exit("Primero registra tu rostro: python "
                 + os.path.basename(sys.argv[0]) + " registrar")
    mi_rostro = np.load(ARCHIVO_ROSTRO)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("No se pudo abrir la camara.")

    consecutivos = 0
    menu_hasta = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        ahora = time.time()
        menu_visible = ahora < menu_hasta

        if not menu_visible:
            pequeno = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb = cv2.cvtColor(pequeno, cv2.COLOR_BGR2RGB)
            ubicaciones = face_recognition.face_locations(rgb)
            codigos = face_recognition.face_encodings(rgb, ubicaciones)

            reconocido = False
            for (arriba, derecha, abajo, izquierda), cod in zip(ubicaciones, codigos):
                distancia = face_recognition.face_distance([mi_rostro], cod)[0]
                es_yo = distancia <= UMBRAL_DISTANCIA
                reconocido = reconocido or es_yo
                color = (0, 200, 0) if es_yo else (0, 0, 255)
                cv2.rectangle(frame, (izquierda * 4, arriba * 4),
                              (derecha * 4, abajo * 4), color, 2)

            consecutivos = consecutivos + 1 if reconocido else 0
            estado = "Rostro reconocido..." if reconocido else "Buscando rostro..."
            cv2.putText(frame, estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

            if consecutivos >= FRAMES_PARA_CONFIRMAR:
                menu_hasta = ahora + SEGUNDOS_MENU
                consecutivos = 0
                print("Rostro verificado. Elige una opcion (1, 2 o 3).")
        else:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 150), (0, 0, 0), -1)
            cv2.putText(frame, "Identidad verificada - elige una opcion:", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for i, (texto, _) in enumerate(ACCIONES.values()):
                cv2.putText(frame, texto, (10, 65 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Seguridad facial", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if menu_visible and tecla in ACCIONES:
            threading.Thread(target=ACCIONES[tecla][1], daemon=True).start()

    alarma_activa.clear()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else ""
    if modo == "registrar":
        registrar()
    elif modo == "iniciar":
        iniciar()
    else:
        print("Uso: python " + os.path.basename(sys.argv[0]) + " registrar | iniciar")