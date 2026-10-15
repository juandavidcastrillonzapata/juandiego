from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import numpy as np
import cv2
import os
import threading
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

ARCHIVO_ROSTRO = "mi_rostro.npy"
UMBRAL_DISTANCIA = 0.5
alarma_activa = threading.Event()

# --- Funciones de acción (reutilizadas de tu código) ---
def llamar_policia():
    print("[POLICIA] Solicitando ayuda...")
    # Aquí va tu lógica de Twilio

def alternar_alarma():
    if alarma_activa.is_set():
        alarma_activa.clear()
        return "Alarma desactivada"
    else:
        alarma_activa.set()
        # Iniciar hilo de sonido (en web el sonido lo haría el navegador)
        return "Alarma activada"

def encender_luces():
    print("[LUCES] Encendiendo...")
    return "Luces encendidas"

# --- Endpoints de la API ---

@app.post("/api/registrar")
async def registrar_rostro(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ubicaciones = face_recognition.face_locations(rgb)
    
    if not ubicaciones:
        raise HTTPException(status_code=400, detail="No se detectó rostro")
    
    # Tomar el rostro más grande
    ubicacion = max(ubicaciones, key=lambda u: (u[2] - u[0]) * (u[1] - u[3]))
    codigo = face_recognition.face_encodings(rgb, [ubicacion])[0]
    
    np.save(ARCHIVO_ROSTRO, codigo)
    return {"mensaje": "Rostro guardado exitosamente"}

@app.post("/api/verificar")
async def verificar_rostro(file: UploadFile = File(...)):
    if not os.path.exists(ARCHIVO_ROSTRO):
        raise HTTPException(status_code=400, detail="No hay rostro registrado")
    
    mi_rostro = np.load(ARCHIVO_ROSTRO)
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ubicaciones = face_recognition.face_locations(rgb)
    codigos = face_recognition.face_encodings(rgb, ubicaciones)
    
    for cod in codigos:
        distancia = face_recognition.face_distance([mi_rostro], cod)[0]
        if distancia <= UMBRAL_DISTANCIA:
            return {"reconocido": True, "distancia": float(distancia)}
    
    return {"reconocido": False}

@app.post("/api/accion/{accion_id}")
async def ejecutar_accion(accion_id: int):
    if accion_id == 1:
        llamar_policia()
        return {"mensaje": "Llamando a la policía"}
    elif accion_id == 2:
        msg = alternar_alarma()
        return {"mensaje": msg}
    elif accion_id == 3:
        msg = encender_luces()
        return {"mensaje": msg}
    raise HTTPException(status_code=404, detail="Acción no encontrada")