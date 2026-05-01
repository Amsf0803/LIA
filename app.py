from flask import Flask, request, jsonify, render_template, session
import ollama 
import chromadb
import os
from dotenv import load_dotenv
import subprocess
import time
import urllib.request
from urllib.error import URLError

# Carga variables de entorno
load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "lia_polibot_secret_2024")

# --- CONFIGURACIÓN DE IA (POLIBOT) ---
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="memoria_cecyt16")

def asegurar_ollama():
    try:
        urllib.request.urlopen("http://127.0.0.1:11434")
    except URLError:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)

def inicializar_base_vectorial():
    ruta = "conocimiento_cecyt16.txt"
    if not os.path.exists(ruta):
        return
    with open(ruta, "r", encoding="utf-8") as f:
        texto = f.read()
    parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    for i, parrafo in enumerate(parrafos):
        emb = ollama.embeddings(model="nomic-embed-text", prompt=parrafo)["embedding"]
        collection.add(ids=[f"doc_{i}"], embeddings=[emb], documents=[parrafo])

# Inicializar IA al arrancar
asegurar_ollama()
inicializar_base_vectorial()

historiales = {}

def get_historial(session_id):
    if session_id not in historiales:
        historiales[session_id] = [
            {"role": "system", "content": "Eres el asistente del LIA y CECyT 16. Ayuda con dudas de la escuela y proyectos del laboratorio como LIOSITO, LSM y el Torniquete."}
        ]
    return historiales[session_id]

# --- RUTAS DE LA PÁGINA LIA ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lsm')
def lsm():
    return render_template('lsm.html')

@app.route('/torniquete')
def torniquete():
    return render_template('torniquete.html')

# --- RUTA DEL CHATBOT ---
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id")
    mensaje_usuario = data.get("mensaje")
    historial = get_historial(session_id)

    # Búsqueda RAG
    vector_pregunta = ollama.embeddings(model="nomic-embed-text", prompt=mensaje_usuario)["embedding"]
    resultados = collection.query(query_embeddings=[vector_pregunta], n_results=2)
    contexto = "\n".join(resultados['documents'][0]) if resultados['documents'] else ""

    prompt_final = f"Contexto: {contexto}\nUsuario: {mensaje_usuario}"
    historial.append({"role": "user", "content": mensaje_usuario})

    respuesta = ollama.chat(model="llama3.2:3b", messages=[historial[0], {"role": "user", "content": prompt_final}])
    mensaje_ia = respuesta["message"]["content"]
    
    historial.append({"role": "assistant", "content": mensaje_ia})
    return jsonify({"respuesta": mensaje_ia, "session_id": session_id})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)