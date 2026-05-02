from flask import Flask, request, jsonify, render_template, session
import chromadb
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types  # Importante para estructurar el chat

# Carga variables de entorno
load_dotenv()

# Inicializar el nuevo cliente (Automáticamente lee GEMINI_API_KEY del entorno)
client = genai.Client()

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "lia_polibot_secret_2024")

# --- CONFIGURACIÓN DE IA (POLIBOT) ---
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="memoria_cecyt16")

def inicializar_base_vectorial():
    ruta = "conocimiento_lia.txt"
    if not os.path.exists(ruta):
        print(f"Advertencia: No se encontró el archivo {ruta}")
        return
    with open(ruta, "r", encoding="utf-8") as f:
        texto = f.read()
    parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    
    for i, parrafo in enumerate(parrafos):
        # Usando el nuevo modelo de embeddings
        response = client.models.embed_content(
            model="gemini-embedding-001", 
            contents=parrafo
        )
        emb = response.embeddings[0].values
        collection.add(ids=[f"doc_{i}"], embeddings=[emb], documents=[parrafo])
    
    print("Base vectorial inicializada correctamente.")

# Inicializar IA al arrancar
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

@app.route('/polibot')
def polibot():
    return render_template('polibot.html')

@app.route('/liosito')
def liosito():
    return render_template('liosito.html')

# --- RUTA DEL CHATBOT ---
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id")
    mensaje_usuario = data.get("mensaje")
    historial = get_historial(session_id)

    # 1. Búsqueda RAG (Usando el nuevo modelo de embeddings)
    response_emb = client.models.embed_content(
        model="gemini-embedding-001", 
        contents=mensaje_usuario
    )
    vector_pregunta = response_emb.embeddings[0].values
    
    resultados = collection.query(query_embeddings=[vector_pregunta], n_results=2)
    contexto = "\n".join(resultados['documents'][0]) if resultados['documents'] else ""

    prompt_final = f"Contexto: {contexto}\nUsuario: {mensaje_usuario}"
    historial.append({"role": "user", "content": mensaje_usuario})

    # 2. Preparar instrucciones del sistema
    system_instruction = historial[0]["content"] if historial else ""
    config = types.GenerateContentConfig(
        system_instruction=system_instruction
    )
    
    # 3. Construir historial para enviar a Gemini (Estructura de google-genai)
    mensajes_gemini = []
    for msg in historial[1:-1]:
        # La API usa 'model' en vez de 'assistant'
        role = "user" if msg["role"] == "user" else "model" 
        mensajes_gemini.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
        )
        
    # Agregamos la última pregunta inyectada con el contexto de ChromaDB
    mensajes_gemini.append(
        types.Content(role="user", parts=[types.Part.from_text(text=prompt_final)])
    )
    
    # 4. Generar respuesta
    respuesta = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=mensajes_gemini,
        config=config
    )

    mensaje_ia = respuesta.text
    
    # Formatear saltos de línea para que se vean bien en HTML
    mensaje_ia = mensaje_ia.replace("\n", "<br>")
    
    # Reemplazar etiquetas por contenido visual interactivo
    if "[MAPA_UBICACION]" in mensaje_ia:
        mapa_html = '<div style="margin-top: 12px; border-radius: 12px; overflow: hidden; border: 1px solid rgba(56, 189, 248, 0.3); box-shadow: 0 4px 12px rgba(0,0,0,0.2);"><iframe width="100%" height="200" style="border:0;" loading="lazy" allowfullscreen src="https://maps.google.com/maps?q=CECyT%2016%20Hidalgo,%20Kil%C3%B3metro%201.500,%20Actopan%20-%20Pachuca,%20San%20Agust%C3%ADn%20Tlaxiaca,%20Hgo.&t=&z=15&ie=UTF8&iwloc=&output=embed"></iframe></div>'
        mensaje_ia = mensaje_ia.replace("[MAPA_UBICACION]", mapa_html)
        
    if "[DIAGRAMA_ESCUELA]" in mensaje_ia:
        img_html = '<div style="margin-top: 12px; border-radius: 12px; overflow: hidden; border: 1px solid rgba(56, 189, 248, 0.3); box-shadow: 0 4px 12px rgba(0,0,0,0.2);"><img src="/static/img/instalaciones_cecyt16.jpg" alt="Instalaciones CECyT 16" style="width: 100%; height: auto; display: block; object-fit: cover;"></div>'
        mensaje_ia = mensaje_ia.replace("[DIAGRAMA_ESCUELA]", img_html)
    
    historial.append({"role": "assistant", "content": mensaje_ia})
    return jsonify({"respuesta": mensaje_ia, "session_id": session_id})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)