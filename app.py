from flask import Flask, request, jsonify, render_template, session
import chromadb
import os
import time
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
# Usamos PersistentClient para guardar la base de datos en una carpeta local
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="memoria_cecyt16")

def inicializar_base_vectorial():
    # Validamos si ya existen documentos para no gastar cuota de la API
    if collection.count() > 0:
        print("Base vectorial ya existe en disco. Omitiendo embeddings.")
        return

    ruta = "conocimiento_lia.txt"
    if not os.path.exists(ruta):
        print(f"Advertencia: No se encontró el archivo {ruta}")
        return
        
    with open(ruta, "r", encoding="utf-8") as f:
        texto = f.read()
    parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    
    print(f"Generando embeddings para {len(parrafos)} párrafos...")
    for i, parrafo in enumerate(parrafos):
        try:
            # Usando el nuevo modelo de embeddings
            response = client.models.embed_content(
                model="gemini-embedding-001", 
                contents=parrafo
            )
            emb = response.embeddings[0].values
            collection.add(ids=[f"doc_{i}"], embeddings=[emb], documents=[parrafo])
            
            # Pausa de 3 segundos entre peticiones para respetar el límite de la API gratuita
            time.sleep(3) 
        except Exception as e:
            print(f"Error al procesar el párrafo {i}: {e}")
            break
            
    print("Base vectorial inicializada y guardada correctamente.")

# Inicializar IA al arrancar
inicializar_base_vectorial()

historiales = {}

def get_historial(session_id):
    if session_id not in historiales:
        historiales[session_id] = [
            {"role": "system", "content": "Eres Polibot, el asistente del LIA y CECyT 16. Responde SIEMPRE en español, de forma breve y directa (máximo 3 oraciones). Ayuda con dudas sobre la escuela y proyectos del laboratorio como LIOSITO, LSM y el Torniquete. Si no sabes algo, dílo en una sola línea."}
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
# --- RUTA DEL CHATBOT ---


# --- RUTA DEL CHATBOT ---
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id")
    mensaje_usuario = data.get("mensaje")
    historial = get_historial(session_id)

    # 1. Búsqueda RAG (Protegida contra error 429 de cuota)
    try:
        response_emb = client.models.embed_content(
            model="gemini-embedding-001", 
            contents=mensaje_usuario
        )
        vector_pregunta = response_emb.embeddings[0].values
        resultados = collection.query(query_embeddings=[vector_pregunta], n_results=1)
        contexto = resultados['documents'][0][0] if resultados['documents'] and resultados['documents'][0] else ""
    except Exception as e:
        print(f"Error al generar embedding: {e}")
        contexto = "" # Si falla por cuota, seguimos sin contexto en lugar de tirar la página

    prompt_final = f"Contexto: {contexto}\nUsuario: {mensaje_usuario}"
    historial.append({"role": "user", "content": mensaje_usuario})

    # 2. Preparar instrucciones del sistema
    system_instruction = historial[0]["content"] if historial else ""
    
    # 3. Construir historial para enviar a Gemini
    mensajes_gemini = []
    for msg in historial[1:-1]:
        role = "user" if msg["role"] == "user" else "model" 
        mensajes_gemini.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
        )
        
    mensajes_gemini.append(
        types.Content(role="user", parts=[types.Part.from_text(text=prompt_final)])
    )
    
    # 4. Generar respuesta (Protegida y con las variables en el scope correcto)
    try:
        respuesta = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=mensajes_gemini,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=300,
                temperature=0.7,
            )
        )
        # Asignamos el mensaje DENTRO del try
        mensaje_ia = respuesta.text
    except Exception as e:
        error_msg = str(e)
        print(f"Error en Gemini: {error_msg}")
        # Asignamos un mensaje de rescate DENTRO del except
        if "429" in error_msg:
            mensaje_ia = "Estoy recibiendo demasiadas consultas en este momento. Por favor, espera un minuto e intenta de nuevo."
        else:
            mensaje_ia = "Hubo un problema de conexión. Intenta más tarde."

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