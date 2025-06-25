from flask import Flask, request, jsonify
from flask_cors import CORS
from docx import Document
import openai
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
import os



app = Flask(__name__)
CORS(app)

# 🔑 Tu API Key

openai.api_key = os.getenv("OPENAI_API_KEY")

# 🧠 Función con reintentos para generar embedding
def generar_embedding(texto):
    for intento in range(3):
        try:
            response = openai.Embedding.create(
                input=texto,
                model="text-embedding-ada-002",
                request_timeout=60  # ⏱️ Tiempo de espera aumentado
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"⚠️ Intento {intento+1} fallido: {e}")
            time.sleep(5)
    raise Exception("❌ Error crítico: No se pudo generar el embedding.")

# 📘 Leer documento Word
print("📘 Cargando documento Word...")
doc = Document("documento.docx")
full_text = ""
for para in doc.paragraphs:
    if para.text.strip():
        full_text += para.text + "\n"

# 🔹 Dividir en fragmentos
print("✂️ Dividiendo en fragmentos...")
chunk_size = 500
chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
print(f"🧩 Total de fragmentos: {len(chunks)}")

# 🧠 Generar embeddings
print("🔁 Generando embeddings...")
embeddings = []
for i, chunk in enumerate(chunks):
    print(f"🔷 Fragmento {i+1}/{len(chunks)}...")
    vector = generar_embedding(chunk)
    embeddings.append((chunk, vector))

print("🚀 Servidor listo en http://127.0.0.1:5000")

# 💬 Ruta para el chatbot
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    pregunta = data.get("pregunta", "")

    pregunta_emb = generar_embedding(pregunta)

    # 🧮 Calcular similitudes
    similitudes = [
        cosine_similarity([pregunta_emb], [emb])[0][0]
        for _, emb in embeddings
    ]

    # 🏅 Tomar los 3 fragmentos más similares
    top_k = 3
    top_idxs = np.argsort(similitudes)[-top_k:][::-1]
    contexto = "\n".join([embeddings[i][0] for i in top_idxs])

    # 🧾 Instrucción para evitar "según el documento"
    mensajes = [
        {
            "role": "system",
            "content": (
                "Responde con claridad y precisión solo usando el siguiente contexto. "
                "No digas frases como 'según el documento'. Responde como si tú supieras la información directamente."
            )
        },
        {
            "role": "user",
            "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"
        }
    ]

    respuesta = openai.ChatCompletion.create(
        model="gpt-4",
        messages=mensajes
    ).choices[0].message["content"]

    return jsonify({"content": respuesta})

if __name__ == "__main__":
    app.run(debug=True)
