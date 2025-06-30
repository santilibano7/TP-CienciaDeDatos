from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Ruta donde se guardó el modelo entrenado
model_path = "./results"

# Cargar el tokenizer y el modelo desde la carpeta results
print("Cargando modelo y tokenizer desde 'results'...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Crear el pipeline de generación de texto
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Prompt de ejemplo (puedes cambiarlo por otro)
prompt = "[INICIO]\nProducto: Samsung Galaxy S21\nMarca: Samsung\nPrecio: 799\nPuntuación: 1 estrellas\n[RESEÑA]\n"

print("Generando texto...")
resultados = generator(
    prompt,
    max_length=100,
    num_return_sequences=1,
    temperature=0.9,           # Más creatividad
    top_k=50,                  # Considera solo los 50 tokens más probables
    top_p=0.95,                # Nucleus sampling
    repetition_penalty=1.2     # Penaliza la repetición
)

print("\nTexto generado:")
print(resultados[0]['generated_text']) 