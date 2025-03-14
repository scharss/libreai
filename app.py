from flask import Flask, render_template, request, Response, stream_with_context, jsonify, send_from_directory
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import logging
import random
import time
import re
import markdown
import html
import fitz  # PyMuPDF
import os
from werkzeug.utils import secure_filename
import math
import pytesseract
from PIL import Image

app = Flask(__name__, static_folder='static')
logging.basicConfig(level=logging.INFO)

# Configuración para subida de archivos
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'}

# Crear carpetas necesarias si no existen
try:
    for folder in [UPLOAD_FOLDER, STATIC_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            app.logger.info(f"Carpeta creada en: {folder}")
except Exception as e:
    app.logger.error(f"Error al crear carpetas: {str(e)}")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['MAX_CHUNK_SIZE'] = 10000
app.config['REQUEST_TIMEOUT'] = 300
app.config['last_image_text'] = None
app.config['image_history'] = {}  # Diccionario para almacenar el historial de imágenes por chat

# Configurar Tesseract
if os.name == 'nt':  # Windows
    try:
        tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if not os.path.exists(tesseract_path):
            raise Exception(f"Tesseract no encontrado en {tesseract_path}")
        
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        version = pytesseract.get_tesseract_version()
        app.logger.info(f"Tesseract OCR configurado correctamente. Versión: {version}")
    except Exception as e:
        app.logger.error(f"Error al configurar Tesseract: {str(e)}")
        app.logger.error("Por favor, asegúrese de que Tesseract OCR está instalado correctamente")
        print("\nERROR: Tesseract OCR no está configurado correctamente")
        print("1. Verifique que Tesseract OCR está instalado en C:\\Program Files\\Tesseract-OCR")
        print("2. Si está instalado en otra ubicación, actualice la ruta en el código")
        print("3. Asegúrese de haber instalado los paquetes de idioma necesarios")
        print(f"Error detallado: {str(e)}\n")

# Configuración de Ollama
OLLAMA_API_URL = 'http://localhost:11434/api/generate'

# Verificar conexión con Ollama
try:
    response = requests.get('http://localhost:11434/api/tags')
    if response.status_code == 200:
        app.logger.info("Conexión con Ollama establecida correctamente")
    else:
        app.logger.error(f"Error al conectar con Ollama. Código de estado: {response.status_code}")
except Exception as e:
    app.logger.error(f"No se pudo conectar con Ollama: {str(e)}")

# Configurar reintentos y timeout
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("http://", adapter)
http.mount("https://", adapter)
http.request = lambda *args, **kwargs: requests.Session.request(http, *args, **{**kwargs, 'timeout': app.config['REQUEST_TIMEOUT']})

# Emojis simplificados
THINKING_EMOJI = '🤔'
RESPONSE_EMOJI = '🤖'
ERROR_EMOJI = '⚠️'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extrae texto de un PDF manteniendo la estructura y el contexto."""
    try:
        doc = fitz.open(file_path)
        total_pages = doc.page_count
        text_parts = []
        
        app.logger.info(f"Iniciando procesamiento de PDF con {total_pages} páginas")
        
        # Procesar el PDF en bloques de páginas para documentos grandes
        PAGES_PER_BLOCK = 50
        for start_page in range(0, total_pages, PAGES_PER_BLOCK):
            end_page = min(start_page + PAGES_PER_BLOCK, total_pages)
            block_text = []
            
            app.logger.info(f"Procesando bloque de páginas {start_page + 1} a {end_page}")
            
            for page_num in range(start_page, end_page):
                try:
                    page = doc[page_num]
                    # Extraer texto manteniendo el formato
                    page_text = page.get_text("text")
                    
                    # Limpiar el texto pero mantener estructura importante
                    page_text = re.sub(r'\s+', ' ', page_text)  # Normalizar espacios
                    page_text = page_text.strip()
                    
                    # Agregar marcador de página para mantener contexto
                    if page_text:
                        block_text.append(f"[Página {page_num + 1}]\n{page_text}")
                    
                except Exception as e:
                    app.logger.error(f"Error en página {page_num + 1}: {str(e)}")
                    continue
            
            if block_text:
                text_parts.append("\n\n".join(block_text))
        
        doc.close()
        
        # Unir todos los bloques de texto
        final_text = "\n\n".join(text_parts)
        
        # Limpiar el texto final
        final_text = re.sub(r'\n{3,}', '\n\n', final_text)  # Normalizar saltos de línea
        final_text = final_text.strip()
        
        app.logger.info(f"PDF procesado completamente. Texto extraído: {len(final_text)} caracteres")
        return final_text
        
    except Exception as e:
        app.logger.error(f"Error extracting text from PDF: {str(e)}")
        return None

def chunk_text(text, chunk_size):
    """Divide el texto en fragmentos semánticos de tamaño aproximado."""
    # Dividir por secciones naturales primero (capítulos, párrafos)
    sections = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    app.logger.info(f"Procesando texto para fragmentación: {len(text)} caracteres")
    
    for section in sections:
        # Estimar el número de tokens (aproximadamente 4 caracteres por token)
        section_size = len(section) // 4
        
        # Si la sección es muy grande, dividirla en párrafos más pequeños
        if section_size > chunk_size:
            paragraphs = re.split(r'(?<=[.!?])\s+', section)
            for paragraph in paragraphs:
                paragraph_size = len(paragraph) // 4
                
                if current_size + paragraph_size > chunk_size and current_chunk:
                    # Asegurar que el chunk termine en un punto final
                    chunk_text = ' '.join(current_chunk).strip()
                    if not chunk_text.endswith(('.', '!', '?')):
                        chunk_text += '.'
                    chunks.append(chunk_text)
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(paragraph)
                current_size += paragraph_size
        else:
            if current_size + section_size > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk).strip()
                if not chunk_text.endswith(('.', '!', '?')):
                    chunk_text += '.'
                chunks.append(chunk_text)
                current_chunk = []
                current_size = 0
            
            current_chunk.append(section)
            current_size += section_size
    
    # Agregar el último chunk si existe
    if current_chunk:
        chunk_text = ' '.join(current_chunk).strip()
        if not chunk_text.endswith(('.', '!', '?')):
            chunk_text += '.'
        chunks.append(chunk_text)
    
    app.logger.info(f"Fragmentación completada: {len(chunks)} fragmentos creados")
    return chunks

@app.route('/')
def home():
    try:
        # Verificar que Ollama esté funcionando
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code != 200:
                app.logger.error("Ollama no está respondiendo correctamente")
                return render_template('error.html', error="Ollama no está respondiendo. Por favor, asegúrate de que Ollama esté corriendo.")
        except requests.exceptions.RequestException as e:
            app.logger.error(f"No se puede conectar con Ollama: {str(e)}")
            return render_template('error.html', error="No se puede conectar con Ollama. Por favor, asegúrate de que Ollama esté corriendo.")
        
        return render_template('index.html')
    except Exception as e:
        app.logger.error(f"Error en la ruta principal: {str(e)}")
        return render_template('error.html', error="Error interno del servidor")

def verify_pdf_processing(text, chunks, filename, file_size):
    """
    Verifica el procesamiento completo de un PDF y genera estadísticas detalladas.
    Retorna un diccionario con los resultados de la verificación y estadísticas.
    """
    verification_results = {
        'success': True,
        'filename': filename,
        'file_size_mb': round(file_size, 2),
        'total_chars': len(text),
        'total_chunks': len(chunks),
        'warnings': [],
        'stats': {}
    }

    # Verificar texto extraído
    if not text or len(text.strip()) == 0:
        verification_results['success'] = False
        verification_results['warnings'].append("No se pudo extraer texto del PDF")
        return verification_results

    # Verificar chunks
    empty_chunks = []
    chunk_sizes = []
    for i, chunk in enumerate(chunks):
        if not chunk or len(chunk.strip()) == 0:
            empty_chunks.append(i + 1)
        chunk_sizes.append(len(chunk))

    if empty_chunks:
        verification_results['warnings'].append(f"Se encontraron chunks vacíos en las posiciones: {empty_chunks}")

    # Calcular estadísticas
    verification_results['stats'] = {
        'avg_chunk_size': sum(chunk_sizes) // len(chunks) if chunks else 0,
        'min_chunk_size': min(chunk_sizes) if chunks else 0,
        'max_chunk_size': max(chunk_sizes) if chunks else 0,
        'total_pages_found': text.count('[Página'),
        'estimated_words': len(text.split()),
        'chunks_distribution': {
            'small_chunks': len([s for s in chunk_sizes if s < 1000]),
            'medium_chunks': len([s for s in chunk_sizes if 1000 <= s < 5000]),
            'large_chunks': len([s for s in chunk_sizes if s >= 5000])
        }
    }

    # Verificar integridad general
    expected_chars = sum(chunk_sizes)
    if abs(len(text) - expected_chars) > 100:  # Permitir pequeña variación por formato
        verification_results['warnings'].append(
            "Posible pérdida de contenido: la suma de caracteres en chunks no coincide con el texto original"
        )

    return verification_results

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            app.logger.error("No se encontró archivo en la solicitud")
            return jsonify({'success': False, 'error': 'No se encontró el archivo'}), 400
        
        file = request.files['file']
        if file.filename == '':
            app.logger.error("Nombre de archivo vacío")
            return jsonify({'success': False, 'error': 'No se seleccionó ningún archivo'}), 400
        
        if not allowed_file(file.filename):
            app.logger.error(f"Tipo de archivo no permitido: {file.filename}")
            return jsonify({'success': False, 'error': 'Tipo de archivo no permitido'}), 400
        
        try:
            original_filename = file.filename
            filename = secure_filename(original_filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            app.logger.info(f"Archivo guardado exitosamente: {file_path}")
            
            if filename.lower().endswith('.pdf'):
                # Verificar tamaño del PDF
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Tamaño en MB
                app.logger.info(f"Tamaño del PDF: {file_size:.2f} MB")
                
                # Procesar PDF
                text = extract_text_from_pdf(file_path)
                if not text:
                    raise Exception("No se pudo extraer texto del PDF")
                
                # Configurar tamaño de chunk basado en el tamaño del documento
                if file_size > 100:  # Para PDFs muy grandes (>100MB)
                    chunk_size = app.config['MAX_CHUNK_SIZE'] // 2
                else:
                    chunk_size = app.config['MAX_CHUNK_SIZE']
                
                chunks = chunk_text(text, chunk_size)
                if not chunks:
                    raise Exception("No se pudo dividir el texto en fragmentos")
                
                # Verificar el procesamiento
                verification_results = verify_pdf_processing(text, chunks, filename, file_size)
                
                # Guardar información del PDF solo si la verificación fue exitosa
                if verification_results['success']:
                    if 'pdf_chunks' not in app.config:
                        app.config['pdf_chunks'] = {}
                    
                    app.config['pdf_chunks'][filename] = {
                        'chunks': chunks,
                        'total_chars': len(text),
                        'total_chunks': len(chunks),
                        'avg_chunk_size': verification_results['stats']['avg_chunk_size'],
                        'file_size_mb': round(file_size, 2),
                        'verification': verification_results,
                        'original_filename': original_filename  # Guardar nombre original
                    }
                
                os.remove(file_path)  # Limpiar archivo temporal
                app.logger.info(f"PDF procesado y verificado: {verification_results}")
                
                return jsonify({
                    'success': verification_results['success'],
                    'verification_results': verification_results,
                    'filename': filename,  # Nombre seguro para uso interno
                    'display_name': original_filename,  # Nombre original para mostrar
                    'message': f'PDF "{original_filename}" procesado exitosamente. Ahora puedes hacer preguntas sobre su contenido.'
                })
            
            return jsonify({
                'success': True, 
                'filename': filename,
                'display_name': original_filename,
                'message': f'Archivo "{original_filename}" subido exitosamente.'
            })
            
        except Exception as e:
            app.logger.error(f"Error procesando archivo: {str(e)}")
            if os.path.exists(file_path):
                os.remove(file_path)  # Limpiar en caso de error
            return jsonify({'success': False, 'error': str(e)}), 500
        
    except Exception as e:
        app.logger.error(f"Error en upload_file: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def clean_math_expressions(text):
    """Limpia y formatea expresiones matemáticas."""
    # No eliminar los backslashes necesarios para LaTeX
    replacements = {
        r'\\begin\{align\*?\}': '',
        r'\\end\{align\*?\}': '',
        r'\\begin\{equation\*?\}': '',
        r'\\end\{equation\*?\}': '',
        r'\\ ': ' '  # Reemplazar \\ espacio con un espacio normal
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    return text

def format_math(text):
    """Formatea expresiones matemáticas para KaTeX."""
    def process_math_content(match):
        content = match.group(1).strip()
        content = clean_math_expressions(content)
        return f'$${content}$$'

    # Procesar comandos especiales de LaTeX antes de los bloques matemáticos
    text = re.sub(r'\\boxed\{\\text\{([^}]*)\}\}', r'<div class="boxed">\1</div>', text)
    text = re.sub(r'\\boxed\{([^}]*)\}', r'<div class="boxed">\1</div>', text)
    
    # Procesar bloques matemáticos inline y display
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: f'$${m.group(1)}$$', text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', lambda m: f'${m.group(1)}$', text)
    text = re.sub(r'\\\[(.*?)\\\]', process_math_content, text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', lambda m: f'${m.group(1)}$', text)
    
    # Preservar comandos LaTeX específicos
    text = re.sub(r'\\times(?![a-zA-Z])', r'\\times', text)  # Preservar \times
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\\frac{\1}{\2}', text)  # Preservar fracciones
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)  # Manejar \text correctamente
    
    return text

def format_code_blocks(text):
    """Formatea bloques de código con resaltado de sintaxis."""
    def replace_code_block(match):
        language = match.group(1) or 'plaintext'
        code = match.group(2).strip()
        return f'```{language}\n{code}\n```'

    # Procesar bloques de código
    text = re.sub(r'```(\w*)\n(.*?)```', replace_code_block, text, flags=re.DOTALL)
    return text

def format_response(text):
    """Formatea la respuesta completa con soporte para markdown, código y matemáticas."""
    # Primero formatear expresiones matemáticas
    text = format_math(text)
    
    # Formatear bloques de código
    text = format_code_blocks(text)
    
    # Convertir markdown a HTML preservando las expresiones matemáticas
    # Escapar temporalmente las expresiones matemáticas
    math_blocks = []
    def math_replace(match):
        math_blocks.append(match.group(0))
        return f'MATH_BLOCK_{len(math_blocks)-1}'

    # Guardar expresiones matemáticas
    text = re.sub(r'\$\$.*?\$\$|\$.*?\$', math_replace, text, flags=re.DOTALL)
    
    # Convertir markdown a HTML
    md = markdown.Markdown(extensions=['fenced_code', 'tables'])
    text = md.convert(text)
    
    # Restaurar expresiones matemáticas
    for i, block in enumerate(math_blocks):
        text = text.replace(f'MATH_BLOCK_{i}', block)
    
    # Limpiar y formatear el texto
    text = text.replace('</think>', '').replace('<think>', '')
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\2', text)
    
    return text.strip()

def decorate_message(message, is_error=False):
    """Decora el mensaje con emojis y formato apropiado."""
    emoji = ERROR_EMOJI if is_error else RESPONSE_EMOJI
    if is_error:
        return f"{emoji} {message}"
    
    formatted_message = format_response(message)
    return f"{emoji} {formatted_message}"

def get_thinking_message():
    """Genera un mensaje de 'pensando' aleatorio."""
    messages = [
        "Analizando tu pregunta...",
        "Procesando la información...",
        "Elaborando una respuesta...",
        "Pensando...",
        "Trabajando en ello...",
    ]
    return f"{THINKING_EMOJI} {random.choice(messages)}"

def find_relevant_chunks(query, chunks, max_chunks=3):
    """
    Encuentra los chunks más relevantes para una consulta dada.
    Utiliza una comparación simple de palabras clave.
    """
    # Normalizar la consulta
    query_words = set(query.lower().split())
    
    # Calcular relevancia para cada chunk
    chunk_scores = []
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk.lower().split())
        # Calcular intersección de palabras
        common_words = query_words & chunk_words
        score = len(common_words)
        chunk_scores.append((i, score))
    
    # Ordenar por relevancia y obtener los índices de los chunks más relevantes
    relevant_indices = sorted(chunk_scores, key=lambda x: x[1], reverse=True)[:max_chunks]
    return [idx for idx, _ in relevant_indices]

def generate_hierarchical_summary(chunks, filename, model):
    """
    Genera un resumen jerárquico de un documento grande.
    """
    app.logger.info(f"Iniciando resumen jerárquico para {filename}")
    
    # Paso 1: Generar resúmenes de nivel base (por cada chunk)
    base_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"""Por favor, genera un resumen conciso de este texto, capturando los puntos principales:

{chunk}

Genera el resumen en no más de 3-4 oraciones."""

        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={'model': model, 'prompt': prompt},
                timeout=30
            )
            if response.status_code == 200:
                summary = response.json().get('response', '').strip()
                base_summaries.append(summary)
                app.logger.info(f"Resumen base {i+1}/{len(chunks)} completado")
        except Exception as e:
            app.logger.error(f"Error en resumen base {i+1}: {str(e)}")
            continue

    # Paso 2: Combinar resúmenes base en resúmenes intermedios
    intermediate_summaries = []
    chunk_size = 5  # Número de resúmenes base a combinar
    
    for i in range(0, len(base_summaries), chunk_size):
        chunk_summaries = base_summaries[i:i + chunk_size]
        combined_text = "\n\n".join(chunk_summaries)
        
        prompt = f"""Combina estos resúmenes en un único resumen coherente, manteniendo los puntos más importantes:

{combined_text}

Genera un resumen unificado en no más de 5 oraciones."""

        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={'model': model, 'prompt': prompt},
                timeout=30
            )
            if response.status_code == 200:
                summary = response.json().get('response', '').strip()
                intermediate_summaries.append(summary)
                app.logger.info(f"Resumen intermedio {len(intermediate_summaries)} completado")
        except Exception as e:
            app.logger.error(f"Error en resumen intermedio: {str(e)}")
            continue

    # Paso 3: Generar resumen final
    final_text = "\n\n".join(intermediate_summaries)
    
    prompt = f"""Crea un resumen ejecutivo final del documento completo basado en estos resúmenes intermedios:

{final_text}

El resumen debe:
1. Capturar los temas y puntos principales del documento
2. Mantener una estructura coherente
3. Incluir las conclusiones más importantes
4. No exceder de 10 párrafos

Organiza el resumen en secciones claras."""

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={'model': model, 'prompt': prompt},
            timeout=60
        )
        if response.status_code == 200:
            final_summary = response.json().get('response', '').strip()
            app.logger.info("Resumen final completado")
            return final_summary
    except Exception as e:
        app.logger.error(f"Error en resumen final: {str(e)}")
        return None

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    model = data.get('model', 'deepseek-r1:7b')
    filename = data.get('pdf_file', None)
    chat_id = data.get('chat_id', None)
    is_pdf_chat = data.get('isPdfChat', False)
    
    app.logger.debug(f"Mensaje recibido: {user_message}")
    app.logger.debug(f"Modelo seleccionado: {model}")
    app.logger.debug(f"Archivo PDF: {filename}")
    app.logger.debug(f"Es chat de PDF: {is_pdf_chat}")

    def generate():
        try:
            # Enviar mensaje inicial de "pensando"
            thinking_msg = get_thinking_message()
            yield json.dumps({
                'thinking': thinking_msg
            }) + '\n'
            
            # Preparar el prompt base
            prompt = user_message
            
            # Si hay historial de imágenes para este chat
            if chat_id and chat_id in app.config['image_history']:
                image_texts = app.config['image_history'][chat_id]
                
                # Detectar si el usuario se refiere específicamente a "esta imagen" o "esta otra imagen"
                if any(phrase in user_message.lower() for phrase in ["esta imagen", "esta otra imagen", "la imagen"]):
                    # Usar solo la última imagen
                    if image_texts:
                        last_image_text = image_texts[-1]
                        context = f"La imagen contiene este texto:\n\n{last_image_text}"
                        prompt = f"""Contexto: {context}

Pregunta del usuario: {user_message}

Por favor, responde la pregunta basándote en el contenido de la imagen mencionada."""
                else:
                    # Si no hay referencia específica, usar todas las imágenes
                    image_contexts = []
                    for idx, img_text in enumerate(image_texts, 1):
                        image_contexts.append(f"Imagen {idx}:\n{img_text}")
                    
                    if image_contexts:
                        context = "\n\n".join(image_contexts)
                        prompt = f"""Contexto: Las siguientes imágenes contienen este texto:

{context}

Pregunta del usuario: {user_message}

Por favor, responde la pregunta basándote en el contenido de todas las imágenes mostradas."""
            
            # Procesar PDF si está activo
            elif is_pdf_chat and filename:
                app.logger.debug(f"Procesando PDF: {filename}")
                if filename not in app.config.get('pdf_chunks', {}):
                    error_msg = f"No se encontró el PDF '{filename}' en la memoria. Por favor, vuelve a subir el archivo."
                    app.logger.error(error_msg)
                    yield json.dumps({
                        'error': decorate_message(error_msg, is_error=True)
                    }) + '\n'
                    return
                
                pdf_data = app.config['pdf_chunks'][filename]
                chunks = pdf_data['chunks']
                
                # Detectar si el usuario pide un resumen
                if any(word in user_message.lower() for word in ['resume', 'resumen', 'síntesis', 'sintetiza']):
                    yield json.dumps({
                        'response': decorate_message("Generando un resumen completo del documento. Esto puede tomar varios minutos...")
                    }) + '\n'
                    
                    final_summary = generate_hierarchical_summary(chunks, filename, model)
                    if final_summary:
                        yield json.dumps({
                            'response': decorate_message(final_summary)
                        }) + '\n'
                        return
                    else:
                        error_msg = "Lo siento, hubo un error generando el resumen. Por favor, intenta nuevamente."
                        yield json.dumps({
                            'error': decorate_message(error_msg, is_error=True)
                        }) + '\n'
                        return
                
                # Encontrar chunks relevantes basados en la pregunta
                relevant_indices = find_relevant_chunks(user_message, chunks)
                context_chunks = []
                
                # Construir contexto con los chunks más relevantes
                for idx in relevant_indices:
                    context_chunks.append(f"[Fragmento {idx + 1} de {len(chunks)}]\n{chunks[idx]}")
                
                context = "\n\n".join(context_chunks)
                
                prompt = f"""Contexto del PDF '{filename}' (mostrando los fragmentos más relevantes):

{context}

Pregunta del usuario:
{user_message}

Por favor, responde la pregunta basándote en el contenido proporcionado del PDF.
Si la respuesta parece incompleta o necesitas más contexto, indícalo."""
            
            payload = {
                'model': model,
                'prompt': prompt,
                'stream': True
            }
            
            app.logger.debug(f"Enviando solicitud a Ollama API")
            
            try:
                response = http.post(
                    OLLAMA_API_URL,
                    json=payload,
                    stream=True,
                    timeout=60
                )
            except requests.exceptions.Timeout:
                error_msg = "La solicitud está tomando más tiempo de lo esperado. Por favor, intenta con un mensaje más corto o espera un momento."
                app.logger.error(error_msg)
                yield json.dumps({
                    'error': decorate_message(error_msg, is_error=True)
                }) + '\n'
                return
            except requests.exceptions.ConnectionError:
                error_msg = "No se pudo conectar con Ollama. Por favor, verifica que Ollama esté corriendo."
                app.logger.error(error_msg)
                yield json.dumps({
                    'error': decorate_message(error_msg, is_error=True)
                }) + '\n'
                return
            
            if response.status_code != 200:
                error_msg = f"Error al conectar con Ollama API. Código de estado: {response.status_code}"
                app.logger.error(error_msg)
                yield json.dumps({
                    'error': decorate_message(error_msg, is_error=True)
                }) + '\n'
                return

            # Limpiar mensaje de "pensando" y comenzar a mostrar la respuesta
            yield json.dumps({'clear_thinking': True}) + '\n'
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                        ai_response = json_response.get('response', '')
                        if ai_response:
                            full_response += ai_response
                            decorated_response = decorate_message(full_response)
                            yield json.dumps({'response': decorated_response}) + '\n'
                    except json.JSONDecodeError as e:
                        app.logger.error(f"Error al decodificar JSON: {str(e)}")
                        continue

        except Exception as e:
            error_msg = f"Error de conexión: {str(e)}"
            app.logger.error(error_msg)
            yield json.dumps({
                'error': decorate_message(error_msg, is_error=True)
            }) + '\n'

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de verificación de salud"""
    status = {
        'status': 'healthy',
        'message': "Servidor en funcionamiento",
        'timestamp': time.time()
    }
    return json.dumps(status)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            app.logger.error("No se encontró imagen en la solicitud")
            return jsonify({'success': False, 'error': 'No se encontró el archivo'})
        
        chat_id = request.form.get('chat_id')  # Obtener el ID del chat desde el formulario
        if not chat_id:
            app.logger.error("No se proporcionó ID de chat")
            return jsonify({'success': False, 'error': 'No se proporcionó ID de chat'})
        
        file = request.files['file']
        if file.filename == '':
            app.logger.error("Nombre de archivo de imagen vacío")
            return jsonify({'success': False, 'error': 'No se seleccionó ningún archivo'})
        
        if not allowed_file(file.filename):
            app.logger.error(f"Tipo de imagen no permitido: {file.filename}")
            return jsonify({'success': False, 'error': 'Tipo de archivo no permitido'})
        
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            static_filepath = os.path.join(app.config['STATIC_FOLDER'], filename)
            
            # Guardar el archivo original
            file.save(filepath)
            app.logger.info(f"Imagen guardada exitosamente: {filepath}")
            
            try:
                # Procesar la imagen para OCR
                image = Image.open(filepath)
                if image.mode in ('RGBA', 'P'):
                    image = image.convert('RGB')
                
                # Guardar una copia en la carpeta estática
                image.save(static_filepath)
                
                text = pytesseract.image_to_string(image, lang='spa+eng')
                text = text.strip()
                
                if not text:
                    app.logger.warning(f"No se pudo extraer texto de la imagen: {filename}")
                    text = "No se pudo extraer texto de esta imagen. Asegúrate de que la imagen contenga texto claro y legible."
                
                os.remove(filepath)  # Limpiar archivo temporal original
                app.logger.info(f"Imagen procesada exitosamente: {len(text)} caracteres extraídos")
                
                # Inicializar el historial de imágenes para este chat si no existe
                if chat_id not in app.config['image_history']:
                    app.config['image_history'][chat_id] = []
                
                # Agregar el texto de la imagen al historial
                app.config['image_history'][chat_id].append(text)
                
                # También mantener la última imagen para compatibilidad
                app.config['last_image_text'] = text
                
                return jsonify({
                    'success': True,
                    'message': '¡Imagen procesada exitosamente! Ahora puedes hacer preguntas sobre su contenido.',
                    'filename': filename,
                    'image_url': f'/static/uploads/{filename}'
                })
                
            except Exception as e:
                app.logger.error(f"Error procesando imagen: {str(e)}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                if os.path.exists(static_filepath):
                    os.remove(static_filepath)
                return jsonify({
                    'success': False,
                    'error': f'Error al procesar la imagen: {str(e)}'
                })
            
        except Exception as e:
            app.logger.error(f"Error guardando imagen: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Error al guardar la imagen: {str(e)}'
            })
            
    except Exception as e:
        app.logger.error(f"Error en upload_image: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error en el servidor: {str(e)}'
        })

# Ruta para servir archivos estáticos de uploads
@app.route('/static/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == '__main__':
    try:
        # Configurar el logger para mostrar más información
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Verificar que el puerto 5000 esté disponible
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 5000))
        sock.close()
        
        if result == 0:
            print("ERROR: El puerto 5000 está en uso.")
            print("Intente estos pasos:")
            print("1. Ejecute 'netstat -ano | findstr :5000' para encontrar el proceso")
            print("2. Cierre la aplicación que está usando el puerto 5000")
            print("3. O inicie la aplicación en un puerto diferente con --port XXXX")
            exit(1)

        # Verificar permisos de la carpeta de uploads
        uploads_path = os.path.abspath(UPLOAD_FOLDER)
        if not os.path.exists(uploads_path):
            try:
                os.makedirs(uploads_path)
                print(f"✓ Carpeta de uploads creada en: {uploads_path}")
            except Exception as e:
                print(f"ERROR: No se pudo crear la carpeta de uploads: {str(e)}")
                exit(1)

        # Verificar conexión con Ollama
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code == 200:
                print("✓ Conexión con Ollama establecida")
            else:
                print("ERROR: Ollama no está respondiendo correctamente")
                print("Por favor, asegúrese de que Ollama esté en ejecución")
                exit(1)
        except requests.exceptions.RequestException as e:
            print("ERROR: No se puede conectar con Ollama")
            print("1. Asegúrese de que Ollama esté instalado")
            print("2. Ejecute Ollama antes de iniciar esta aplicación")
            print(f"Error detallado: {str(e)}")
            exit(1)

        print("\n=== Iniciando Servidor de Chat IA ===")
        print("✓ Todas las verificaciones completadas")
        print("✓ Servidor iniciando en: http://127.0.0.1:5000")
        print("* Presione Ctrl+C para detener el servidor")
        print("=====================================\n")

        # Iniciar el servidor con host='0.0.0.0' para permitir conexiones externas
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except Exception as e:
        print(f"\nERROR CRÍTICO: No se pudo iniciar el servidor")
        print(f"Causa: {str(e)}")
        print("\nPor favor, verifique:")
        print("1. Que no haya otra aplicación usando el puerto 5000")
        print("2. Que tenga permisos de administrador si es necesario")
        print("3. Que todas las dependencias estén instaladas correctamente")
        exit(1) 