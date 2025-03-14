# Libre AI - Chat con IA Local

Una aplicación web de chat que utiliza modelos de IA locales a través de Ollama, con soporte para procesamiento de PDFs, OCR de imágenes y múltiples conversaciones.

## 📋 Requisitos Previos

### 1. Instalar Ollama

#### Windows
1. Instala WSL2 (Windows Subsystem for Linux 2) si no lo tienes:
   ```powershell
   wsl --install
   ```
2. Descarga e instala Ollama desde [https://ollama.ai/download](https://ollama.ai/download)

#### macOS
1. Descarga e instala Ollama desde [https://ollama.ai/download](https://ollama.ai/download)

#### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Instalar Python
- Descarga e instala Python 3.8 o superior desde [python.org](https://www.python.org/downloads/)
- Asegúrate de marcar la opción "Add Python to PATH" durante la instalación en Windows

### 3. Instalar Tesseract OCR

#### Windows
1. Descarga el instalador de Tesseract desde [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
2. Ejecuta el instalador y asegúrate de:
   - Marcar "Add to PATH"
   - Instalar los paquetes de idiomas que necesites

#### macOS
```bash
brew install tesseract
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-spa  # Para español
```

## 🚀 Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/scharss/libreai.git
cd [NOMBRE_DEL_DIRECTORIO]
```

2. Crea un entorno virtual:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 📥 Descargar Modelos de IA

1. Inicia Ollama:
```bash
# Windows (en WSL)
ollama serve

# macOS/Linux
ollama serve
```

2. Descarga un modelo (elige uno o varios):
```bash
# Modelo pequeño y rápido
ollama pull tinyllama

# Modelo balanceado (recomendado para empezar)
ollama pull mistral

# Modelo más potente
ollama pull deepseek-coder

# Otros modelos disponibles
ollama pull llama2
ollama pull codellama
ollama pull neural-chat
```

## ▶️ Ejecutar la Aplicación

1. Asegúrate de que Ollama esté corriendo en segundo plano

2. Activa el entorno virtual si no está activado:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. Inicia la aplicación:
```bash
python app.py
```

4. Abre tu navegador y ve a:
```
http://localhost:5000
```

## 🎯 Características

- 💬 Chat interactivo con IA local
- 📷 OCR de imágenes con soporte multilenguaje
- 📁 Soporte para múltiples conversaciones
- 📄 Procesamiento y análisis de PDFs
- 🌓 Tema claro/oscuro
- 🌎 Soporte multiidioma (ES/EN)
- ✨ Resaltado de código
- 📋 Copiar respuestas con un clic
- 🧮 Soporte para fórmulas matemáticas

## 🛠️ Configuración

1. **Seleccionar Modelo**: 
   - Haz clic en el botón de configuración en la barra lateral
   - Elige el modelo que desees usar de la lista de modelos disponibles

2. **Cambiar Idioma**:
   - En la configuración, selecciona entre Español o Inglés

3. **Cambiar Tema**:
   - En la configuración, selecciona entre tema Claro u Oscuro

## 📝 Uso de PDFs

1. Haz clic en el botón de subir archivo (📎) junto al campo de mensaje
2. Selecciona un archivo PDF
3. Espera a que se procese el archivo
4. Realiza preguntas sobre el contenido del PDF

## 📝 Uso de Imágenes

1. Haz clic en el botón de cámara (📷) junto al campo de mensaje
2. Selecciona una imagen (formatos soportados: JPEG, PNG, GIF, BMP, WEBP, TIFF)
3. Espera a que se procese la imagen
4. El sistema extraerá el texto de la imagen usando OCR
5. Realiza preguntas sobre el contenido de la imagen

## ⚠️ Solución de Problemas

1. **Ollama no responde**:
   - Verifica que Ollama esté corriendo: `ollama serve`
   - Comprueba que el modelo esté instalado: `ollama list`

2. **Error al cargar modelo**:
   - Intenta descargar el modelo nuevamente: `ollama pull [nombre-modelo]`
   - Verifica los requisitos de sistema para el modelo

3. **Problemas con PDFs**:
   - Asegúrate de que el PDF no esté protegido
   - Verifica que el PDF contenga texto seleccionable

4. **Problemas con OCR**:
   - Asegúrate de que Tesseract OCR está instalado correctamente
   - Verifica que la imagen es clara y el texto es legible
   - Para mejorar el reconocimiento, usa imágenes con buen contraste
   - Si el texto está en otro idioma, instala el paquete de idioma correspondiente

## 💻 Requisitos de Sistema

- **CPU**: 4 núcleos o más
- **RAM**: 8GB mínimo (16GB recomendado)
- **Espacio en Disco**: 10GB mínimo
- **GPU**: Opcional, mejora el rendimiento
- **Sistema Operativo**: Windows 10/11 con WSL2, macOS 12+, o Linux

## 🔒 Privacidad

Toda la inferencia del modelo se realiza localmente en tu máquina. Ningún dato se envía a servicios externos.
