# Asistente de Chat con Procesamiento de PDF e Imágenes

Esta aplicación es un asistente de chat potenciado por Ollama que puede procesar y analizar PDFs e imágenes, realizar OCR (reconocimiento óptico de caracteres) y responder preguntas sobre el contenido de los documentos.

## Características Principales

- 📄 Procesamiento y análisis de documentos PDF
- 🖼️ OCR para extracción de texto de imágenes
- 💬 Chat interactivo con IA usando Ollama
- 📊 Resúmenes automáticos de documentos largos
- 🔍 Búsqueda inteligente en documentos
- 🌐 Interfaz web intuitiva

## Requisitos Previos

### Para todos los sistemas operativos:
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Ollama instalado y funcionando

## Guía de Instalación

### 1. Instalación de Python
#### Windows:
1. Descarga Python desde [python.org](https://www.python.org/downloads/)
2. Ejecuta el instalador y marca la opción "Add Python to PATH"
3. Verifica la instalación:
```bash
python --version
pip --version
```

#### macOS:
```bash
# Usando Homebrew
brew install python
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install python3 python3-pip
```

### 2. Instalación de Ollama

#### Windows:
1. Descarga el instalador desde [ollama.ai](https://ollama.ai/download)
2. Ejecuta el instalador
3. Abre PowerShell y ejecuta:
```powershell
ollama run deepseek-r1:7b
```

#### macOS:
```bash
# Usando Homebrew
brew install ollama
# Inicia Ollama
ollama run deepseek-r1:7b
```

#### Linux:
```bash
# Instala Ollama
curl -fsSL https://ollama.ai/install.sh | sh
# Inicia Ollama
ollama run deepseek-r1:7b
```

### 3. Instalación de Tesseract OCR

#### Windows:
1. Descarga el instalador de [Tesseract para Windows](https://github.com/UB-Mannheim/tesseract/wiki)
2. Ejecuta el instalador
3. Asegúrate de que se instale en `C:\Program Files\Tesseract-OCR`
4. Añade Tesseract a las variables de entorno del sistema:
   - Variable: `PATH`
   - Valor a añadir: `C:\Program Files\Tesseract-OCR`

#### macOS:
```bash
brew install tesseract
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-spa  # Para soporte en español
```

### 4. Configuración del Proyecto

1. Clona o descarga este repositorio:
```bash
git clone <url-del-repositorio>
cd <nombre-del-directorio>
```

2. Crea y activa un entorno virtual:

#### Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

#### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Ejecución de la Aplicación

1. Asegúrate de que Ollama esté corriendo en segundo plano
2. Activa el entorno virtual si no está activado
3. Inicia la aplicación:

```bash
python app.py
```

La aplicación estará disponible en: http://localhost:5000

## Solución de Problemas Comunes

### Error de Tesseract no encontrado
- **Windows**: Verifica que la ruta de instalación sea `C:\Program Files\Tesseract-OCR`
- **Linux/macOS**: Ejecuta `which tesseract` para verificar la instalación

### Error de conexión con Ollama
1. Verifica que Ollama esté corriendo:
   - Windows: Revisa en el Administrador de tareas
   - Linux/macOS: `ps aux | grep ollama`
2. Comprueba que el puerto 11434 esté disponible:
   - Windows: `netstat -ano | findstr 11434`
   - Linux/macOS: `lsof -i :11434`

### Error de puerto 5000 en uso
1. Encuentra el proceso que usa el puerto:
   - Windows: `netstat -ano | findstr :5000`
   - Linux/macOS: `lsof -i :5000`
2. Termina el proceso o usa un puerto diferente

## Notas Adicionales

- La aplicación requiere aproximadamente 8GB de RAM para un funcionamiento óptimo
- Se recomienda tener al menos 10GB de espacio libre en disco
- Para documentos PDF grandes, aumenta el tiempo de espera en `config.py` si es necesario

## Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.
