#!/bin/bash
# Script para levantar todo el stack Ready4Hire (Ollama, API Python, Blazor)
# Ejecutar desde la carpeta scripts: ./run_ready4hire.sh

set -e

# 1. Iniciar Ollama server (en background si no está corriendo)
echo "[1/4] Verificando servidor Ollama..."
if ! pgrep -x "ollama" > /dev/null; then
  echo "Iniciando servidor Ollama..."
  nohup ollama serve > ../ollama.log 2>&1 &
  sleep 2
else
  echo "Ollama ya está corriendo."
fi

# 2. Descargar imagen llama3
echo "[2/4] Descargando imagen llama3 para Ollama..."
ollama pull llama3

# 3. Activar entorno virtual y ejecutar API Python (FastAPI)
echo "[3/4] Activando entorno virtual y lanzando API Python..."
cd ../Ready4Hire/app
source ../../venv/bin/activate
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > ../../fastapi.log 2>&1 &
cd ../../scripts
sleep 2

# 4. Compilar y ejecutar Blazor Server
# (Asegúrate de tener dotnet instalado y en PATH)
echo "[4/4] Compilando y ejecutando Blazor Server..."
cd ../WebApp/Ready4Hire

dotnet build#!/bin/bash
# Script para levantar todo el stack Ready4Hire (Ollama, API Python, Blazor)
# Ejecutar desde la carpeta scripts: ./run_ready4hire.sh

set -e

# 1. Iniciar Ollama server (en background si no está corriendo)
echo "[1/4] Verificando servidor Ollama..."
if ! pgrep -x "ollama" > /dev/null; then
  echo "Iniciando servidor Ollama..."
  nohup ollama serve > ../ollama.log 2>&1 &
  sleep 2
else
  echo "Ollama ya está corriendo."
fi

# 2. Descargar imagen llama3
echo "[2/4] Descargando imagen llama3 para Ollama..."
ollama pull llama3

# 3. Activar entorno virtual y ejecutar API Python (FastAPI)
echo "[3/4] Activando entorno virtual y lanzando API Python..."
cd ../Ready4Hire/app
source ../../venv/bin/activate
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > ../../fastapi.log 2>&1 &
cd ../../scripts
sleep 2

# 4. Compilar y ejecutar Blazor Server
# (Asegúrate de tener dotnet instalado y en PATH)
echo "[4/4] Compilando y ejecutando Blazor Server..."
cd ../WebApp/

dotnet build

dotnet run --urls=http://localhost:5214

dotnet run --urls=http://localhost:5214
