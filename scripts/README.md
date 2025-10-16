# 🚀 Ready4Hire - Scripts de Utilidad

Scripts multiplataforma para gestionar el sistema Ready4Hire de forma sencilla.

## 📋 Scripts Disponibles

| Script | Plataforma | Descripción |
|--------|------------|-------------|
| `run.sh` | 🐧 Linux / macOS | Script Bash principal |
| `run.ps1` | 🪟 Windows | Script PowerShell (recomendado) |
| `run.bat` | 🪟 Windows | Script Batch/CMD (alternativo) |

---

## 🐧 Linux / macOS - `run.sh`

Script completo para levantar todo el stack de Ready4Hire (Ollama + FastAPI + Blazor).

### Características

- ✅ Gestión automática de servicios
- ✅ Verificación de dependencias
- ✅ Logs centralizados
- ✅ Health checks automáticos
- ✅ Colores y formato mejorado
- ✅ Múltiples modos de operación

### Uso

```bash
# Iniciar todos los servicios
./run.sh

# Modo desarrollo (con auto-reload)
./run.sh --dev

# Detener todos los servicios
./run.sh --stop

# Ver estado de servicios
./run.sh --status

# Ver ayuda
./run.sh --help
```

---

## 🪟 Windows - `run.ps1` (PowerShell)

Script PowerShell equivalente a `run.sh` con todas las funcionalidades.

### Características

- ✅ Todas las características del script Bash
- ✅ Gestión nativa de procesos Windows
- ✅ Detección de puertos con Get-NetTCPConnection
- ✅ Colores en consola PowerShell
- ✅ Manejo de errores robusto

### Uso

```powershell
# Iniciar todos los servicios
.\run.ps1

# Modo desarrollo (con auto-reload)
.\run.ps1 -Dev

# Detener todos los servicios
.\run.ps1 -Stop

# Ver estado de servicios
.\run.ps1 -Status

# Ver ayuda
.\run.ps1 -Help
```

### Requisitos PowerShell

Si obtienes error de política de ejecución:

```powershell
# Ver política actual
Get-ExecutionPolicy

# Permitir scripts (como administrador)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# O ejecutar con bypass
PowerShell -ExecutionPolicy Bypass -File .\run.ps1
```

---

## 🪟 Windows - `run.bat` (CMD)

Script Batch para usuarios que prefieren CMD sobre PowerShell.

### Características

- ✅ Compatible con CMD tradicional
- ✅ No requiere cambios en política de ejecución
- ✅ Funcionalidad completa
- ✅ Colores básicos (ANSI escape codes)

### Uso

```batch
REM Iniciar todos los servicios
run.bat

REM Detener todos los servicios
run.bat stop

REM Ver estado de servicios
run.bat status

REM Ver ayuda
run.bat help
```

---

## 🔧 Variables de Entorno (Todas las Plataformas)

Personaliza el comportamiento mediante variables de entorno:

| Variable | Descripción | Valor por Defecto |
|----------|-------------|-------------------|
| `OLLAMA_MODEL` | Modelo de Ollama | `ready4hire:latest` |
| `API_HOST` | Host de la API | `0.0.0.0` |
| `API_PORT` | Puerto de la API | `8001` |
| `WEBAPP_PORT` | Puerto WebApp | `5214` |

### Ejemplos

**Linux/macOS:**

```bash
export API_PORT=8002
./run.sh
```

**Windows PowerShell:**

```powershell
$env:API_PORT = 8002
.\run.ps1
```

**Windows CMD:**

```batch
set API_PORT=8002
run.bat
```

#### Lo que hace el script

1. **Inicia Ollama Server**
   - Verifica si Ollama está instalado
   - Inicia el servidor si no está corriendo
   - Descarga el modelo `llama3.2:3b` si no existe
   - Verifica conectividad

2. **Inicia API Python (FastAPI v2 DDD)**
   - Verifica y activa virtual environment
   - Verifica dependencias instaladas
   - Inicia API en puerto 8000
   - Realiza health check
   - Soporta modo desarrollo con auto-reload

3. **Inicia WebApp (Blazor)**
   - Verifica .NET SDK instalado
   - Compila el proyecto
   - Inicia en puerto 5214
   - Maneja errores gracefully

4. **Muestra Resumen**
   - URLs de todos los servicios
   - Ubicación de logs
   - Comandos útiles

#### Variables de Entorno

Puedes personalizar el comportamiento con variables de entorno:

```bash
# Cambiar modelo de Ollama
export OLLAMA_MODEL=llama3:latest
./run.sh

# Cambiar puerto de API
export API_PORT=8080
./run.sh

# Cambiar puerto de WebApp
export WEBAPP_PORT=5000
./run.sh
```

#### Logs

Todos los logs se guardan en `Ready4Hire/logs/`:

- `ollama.log` - Logs del servidor Ollama
- `ready4hire_api.log` - Logs de la API FastAPI
- `webapp.log` - Logs de la WebApp Blazor

Ver logs en tiempo real:

```bash
# API
tail -f Ready4Hire/logs/ready4hire_api.log

# Ollama
tail -f Ready4Hire/logs/ollama.log

# WebApp
tail -f Ready4Hire/logs/webapp.log
```

#### Ejemplos

**Inicio rápido**:

```bash
cd scripts
./run.sh
```

**Desarrollo con auto-reload**:

```bash
./run.sh --dev
# La API se reiniciará automáticamente al cambiar código
```

**Detener todo**:

```bash
./run.sh --stop
```

**Ver estado**:

```bash
./run.sh --status
# Salida:
# ✓ Ollama: RUNNING
# ✓ API Python: RUNNING (puerto 8000)
# ✓ WebApp: RUNNING (puerto 5214)
```

---

## 🔧 Troubleshooting

### Error: "Ollama no está instalado"

**Solución**:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Error: "Virtual environment no encontrado"

**Solución**:

```bash
cd Ready4Hire
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Error: "dotnet no está instalado"

**Solución**: El script continuará sin la WebApp. Para instalar .NET:

```bash
# Fedora/RHEL
sudo dnf install dotnet-sdk-9.0

# Ubuntu/Debian
wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install -y dotnet-sdk-9.0
```

### Puerto ya en uso

El script detecta automáticamente y detiene servicios en los puertos necesarios.

Si persiste el problema:

```bash
# Ver qué está usando el puerto
lsof -i :8000

# Matar proceso manualmente
kill -9 <PID>
```

### API no responde al health check

**Diagnóstico**:

```bash
# Ver logs de API
tail -50 Ready4Hire/logs/ready4hire_api.log

# Probar health endpoint
curl http://localhost:8000/api/v2/health
```

**Soluciones comunes**:

1. Esperar unos segundos más (cold start)
2. Verificar que Ollama esté corriendo: `./run.sh --status`
3. Ver documentación completa: [docs/TROUBLESHOOTING.md](../Ready4Hire/docs/TROUBLESHOOTING.md)

---

## 📚 Documentación Adicional

- **Guía Completa**: [docs/INDEX.md](../Ready4Hire/docs/INDEX.md)
- **Configuración**: [docs/CONFIGURATION.md](../Ready4Hire/docs/CONFIGURATION.md)
- **Troubleshooting**: [docs/TROUBLESHOOTING.md](../Ready4Hire/docs/TROUBLESHOOTING.md)
- **API Documentation**: [docs/API_DOCUMENTATION.md](../Ready4Hire/docs/API_DOCUMENTATION.md)
- **Deployment**: [docs/DEPLOYMENT.md](../Ready4Hire/docs/DEPLOYMENT.md)

---

## 🎯 Quick Reference

| Acción | Comando |
|--------|---------|
| Iniciar todo | `./run.sh` |
| Modo desarrollo | `./run.sh --dev` |
| Detener todo | `./run.sh --stop` |
| Ver estado | `./run.sh --status` |
| Ver logs API | `tail -f ../Ready4Hire/logs/ready4hire_api.log` |
| Health check | `curl http://localhost:8000/api/v2/health` |

---

## 🔗 URLs de Servicios

Después de ejecutar `./run.sh`:

- **Ollama API**: http://localhost:11434
- **FastAPI Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/api/v2/health
- **WebApp**: http://localhost:5214

---

**Ready4Hire v2.0** - Sistema de Entrevistas Técnicas con IA ✅
