# ⚡ Quick Start - Ready4Hire Mobile App

## 🎯 Metro está corriendo ✅

Ahora necesitas abrir el emulador o dispositivo para conectar la app.

## 📱 Opciones para Conectar

### Opción 1: Android Emulator (Recomendado)

```bash
# En una nueva terminal (Terminal 3)
cd MobileApp

# Verificar que Metro esté corriendo en Terminal 2
# Si no ves errores, continúa

# Ejecutar en Android
npm run android

# O si tienes el emulador ya abierto
npx react-native run-android
```

### Opción 2: Dispositivo Android Físico

1. Habilita USB Debugging en tu teléfono
2. Conecta por USB
3. Ejecuta:

```bash
npm run android
```

### Opción 3: iOS Simulator (Solo macOS)

```bash
npm run ios
```

## 🐛 Si ves "No apps connected"

Esto es **normal** si aún no has ejecutado la app.

### Solución:

```bash
# Terminal 3: Ejecutar la app
npm run android

# Una vez la app esté corriendo en el emulador,
# Metro se conectará automáticamente
```

## ✅ Flujo Completo

```bash
# Terminal 1: WebApp (.NET)
cd WebApp
dotnet run

# Terminal 2: FastAPI (Python)
cd Ready4Hire
source venv/bin/activate
python -m uvicorn app.main_v2_improved:app --reload

# Terminal 3: Metro (YA ESTÁ CORRIENDO ✅)
cd MobileApp
npm start
# Metro está corriendo, esperando conexión...

# Terminal 4: EJECUTAR LA APP
cd MobileApp
npm run android
```

## 🎉 Una vez que la app esté corriendo:

Metro mostrará:
- ✅ "React Native app is now connected"
- ✅ Compilación de JavaScript
- ✅ Hot reload habilitado

## 📋 Próximos Pasos

1. **Abre la app** en el emulador/dispositivo
2. **Registra un usuario** o inicia sesión
3. **Prueba las entrevistas**
4. **Explora la gamificación**

## 🔧 Atajos Útiles en Metro

- `r` - Recargar la app
- `d` - Abrir menú de desarrollador
- `i` - Ejecutar en iOS
- `a` - Ejecutar en Android
- `Ctrl+C` - Detener Metro

---

**Metro está listo. Solo necesitas ejecutar `npm run android` en otra terminal.**
