# 📱 Ready4Hire Mobile App

Aplicación móvil para la plataforma de entrevistas técnicas Ready4Hire, construida con React Native y TypeScript.

## 🚀 Características

- ✅ **Entrevistas con IA** - Entrevistas técnicas y de soft skills con feedback en tiempo real
- 🎮 **Gamificación** - Sistema de badges, niveles, XP y leaderboard
- 📊 **Dashboard Personalizado** - Estadísticas y progreso del usuario
- 🔔 **Push Notifications** - Notificaciones y recordatorios
- 💾 **Modo Offline** - Funcionalidad sin conexión con sincronización automática
- 🌐 **Multi-idioma** - Soporte para ES, EN, PT, FR

## 📋 Requisitos Previos

- Node.js >= 18
- React Native CLI
- Xcode (para iOS - solo macOS)
- Android Studio (para Android)
- Watchman (recomendado)

## 🛠️ Instalación

```bash
# 1. Instalar dependencias
npm install

# 2. iOS (solo en macOS)
cd ios && pod install && cd ..

# 3. Iniciar Metro bundler
npm start

# 4. Ejecutar en Android
npm run android

# O en iOS
npm run ios
```

## 🏗️ Estructura del Proyecto

```
MobileApp/
├── src/
│   ├── App.tsx                 # Componente principal
│   ├── navigation/              # Sistema de navegación
│   │   ├── AppNavigator.tsx    # Navegador principal
│   │   ├── AuthNavigator.tsx   # Pantallas de autenticación
│   │   └── MainNavigator.tsx   # Pantallas principales
│   ├── screens/                # Pantallas de la app
│   │   ├── auth/               # Login, Register
│   │   ├── home/               # Dashboard
│   │   ├── interview/         # Entrevistas
│   │   ├── gamification/      # Gamificación
│   │   ├── profile/           # Perfil de usuario
│   │   └── settings/          # Configuración
│   ├── services/               # Servicios de la app
│   │   ├── api/               # Cliente API
│   │   ├── CacheService.ts    # Cache de datos
│   │   ├── OfflineService.ts  # Modo offline
│   │   └── NotificationService.ts # Push notifications
│   ├── store/                  # Redux Store
│   │   ├── index.ts           # Configuración del store
│   │   └── slices/            # Redux slices
│   ├── theme/                  # Tema de la app
│   ├── types/                  # TypeScript definitions
│   └── utils/                  # Utilidades
├── android/                    # Código nativo Android
├── ios/                        # Código nativo iOS
└── package.json
```

## 🔧 Configuración

### Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
API_BASE_URL=http://localhost:8001
API_VERSION=v2
WEBAPP_BASE_URL=http://localhost:5214
ENABLE_PUSH_NOTIFICATIONS=true
```

### Tecnologías Utilizadas

- **React Native 0.72.6** - Framework móvil
- **TypeScript 5.3** - Tipado estático
- **Redux Toolkit** - Gestión de estado
- **React Navigation 6** - Navegación
- **React Native Paper** - UI components
- **Axios** - HTTP client
- **AsyncStorage** - Persistencia local

## 📱 Uso de la Aplicación

### Autenticación
1. **Registro**: Crea una cuenta con email y contraseña
2. **Login**: Inicia sesión con tus credenciales
3. **Sesión persistente**: La app recuerda tu sesión automáticamente

### Entrevistas
1. **Iniciar**: Tap en "Iniciar Entrevista" desde el dashboard
2. **Responder**: Responde las preguntas de contexto y técnicas
3. **Evaluación**: Recibe feedback en tiempo real
4. **Finalizar**: Ver resultados y estadísticas

### Gamificación
1. **Badges**: Desbloquea badges únicos completando logros
2. **Niveles**: Sube de nivel ganando XP
3. **Juegos**: Juega diferentes tipos de juegos con IA
4. **Leaderboard**: Compite con otros usuarios

## 🔌 Integración con Backend

### Endpoints Principales

#### Entrevistas
- `POST /api/v2/interviews` - Iniciar entrevista
- `POST /api/v2/interviews/{id}/answers` - Enviar respuesta
- `POST /api/v2/interviews/{id}/end` - Finalizar entrevista

#### Gamificación
- `GET /api/v2/gamification/stats/{user_id}` - Estadísticas del usuario
- `GET /api/v2/badges` - Lista de badges
- `GET /api/v2/users/{user_id}/badges` - Badges del usuario
- `GET /api/v2/gamification/leaderboard` - Leaderboard
- `GET /api/v2/games` - Lista de juegos

## 🚢 Build y Deployment

### Android

```bash
cd android
./gradlew assembleRelease  # APK
./gradlew bundleRelease    # AAB para Play Store
```

### iOS

```bash
cd ios
open Ready4Hire.xcworkspace
# Build en Xcode y distribuir
```

## 🧪 Testing

```bash
npm test
```

## 🐛 Troubleshooting

### Metro no inicia
```bash
rm -rf node_modules
npm install
npm start --reset-cache
```

### Errores de iOS
```bash
cd ios
pod deintegrate
pod install
```

### Errores de Android
```bash
cd android
./gradlew clean
```

## 📚 Documentación Adicional

- [React Native Docs](https://reactnative.dev/)
- [React Navigation](https://reactnavigation.org/)
- [Redux Toolkit](https://redux-toolkit.js.org/)
- [React Native Paper](https://callstack.github.io/react-native-paper/)

## 📄 Licencia

MIT

## 👥 Equipo

- Ready4Hire Development Team

---

**Versión**: 2.0.0
