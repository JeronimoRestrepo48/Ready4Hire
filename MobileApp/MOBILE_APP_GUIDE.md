jeronimorestrepoangel@fedora:~/Documentos/Ready4Hire/MobileApp$ npm run android

> Ready4Hire@2.0.0 android
> react-native run-android

error Android project not found. Are you sure this is a React Native project? If your Android files are located in a non-standard location (e.g. not inside 'android' folder), consider setting `project.android.sourceDir` option to point to a new location.# 📱 Ready4Hire Mobile App - Guía Completa

## 🎯 Descripción

La aplicación móvil de Ready4Hire es una plataforma completa de entrevistas técnicas con IA, disponible para iOS y Android. Está construida con React Native y TypeScript, proporcionando una experiencia nativa de alta calidad.

## ✨ Características Principales

### 🎯 Entrevistas con IA
- Entrevistas técnicas y de soft skills
- Feedback en tiempo real
- Evaluación automática con LLM
- 40+ profesiones soportadas
- Adaptación de dificultad automática

### 🎮 Gamificación
- 22 badges únicos (4 niveles de rareza)
- Sistema de niveles y XP
- 6 juegos interactivos con IA
- Leaderboard global
- Rachas de práctica diaria

### 📊 Perfil y Estadísticas
- Dashboard personalizado
- Historial de entrevistas
- Progreso visual
- Certificados descargables

### 🔔 Notificaciones
- Push notifications
- Recordatorios de práctica
- Badges desbloqueados
- Logros alcanzados

### 💾 Funcionalidades
- Modo offline
- Cache inteligente
- Sincronización automática
- Multi-idioma (ES, EN, PT, FR)

## 🏗️ Arquitectura

### Estructura del Proyecto

```
MobileApp/
├── src/
│   ├── App.tsx                     # Componente principal
│   ├── navigation/                  # Sistema de navegación
│   │   ├── AppNavigator.tsx        # Navegador principal
│   │   ├── AuthNavigator.tsx       # Pantallas de autenticación
│   │   └── MainNavigator.tsx       # Pantallas principales
│   ├── screens/                     # Pantallas de la app
│   │   ├── auth/                   # Login, Register
│   │   ├── home/                   # Dashboard principal
│   │   ├── interview/              # Entrevistas
│   │   ├── gamification/           # Gamificación
│   │   ├── profile/                # Perfil de usuario
│   │   └── settings/               # Configuración
│   ├── services/                    # Servicios de la app
│   │   ├── api/                    # Cliente API
│   │   ├── CacheService.ts         # Cache de datos
│   │   ├── OfflineService.ts      # Modo offline
│   │   └── NotificationService.ts  # Push notifications
│   ├── store/                       # Redux Store
│   │   ├── index.ts                # Configuración del store
│   │   └── slices/                 # Redux slices
│   ├── components/                  # Componentes reutilizables
│   ├── theme/                       # Tema de la app
│   ├── types/                       # TypeScript definitions
│   └── utils/                       # Utilidades
├── android/                         # Código nativo Android
├── ios/                             # Código nativo iOS
├── package.json
└── tsconfig.json
```

### Tecnologías Utilizadas

- **React Native 0.72.6** - Framework móvil
- **TypeScript 5.3** - Tipado estático
- **Redux Toolkit** - Gestión de estado
- **React Navigation 6** - Navegación
- **React Native Paper** - UI components
- **Axios** - HTTP client
- **AsyncStorage** - Persistencia local
- **Push Notifications** - Notificaciones

## 🚀 Instalación y Setup

### Requisitos Previos

- Node.js >= 18
- React Native CLI
- Xcode (para iOS - solo macOS)
- Android Studio (para Android)
- Watchman (recomendado)

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/your-org/Ready4Hire.git
cd Ready4Hire/MobileApp

# 2. Instalar dependencias
npm install

# 3. iOS (solo en macOS)
cd ios && pod install && cd ..

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus configuraciones

# 5. Iniciar Metro bundler
npm start

# 6. Ejecutar en Android
npm run android

# O en iOS
npm run ios
```

### Configuración de Ambiente

Edita el archivo `.env`:

```env
API_BASE_URL=http://localhost:8001
API_VERSION=v2
ENABLE_PUSH_NOTIFICATIONS=true
ENABLE_OFFLINE_MODE=true
CACHE_TTL=3600
DEBUG=true
```

## 📱 Uso de la Aplicación

### Autenticación

1. **Registro**: Crea una cuenta con email y contraseña
2. **Login**: Inicia sesión con tus credenciales
3. **Sesión persistente**: La app recuerda tu sesión

### Entrevistas

1. **Iniciar**: Tap en "Iniciar Entrevista" desde el dashboard
2. **Responder**: Responde las preguntas de contexto (5 preguntas)
3. **Evaluación**: Recibe feedback en tiempo real
4. **Finalizar**: Ver resultados y descargar certificado

### Gamificación

1. **Badges**: Desbloquea 22 badges únicos
2. **Niveles**: Sube de nivel ganando XP
3. **Juegos**: Juega 6 tipos diferentes de juegos con IA
4. **Leaderboard**: Compite con otros usuarios

### Perfil

1. **Ver Stats**: Dashboard con tu progreso
2. **Historial**: Lista de todas tus entrevistas
3. **Logros**: Badges desbloqueados
4. **Configuración**: Ajusta preferencias

## 🔌 Integración con Backend

### Endpoints Utilizados

#### Interview Endpoints
- `POST /api/v2/interviews` - Iniciar entrevista
- `POST /api/v2/interviews/{id}/answers` - Enviar respuesta
- `POST /api/v2/interviews/{id}/end` - Finalizar entrevista

#### Gamification Endpoints
- `GET /api/v2/gamification/stats/{user_id}` - Estadísticas del usuario
- `GET /api/v2/badges` - Lista de badges
- `GET /api/v2/users/{user_id}/badges` - Badges del usuario
- `GET /api/v2/gamification/leaderboard` - Leaderboard
- `GET /api/v2/games` - Lista de juegos

### Ejemplo de Uso de API

```typescript
import {apiClient} from './services/api/ApiClient';

// Iniciar entrevista
const interview = await apiClient.post('/interviews', {
  userId: 'user123',
  role: 'Backend Developer',
  difficulty: 'mid',
  category: 'technical'
});

// Enviar respuesta
const response = await apiClient.post(`/interviews/${interview.id}/answers`, {
  answer: 'Docker is a...',
  timeTaken: 45
});
```

## 🎨 Personalización

### Tema

Edita `src/theme/index.ts`:

```typescript
export const theme = {
  colors: {
    primary: '#6366F1',  // Cambia el color principal
    secondary: '#8B5CF6',
    // ...
  }
};
```

### Colores de Badges

```typescript
export const getBadgeColor = (rarity: string): string => {
  switch (rarity) {
    case 'legendary': return '#FBBF24'; // Dorado
    case 'epic': return '#F87171'; // Naranja
    case 'rare': return '#A78BFA'; // Morado
    default: return '#60A5FA'; // Azul
  }
};
```

## 🧪 Testing

### Ejecutar Tests

```bash
npm test
```

### Tests Disponibles

- Unit tests
- Integration tests
- E2E tests (Playwright)

## 🚢 Deployment

### Android

```bash
# Build APK
cd android
./gradlew assembleRelease

# Build AAB (para Play Store)
./gradlew bundleRelease
```

### iOS

```bash
# Abrir en Xcode
cd ios
open Ready4Hire.xcworkspace

# Build en Xcode y distribuir
```

### OTA Updates (CodePush)

```bash
# Subir actualización
npx code-push release Ready4Hire-Android android/
npx code-push release Ready4Hire-iOS ios/
```

## 🐛 Troubleshooting

### Problemas Comunes

#### Metro no inicia
```bash
# Limpiar cache
rm -rf node_modules
npm install
npm start --reset-cache
```

#### Errores de iOS
```bash
cd ios
pod deintegrate
pod install
```

#### Errores de Android
```bash
cd android
./gradlew clean
```

### Debugging

```bash
# React Native Debugger
npm install -g react-native-debugger

# Chrome DevTools
# Shake device → Dev Settings → Debug JS Remotely
```

## 📚 Documentación Adicional

- [React Native Docs](https://reactnative.dev/)
- [React Navigation](https://reactnavigation.org/)
- [Redux Toolkit](https://redux-toolkit.js.org/)
- [React Native Paper](https://callstack.github.io/react-native-paper/)

## 🤝 Contribuir

1. Fork el repositorio
2. Crea una branch (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

MIT

## 👥 Equipo

- **Jeronimo Restrepo Angel** - Lead Developer
- **AI Assistant** - Architecture & Implementation

---

**Versión**: 2.0.0  
**Última actualización**: Octubre 2025

