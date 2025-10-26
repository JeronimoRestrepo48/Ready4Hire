# 📱 Ready4Hire Mobile App

**Versión:** 1.0.0  
**Plataforma:** React Native (iOS/Android)

Aplicación móvil nativa para Ready4Hire con sincronización offline, notificaciones push y acceso completo a todas las funcionalidades del sistema.

---

## 🚀 Características

### Core Features
- ✅ **Autenticación** con JWT y biometría
- ✅ **Entrevistas IA** con reconocimiento de voz
- ✅ **Gamificación** completa (badges, niveles, juegos)
- ✅ **Reportes** con gráficas interactivas
- ✅ **Certificados** descargables y compartibles
- ✅ **Sincronización offline** con AsyncStorage
- ✅ **Notificaciones push** para recordatorios

### Mobile-Specific
- 🎤 **Reconocimiento de voz** para respuestas
- 📸 **Escaneo de CV** con OCR
- 📲 **Compartir** en redes sociales
- 🔔 **Notificaciones** de progreso y logros
- 💾 **Modo offline** con sincronización automática
- 🌓 **Dark mode** nativo

---

## 📋 Requisitos Previos

- **Node.js** >= 18.x
- **React Native CLI** o **Expo CLI**
- **Android Studio** (para Android)
- **Xcode** (para iOS, solo macOS)
- **Java JDK** 11 o superior

---

## 🛠️ Instalación

### 1. Instalar dependencias

```bash
cd MobileApp
npm install
```

### 2. Configurar variables de entorno

Crear `.env`:

```env
API_URL=http://localhost:8001
GRAPHQL_URL=http://localhost:8001/graphql
WS_URL=ws://localhost:8001/subscriptions
```

### 3. Configurar plataformas

#### Android
```bash
# Abrir Android Studio y sincronizar
cd android
./gradlew clean
cd ..
```

#### iOS (solo macOS)
```bash
cd ios
pod install
cd ..
```

---

## 🏃 Ejecutar

### Modo Desarrollo

```bash
# Iniciar Metro bundler
npm start

# Android
npm run android

# iOS
npm run ios
```

### Modo Producción

```bash
# Android APK
cd android
./gradlew assembleRelease

# iOS (requiere cuenta de desarrollador)
cd ios
xcodebuild -workspace Ready4Hire.xcworkspace -scheme Ready4Hire -configuration Release
```

---

## 📁 Estructura del Proyecto

```
MobileApp/
├── src/
│   ├── screens/              # Pantallas de la app
│   │   ├── Auth/
│   │   │   ├── LoginScreen.tsx
│   │   │   └── RegisterScreen.tsx
│   │   ├── Interview/
│   │   │   ├── InterviewListScreen.tsx
│   │   │   ├── InterviewActiveScreen.tsx
│   │   │   └── InterviewResultScreen.tsx
│   │   ├── Gamification/
│   │   │   ├── BadgesScreen.tsx
│   │   │   ├── LeaderboardScreen.tsx
│   │   │   └── GamesScreen.tsx
│   │   ├── Profile/
│   │   │   ├── ProfileScreen.tsx
│   │   │   └── SettingsScreen.tsx
│   │   └── Reports/
│   │       ├── ReportsListScreen.tsx
│   │       └── ReportDetailScreen.tsx
│   │
│   ├── components/           # Componentes reutilizables
│   │   ├── common/
│   │   │   ├── Button.tsx
│   │   │   ├── Card.tsx
│   │   │   └── Input.tsx
│   │   ├── interview/
│   │   │   ├── QuestionCard.tsx
│   │   │   └── AnswerInput.tsx
│   │   └── gamification/
│   │       ├── BadgeCard.tsx
│   │       └── ProgressBar.tsx
│   │
│   ├── navigation/           # Navegación
│   │   ├── AppNavigator.tsx
│   │   ├── AuthNavigator.tsx
│   │   └── TabNavigator.tsx
│   │
│   ├── services/             # Servicios y API
│   │   ├── api/
│   │   │   ├── apolloClient.ts
│   │   │   ├── interviewApi.ts
│   │   │   └── authApi.ts
│   │   ├── storage/
│   │   │   └── AsyncStorageService.ts
│   │   └── notifications/
│   │       └── PushNotificationService.ts
│   │
│   ├── graphql/              # Queries y Mutations GraphQL
│   │   ├── queries/
│   │   ├── mutations/
│   │   └── subscriptions/
│   │
│   ├── hooks/                # Custom hooks
│   │   ├── useAuth.ts
│   │   ├── useInterview.ts
│   │   └── useOfflineSync.ts
│   │
│   ├── store/                # Estado global (Context o Redux)
│   │   ├── AuthContext.tsx
│   │   ├── InterviewContext.tsx
│   │   └── ThemeContext.tsx
│   │
│   ├── utils/                # Utilidades
│   │   ├── validation.ts
│   │   ├── formatting.ts
│   │   └── constants.ts
│   │
│   └── types/                # TypeScript types
│       ├── api.types.ts
│       ├── navigation.types.ts
│       └── models.types.ts
│
├── android/                  # Proyecto Android nativo
├── ios/                      # Proyecto iOS nativo
├── __tests__/                # Tests
├── .env                      # Variables de entorno
├── app.json                  # Configuración de la app
├── babel.config.js           # Configuración de Babel
├── metro.config.js           # Configuración de Metro
├── tsconfig.json             # Configuración de TypeScript
└── package.json
```

---

## 🔌 API Integration

### GraphQL API

Todas las operaciones usan GraphQL para eficiencia:

```typescript
// Ejemplo: Iniciar entrevista
import { useMutation } from '@apollo/client';
import { START_INTERVIEW } from '../graphql/mutations';

const [startInterview] = useMutation(START_INTERVIEW);

const handleStart = async () => {
  const { data } = await startInterview({
    variables: {
      userId: user.id,
      role: 'Software Engineer',
      mode: 'practice'
    }
  });
};
```

### REST API (Fallback)

Para funcionalidades específicas:

```typescript
import axios from 'axios';

const api = axios.create({
  baseURL: process.env.API_URL,
  headers: {
    'Authorization': `Bearer ${token}`
  }
});
```

---

## 💾 Sincronización Offline

### Estrategia

1. **Lectura**: Siempre desde AsyncStorage primero
2. **Escritura**: Queue local + sync cuando hay conexión
3. **Conflictos**: Last-write-wins con timestamps

### Implementación

```typescript
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';

// Guardar offline
await AsyncStorage.setItem('interview_draft', JSON.stringify(data));

// Sincronizar cuando hay conexión
NetInfo.addEventListener(state => {
  if (state.isConnected) {
    syncOfflineData();
  }
});
```

---

## 🔔 Notificaciones Push

### Configuración

```typescript
import PushNotification from 'react-native-push-notification';

PushNotification.configure({
  onNotification: function (notification) {
    console.log('NOTIFICATION:', notification);
  },
  permissions: {
    alert: true,
    badge: true,
    sound: true,
  },
});
```

### Tipos de Notificaciones

- ✅ Recordatorio de entrevista pendiente
- 🎉 Logro desbloqueado
- 📊 Nuevo reporte disponible
- 🏆 Certificado generado
- 📈 Nuevo ranking en leaderboard

---

## 🎨 Theming

### Dark Mode

```typescript
import { useColorScheme } from 'react-native';

const theme = useColorScheme() === 'dark' ? darkTheme : lightTheme;
```

### Colores

```typescript
export const colors = {
  primary: '#6366f1',
  secondary: '#10b981',
  background: {
    light: '#ffffff',
    dark: '#0a0e1a'
  },
  text: {
    light: '#1e293b',
    dark: '#f1f5f9'
  }
};
```

---

## 🧪 Testing

```bash
# Unit tests
npm test

# E2E tests con Detox
npm run test:e2e
```

---

## 📦 Build & Deploy

### Android

```bash
cd android
./gradlew assembleRelease
# APK en: android/app/build/outputs/apk/release/app-release.apk
```

### iOS

```bash
# Requiere cuenta de desarrollador de Apple
xcodebuild -workspace ios/Ready4Hire.xcworkspace \
  -scheme Ready4Hire \
  -configuration Release
```

---

## 🤝 Contribuir

Ver [CONTRIBUTING.md](../CONTRIBUTING.md) en el proyecto principal.

---

## 📄 Licencia

MIT License - Ver [LICENSE](../LICENSE)

---

## 🔗 Links

- [Documentación Principal](../README.md)
- [API Documentation](../docs/API_DOCUMENTATION.md)
- [Backend Repository](../Ready4Hire)
- [WebApp Repository](../WebApp)

---

**Ready4Hire Mobile** - Tu preparación para entrevistas, siempre contigo 📱

