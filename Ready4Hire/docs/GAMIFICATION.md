# 🎮 Sistema de Gamificación - Ready4Hire v3.0

**Documentación Técnica Completa**

---

## 📋 Tabla de Contenidos

1. [Overview](#overview)
2. [Arquitectura](#arquitectura)
3. [Sistema de Badges](#sistema-de-badges)
4. [Sistema de Niveles](#sistema-de-niveles)
5. [Sistema de Puntos](#sistema-de-puntos)
6. [Juegos Interactivos](#juegos-interactivos)
7. [Base de Datos](#base-de-datos)
8. [API Endpoints](#api-endpoints)
9. [Frontend Integration](#frontend-integration)
10. [Configuración](#configuración)
11. [Testing](#testing)
12. [Troubleshooting](#troubleshooting)

---

## Overview

### ¿Qué es el Sistema de Gamificación?

El sistema de gamificación de Ready4Hire v3.0 transforma la preparación de entrevistas en una **experiencia motivadora e interactiva**. Combina:

- **Badges (Insignias)**: 22 logros únicos para desbloquear
- **Niveles y XP**: Progresión mediante experiencia acumulativa
- **Puntos**: Recompensas por completar actividades
- **Juegos Interactivos**: 6 juegos con IA para entrenar habilidades
- **Rachas**: Bonificaciones por días consecutivos de práctica
- **Leaderboard**: Competencia global entre usuarios

### Objetivos

1. **Motivación**: Incentivar la práctica regular
2. **Engagement**: Aumentar el tiempo de uso de la plataforma
3. **Progreso Visible**: Tracking claro del avance del usuario
4. **Competencia Sana**: Leaderboard para motivar mejora
5. **Diversión**: Hacer el aprendizaje entretenido

---

## Arquitectura

### Stack Tecnológico

```
┌──────────────────────────────────────────────────────────┐
│           Frontend (Blazor C#)                           │
│  - GamificationView.razor (Vista principal)              │
│  - ProfileView.razor (Perfil con badges)                 │
│  - GamificationService.cs (HTTP client)                  │
│  - GamificationModels.cs (DTOs)                          │
└─────────────────────┬────────────────────────────────────┘
                      │ HTTP/REST
                      ↓
┌──────────────────────────────────────────────────────────┐
│           Backend (FastAPI Python)                       │
│  - gamification_service.py (Lógica de negocio)          │
│  - game_engine_service.py (Generación de juegos)        │
│  - badge_service.py (Gestión de badges)                 │
│  - gamification_routes.py (Endpoints API)               │
│  - gamification_dto.py (Data Transfer Objects)          │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ↓
┌──────────────────────────────────────────────────────────┐
│         PostgreSQL Database                              │
│  - Users (Level, Experience, TotalPoints, Streak)        │
│  - Badges (22 badges con metadata)                       │
│  - UserBadges (Progreso y desbloqueos)                   │
└──────────────────────────────────────────────────────────┘
```

### Flujo de Datos

```
Usuario realiza acción → Backend valida → Calcula recompensas
                                        ↓
                          Actualiza BD (Points, XP, Badges)
                                        ↓
                          Frontend recibe actualización
                                        ↓
                          UI muestra progreso visual
```

---

## Sistema de Badges

### Overview

**22 badges únicos** organizados en **4 niveles de rareza** y **5 categorías**.

### Niveles de Rareza

| Rareza | Color | Cantidad | Descripción |
|--------|-------|----------|-------------|
| **Common** | 🔵 Azul | 3 | Logros iniciales para nuevos usuarios |
| **Rare** | 🟣 Morado | 6 | Para usuarios comprometidos con práctica regular |
| **Epic** | 🟠 Naranja | 7 | Para expertos dedicados con altos logros |
| **Legendary** | 🟡 Dorado | 4 | Para verdaderas leyendas del sistema |

### Categorías

1. **General** (general): Logros básicos de uso
2. **Technical** (technical): Logros relacionados con habilidades técnicas
3. **Soft Skills** (soft_skills): Logros de habilidades interpersonales
4. **Achievement** (achievement): Logros de cantidad (X entrevistas)
5. **Milestone** (milestone): Hitos importantes del usuario

### Lista Completa de Badges

#### 🔵 Common (3 badges)

| ID | Nombre | Icon | Requisito | Recompensa |
|----|--------|------|-----------|------------|
| 1 | Primer Paso | 🎯 | Completar 1 entrevista | 50 pts, 100 XP |
| 2 | Practicante | 📚 | Completar 10 entrevistas | 100 pts, 200 XP |
| 3 | Novato Aplicado | 📖 | 5 entrevistas en modo práctica | 75 pts, 150 XP |

#### 🟣 Rare (6 badges)

| ID | Nombre | Icon | Requisito | Recompensa |
|----|--------|------|-----------|------------|
| 4 | Experto | 🎓 | 50 entrevistas completadas | 200 pts, 400 XP |
| 5 | Velocista | ⚡ | Entrevista en <10 minutos | 150 pts, 300 XP |
| 6 | Consistente | 📅 | 7 días consecutivos | 180 pts, 350 XP |
| 7 | Estudioso | 📝 | 20 entrevistas en práctica | 150 pts, 300 XP |
| 8 | Valiente | 💪 | 5 entrevistas en examen | 200 pts, 400 XP |
| 9 | Curioso | 🔍 | Probar todos los tipos | 150 pts, 300 XP |

#### 🟠 Epic (7 badges)

| ID | Nombre | Icon | Requisito | Recompensa |
|----|--------|------|-----------|------------|
| 10 | Maestro | 🥇 | 100 entrevistas | 300 pts, 600 XP |
| 11 | Perfeccionista | 💎 | Score 100% | 350 pts, 700 XP |
| 12 | Racha de Fuego | 🔥 | 30 días consecutivos | 400 pts, 800 XP |
| 13 | Gamer Pro | 🎮 | 50 juegos ganados | 300 pts, 600 XP |
| 14 | Madrugador | 🌅 | Entrevista antes 6am | 250 pts, 500 XP |
| 15 | Nocturno | 🌙 | Entrevista después 11pm | 250 pts, 500 XP |
| 16 | Multilingüe | 🌍 | 3 idiomas diferentes | 350 pts, 700 XP |

#### 🟡 Legendary (4 badges)

| ID | Nombre | Icon | Requisito | Recompensa |
|----|--------|------|-----------|------------|
| 17 | Campeón | 👑 | 500 entrevistas | 1000 pts, 2000 XP |
| 18 | Leyenda | ⭐ | Top 10 leaderboard | 1500 pts, 3000 XP |
| 19 | Imparable | 🚀 | 100 días de racha | 2000 pts, 4000 XP |
| 20 | Leyenda Viva | 🏆 | Todos los badges | 5000 pts, 10000 XP |

### Progreso de Badges

Cada badge tiene un **progreso de 0% a 100%**:

```python
progress = current_value / requirement_value

# Ejemplo: Practicante (10 entrevistas)
# Usuario ha completado 3 entrevistas
progress = 3 / 10 = 0.30 = 30%
```

### Desbloqueo de Badges

Cuando el progreso llega al 100%, el badge se desbloquea:

```python
if progress >= 1.0:
    badge.is_unlocked = True
    badge.unlocked_at = datetime.now()
    user.total_points += badge.reward_points
    user.experience += badge.reward_xp
```

---

## Sistema de Niveles

### Fórmula de XP

El sistema usa una **fórmula exponencial** para escalar la dificultad:

```
XP_necesario = 100 * nivel²

Nivel 1:  0 XP
Nivel 2:  100 XP      (diferencia: 100)
Nivel 3:  400 XP      (diferencia: 300)
Nivel 4:  900 XP      (diferencia: 500)
Nivel 5:  2,500 XP    (diferencia: 1,600)
Nivel 10: 10,000 XP   (diferencia: 6,400)
Nivel 20: 40,000 XP
Nivel 50: 250,000 XP
Nivel 100: 1,000,000 XP
```

### Cálculo de Nivel Actual

```python
def calculate_level(experience: int) -> int:
    """
    Calcula el nivel basado en XP acumulado.
    Formula inversa: nivel = √(experience / 100)
    """
    import math
    level = int(math.sqrt(experience / 100)) + 1
    return max(1, level)
```

### Progreso al Siguiente Nivel

```python
def get_next_level_xp(current_level: int) -> int:
    """XP necesario para el siguiente nivel"""
    return 100 * (current_level + 1) ** 2

def get_xp_progress(current_xp: int, current_level: int) -> float:
    """Progreso porcentual al siguiente nivel"""
    current_level_xp = 100 * current_level ** 2
    next_level_xp = get_next_level_xp(current_level)
    xp_in_current_level = current_xp - current_level_xp
    xp_needed_for_next = next_level_xp - current_level_xp
    return (xp_in_current_level / xp_needed_for_next) * 100
```

### Ganancia de XP

| Acción | XP Ganado |
|--------|-----------|
| Entrevista completada | 50-150 XP (según score) |
| Respuesta correcta | 5 XP |
| Juego ganado | 25-50 XP (según dificultad) |
| Badge desbloqueado | Variable (100-10,000 XP) |
| Día de racha | +2 XP por día |

---

## Sistema de Puntos

### Ganancia de Puntos

| Acción | Puntos |
|--------|--------|
| **Entrevista completada** | 100 pts |
| **Respuesta correcta** | 10 pts |
| **Juego ganado** | 50 pts |
| **Badge Common** | 50-100 pts |
| **Badge Rare** | 150-200 pts |
| **Badge Epic** | 250-400 pts |
| **Badge Legendary** | 1000-5000 pts |
| **Día de racha** | +5 pts |

### Multiplicadores

Multiplicadores aplicados según rareza del badge:

```python
multipliers = {
    "common": 1.0,
    "rare": 1.5,
    "epic": 2.0,
    "legendary": 3.0
}
```

### Leaderboard

Los usuarios se rankean por **total de puntos**:

```sql
SELECT 
    id,
    name,
    total_points,
    level,
    total_games_won,
    COUNT(user_badges.id) as achievements_count
FROM users
LEFT JOIN user_badges ON user_badges.user_id = users.id
    AND user_badges.is_unlocked = true
ORDER BY total_points DESC
LIMIT 100;
```

---

## Juegos Interactivos

### Lista de Juegos

#### 1. 🧩 Code Challenge

**Descripción**: Resolver problemas de código con IA generando el desafío.

**Dificultad**: Easy, Medium, Hard

**Duración**: 10-20 minutos

**Mecánica**:
1. IA genera un problema de código según profesión y dificultad
2. Usuario escribe solución
3. IA evalúa corrección, eficiencia y estilo
4. Feedback detallado con sugerencias

#### 2. ⚡ Quick Quiz

**Descripción**: Quiz rápido de conocimientos técnicos.

**Dificultad**: Easy, Medium, Hard

**Duración**: 5-10 minutos

**Mecánica**:
1. 10 preguntas de opción múltiple
2. Generadas dinámicamente por IA
3. Timer de 30 segundos por pregunta
4. Feedback inmediato por respuesta

#### 3. 🎭 Scenario Simulator

**Descripción**: Simular escenarios de trabajo real.

**Dificultad**: Medium

**Duración**: 15-20 minutos

**Mecánica**:
1. IA describe situación laboral compleja
2. Usuario toma decisiones paso a paso
3. Cada decisión afecta el resultado
4. Evaluación final de habilidades de toma de decisiones

#### 4. ⏱️ Speed Round

**Descripción**: Respuestas rápidas contra el tiempo.

**Dificultad**: Hard

**Duración**: 5 minutos

**Mecánica**:
1. 20 preguntas cortas
2. 15 segundos por pregunta
3. Sin posibilidad de volver atrás
4. Puntuación por velocidad y precisión

#### 5. 📚 Skill Builder

**Descripción**: Entrenamiento progresivo de habilidades.

**Dificultad**: Progressive

**Duración**: 15-30 minutos

**Mecánica**:
1. Seleccionar skill a entrenar
2. Serie de ejercicios de dificultad creciente
3. Feedback adaptativo según desempeño
4. Recomendaciones de mejora

#### 6. 🔧 Problem Solver

**Descripción**: Resolución de problemas complejos paso a paso.

**Dificultad**: Hard

**Duración**: 20-30 minutos

**Mecánica**:
1. IA presenta problema complejo
2. Usuario descompone en sub-problemas
3. Resuelve cada paso con guía de IA
4. Evaluación de enfoque y solución

### Generación de Contenido con IA

Todos los juegos usan el LLM para generar contenido dinámico:

```python
def generate_game_content(
    game_type: str,
    difficulty: str,
    profession: str,
    user_skills: List[str]
) -> dict:
    """
    Genera contenido de juego personalizado usando LLM.
    """
    prompt = f"""
    Generate a {game_type} game for a {profession} professional.
    Difficulty: {difficulty}
    Skills to focus: {', '.join(user_skills)}
    
    Requirements:
    - Relevant to profession
    - Appropriate difficulty
    - Clear instructions
    - Measurable success criteria
    """
    
    response = llm_service.generate(prompt)
    return parse_game_content(response)
```

---

## Base de Datos

### Schema

#### Tabla: Users (modificada)

```sql
CREATE TABLE "Users" (
    "Id" SERIAL PRIMARY KEY,
    "Email" VARCHAR(255) UNIQUE NOT NULL,
    "PasswordHash" VARCHAR(255) NOT NULL,
    "Name" VARCHAR(100),
    "LastName" VARCHAR(100),
    "Country" VARCHAR(100),
    "Skills" JSONB,
    "Softskills" JSONB,
    "Interests" JSONB,
    
    -- Gamification fields (NUEVOS)
    "Profession" VARCHAR(200),
    "ProfessionCategory" VARCHAR(100),
    "ExperienceLevel" VARCHAR(50),  -- junior, mid, senior
    "Level" INTEGER DEFAULT 1,
    "Experience" INTEGER DEFAULT 0,
    "TotalPoints" INTEGER DEFAULT 0,
    "StreakDays" INTEGER DEFAULT 0,
    "LastActivityDate" TIMESTAMP,
    "TotalGamesPlayed" INTEGER DEFAULT 0,
    "TotalGamesWon" INTEGER DEFAULT 0,
    "AvatarUrl" VARCHAR(500),
    "Bio" TEXT,
    
    "CreatedAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX "IX_Users_Email" ON "Users"("Email");
CREATE INDEX "IX_Users_TotalPoints" ON "Users"("TotalPoints");
CREATE INDEX "IX_Users_Level" ON "Users"("Level");
```

#### Tabla: Badges (nueva)

```sql
CREATE TABLE "Badges" (
    "Id" SERIAL PRIMARY KEY,
    "Name" VARCHAR(100) NOT NULL,
    "Description" TEXT NOT NULL,
    "Icon" VARCHAR(10) DEFAULT '🏆',
    "Category" VARCHAR(50) NOT NULL,  -- general, technical, soft_skills, achievement, milestone
    "Rarity" VARCHAR(50) NOT NULL,    -- common, rare, epic, legendary
    "PointsRequired" INTEGER DEFAULT 0,
    "RequirementType" VARCHAR(100),   -- interviews_completed, games_won, streak_days, etc.
    "RequirementValue" INTEGER NOT NULL,
    "RewardPoints" INTEGER DEFAULT 0,
    "RewardXp" INTEGER DEFAULT 0,
    "IsActive" BOOLEAN DEFAULT TRUE,
    "CreatedAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX "IX_Badges_Category" ON "Badges"("Category");
CREATE INDEX "IX_Badges_Rarity" ON "Badges"("Rarity");
```

#### Tabla: UserBadges (nueva)

```sql
CREATE TABLE "UserBadges" (
    "Id" SERIAL PRIMARY KEY,
    "UserId" INTEGER NOT NULL,
    "BadgeId" INTEGER NOT NULL,
    "UnlockedAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "Progress" REAL DEFAULT 0,      -- 0.0 a 1.0 (0% a 100%)
    "IsUnlocked" BOOLEAN DEFAULT FALSE,
    
    FOREIGN KEY ("UserId") REFERENCES "Users"("Id") ON DELETE CASCADE,
    FOREIGN KEY ("BadgeId") REFERENCES "Badges"("Id") ON DELETE CASCADE
);

CREATE UNIQUE INDEX "IX_UserBadges_UserId_BadgeId" 
    ON "UserBadges"("UserId", "BadgeId");
```

### Migraciones

```bash
# Crear migración
cd WebApp
dotnet ef migrations add AddGamificationAndBadges

# Aplicar migración
dotnet ef database update

# Ver migraciones
dotnet ef migrations list
```

---

## API Endpoints

### Gamification Stats

```http
GET /api/v2/gamification/stats/{user_id}

Response 200:
{
  "level": 5,
  "experience": 1250,
  "next_level_xp": 3600,
  "xp_progress": 34.7,
  "total_points": 5420,
  "streak_days": 7,
  "rank": 42,
  "total_games_played": 15,
  "total_games_won": 12,
  "win_rate": 80.0
}
```

### Badges

```http
GET /api/v2/badges

Query params:
  - category (optional): general, technical, soft_skills, achievement, milestone
  - rarity (optional): common, rare, epic, legendary

Response 200:
[
  {
    "id": 1,
    "name": "Primer Paso",
    "description": "Completa tu primera entrevista",
    "icon": "🎯",
    "category": "general",
    "rarity": "common",
    "points_required": 0,
    "requirement_type": "interviews_completed",
    "requirement_value": 1,
    "reward_points": 50,
    "reward_xp": 100,
    "is_active": true
  },
  ...
]
```

```http
GET /api/v2/users/{user_id}/badges

Response 200:
[
  {
    "badge_id": 1,
    "badge_name": "Primer Paso",
    "progress": 1.0,
    "is_unlocked": true,
    "unlocked_at": "2025-10-23T10:30:00Z"
  },
  {
    "badge_id": 2,
    "badge_name": "Practicante",
    "progress": 0.6,  // 6/10 entrevistas
    "is_unlocked": false,
    "unlocked_at": null
  },
  ...
]
```

```http
POST /api/v2/users/{user_id}/badges/check

Body: {
  "action": "interview_completed",
  "metadata": {
    "score": 85,
    "duration_minutes": 15
  }
}

Response 200:
{
  "badges_unlocked": [
    {
      "badge_id": 1,
      "badge_name": "Primer Paso",
      "reward_points": 50,
      "reward_xp": 100
    }
  ],
  "total_points_earned": 50,
  "total_xp_earned": 100,
  "new_level": 2
}
```

### Games

```http
GET /api/v2/games

Response 200:
[
  {
    "id": "code_challenge",
    "name": "Code Challenge",
    "description": "Resuelve problemas de código",
    "icon": "🧩",
    "difficulty_range": ["easy", "hard"],
    "estimated_duration": 15
  },
  ...
]
```

```http
POST /api/v2/games/start

Body: {
  "user_id": "user-123",
  "game_type": "code_challenge",
  "difficulty": "medium",
  "profession": "Backend Developer"
}

Response 200:
{
  "game_session_id": "game-session-uuid",
  "game_type": "code_challenge",
  "challenge": "Implement a function that...",
  "rules": [...],
  "time_limit_minutes": 15
}
```

```http
POST /api/v2/games/submit

Body: {
  "game_session_id": "game-session-uuid",
  "user_id": "user-123",
  "answer": "..."
}

Response 200:
{
  "score": 85,
  "feedback": "Great solution! ...",
  "points_earned": 50,
  "xp_earned": 25,
  "game_won": true
}
```

### Leaderboard

```http
GET /api/v2/gamification/leaderboard

Query params:
  - limit (default: 100)
  - offset (default: 0)

Response 200:
[
  {
    "rank": 1,
    "user_id": "user-456",
    "username": "John Doe",
    "total_points": 15420,
    "level": 12,
    "games_won": 45,
    "achievements_count": 15
  },
  ...
]
```

---

## Frontend Integration

### GamificationView.razor

Vista principal de gamificación con 3 tabs:

```razor
@page "/gamification"

<div class="gamification-container">
    <!-- Header con stats del usuario -->
    <div class="gamification-header">
        <h1>🎮 Gamificación</h1>
        <div class="user-level-card">
            <!-- Nivel y XP -->
        </div>
    </div>

    <!-- Tabs -->
    <div class="tabs">
        <button @onclick="() => SetTab('games')">Juegos</button>
        <button @onclick="() => SetTab('achievements')">Logros</button>
        <button @onclick="() => SetTab('leaderboard')">Ranking</button>
    </div>

    <!-- Contenido según tab -->
    @if (activeTab == "games")
    {
        <!-- Grid de juegos -->
    }
    else if (activeTab == "achievements")
    {
        <!-- Grid de badges -->
    }
    else if (activeTab == "leaderboard")
    {
        <!-- Tabla de leaderboard -->
    }
</div>
```

### ProfileView.razor

Perfil del usuario con 4 tabs:

```razor
@page "/profile"

<div class="profile-container">
    <!-- Header con avatar y quick stats -->
    
    <!-- Tabs -->
    <div class="profile-tabs">
        <button @onclick="() => SetTab('info')">Información</button>
        <button @onclick="() => SetTab('badges')">Badges</button>
        <button @onclick="() => SetTab('progress')">Progreso</button>
        <button @onclick="() => SetTab('settings')">Configuración</button>
    </div>

    <!-- Contenido -->
</div>
```

### GamificationService.cs

```csharp
public class GamificationService
{
    private readonly HttpClient _httpClient;

    public async Task<UserStats> GetUserStatsAsync(int userId)
    {
        var response = await _httpClient.GetAsync($"/api/v2/gamification/stats/{userId}");
        return await response.Content.ReadFromJsonAsync<UserStats>();
    }

    public async Task<List<Badge>> GetBadgesAsync()
    {
        var response = await _httpClient.GetAsync("/api/v2/badges");
        return await response.Content.ReadFromJsonAsync<List<Badge>>();
    }

    public async Task<List<UserBadge>> GetUserBadgesAsync(int userId)
    {
        var response = await _httpClient.GetAsync($"/api/v2/users/{userId}/badges");
        return await response.Content.ReadFromJsonAsync<List<UserBadge>>();
    }
}
```

---

## Configuración

### Backend

```env
# En Ready4Hire/.env
ENABLE_GAMIFICATION=true
GAMIFICATION_XP_MULTIPLIER=1.0
GAMIFICATION_POINTS_MULTIPLIER=1.0
```

### Frontend

```json
// En WebApp/appsettings.json
{
  "Gamification": {
    "Enabled": true,
    "ShowLeaderboard": true,
    "MaxBadgesPerPage": 20
  }
}
```

---

## Testing

### Backend Tests

```python
# tests/test_gamification.py

def test_badge_unlock():
    service = BadgeService()
    user = User(id=1, interviews_completed=1)
    
    unlocked = service.check_badge_unlock(user, "primer_paso")
    assert unlocked == True

def test_level_calculation():
    assert calculate_level(0) == 1
    assert calculate_level(100) == 2
    assert calculate_level(400) == 3
    assert calculate_level(10000) == 10

def test_xp_progress():
    progress = get_xp_progress(current_xp=150, current_level=2)
    assert 0 <= progress <= 100
```

### Frontend Tests

```csharp
// WebApp.Tests/GamificationTests.cs

[Fact]
public async Task GetUserStats_ReturnsStats()
{
    var service = new GamificationService(_httpClient);
    var stats = await service.GetUserStatsAsync(1);
    
    Assert.NotNull(stats);
    Assert.True(stats.Level > 0);
}

[Fact]
public async Task GetBadges_Returns22Badges()
{
    var service = new GamificationService(_httpClient);
    var badges = await service.GetBadgesAsync();
    
    Assert.Equal(22, badges.Count);
}
```

---

## Troubleshooting

### Badges no aparecen

**Problema**: Los badges no se muestran en la UI.

**Solución**:
```bash
# Verificar que hay 22 badges en la BD
psql -h localhost -U ready4hire_user -d ready4hire
SELECT COUNT(*) FROM "Badges";

# Si no hay badges, correr el seed
cd WebApp
dotnet run
# El seed se ejecuta automáticamente
```

### Progreso de badge no actualiza

**Problema**: El progreso de un badge no se actualiza después de completar requisito.

**Solución**:
```python
# En badge_service.py, verificar que check_badge_unlock se llama
# después de cada acción relevante

# Ejemplo: después de completar entrevista
POST /api/v2/users/{user_id}/badges/check
Body: {"action": "interview_completed"}
```

### Nivel no cambia al ganar XP

**Problema**: El usuario gana XP pero el nivel no aumenta.

**Solución**:
```python
# Verificar cálculo de nivel
def update_user_xp(user_id: int, xp_gained: int):
    user = get_user(user_id)
    user.experience += xp_gained
    user.level = calculate_level(user.experience)  # ← Importante
    save_user(user)
```

### Leaderboard no muestra usuarios

**Problema**: El leaderboard está vacío.

**Solución**:
```sql
-- Verificar que hay usuarios con puntos
SELECT id, name, total_points FROM "Users" WHERE total_points > 0;

-- Si no hay usuarios, crear datos de prueba
UPDATE "Users" SET total_points = 1000 WHERE id = 1;
```

---

## Mejores Prácticas

### 1. Actualizar stats en tiempo real

Usar websockets o polling para actualizar stats sin recargar:

```csharp
// Polling cada 30 segundos
private Timer _statsTimer;

protected override void OnInitialized()
{
    _statsTimer = new Timer(30000);
    _statsTimer.Elapsed += async (s, e) => await RefreshStats();
    _statsTimer.Start();
}
```

### 2. Cachear badges

Los badges no cambian frecuentemente, cachearlos en cliente:

```csharp
private static List<Badge> _cachedBadges;
private static DateTime _cacheTime;

public async Task<List<Badge>> GetBadgesAsync()
{
    if (_cachedBadges != null && 
        (DateTime.Now - _cacheTime).TotalMinutes < 60)
    {
        return _cachedBadges;
    }
    
    _cachedBadges = await FetchBadgesFromApi();
    _cacheTime = DateTime.Now;
    return _cachedBadges;
}
```

### 3. Animaciones de desbloqueo

Mostrar animación cuando se desbloquea un badge:

```css
@keyframes badgeUnlock {
    0% { transform: scale(0); opacity: 0; }
    50% { transform: scale(1.2); }
    100% { transform: scale(1); opacity: 1; }
}

.badge-unlocked {
    animation: badgeUnlock 0.5s ease-out;
}
```

---

## Conclusión

El sistema de gamificación de Ready4Hire v3.0 proporciona una **experiencia completa y motivadora** para usuarios que preparan entrevistas. Con 22 badges, sistema de niveles, juegos interactivos y leaderboard, los usuarios tienen múltiples razones para volver y seguir practicando.

**Próximas mejoras**:
- Badges dinámicos generados por IA
- Torneos semanales/mensuales
- Sistema de clanes/equipos
- Recompensas físicas (merchandising)
- Integración con redes sociales

---

**Documentación actualizada**: Octubre 2025  
**Versión**: 3.0  
**Mantenedor**: Ready4Hire Team

