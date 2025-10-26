using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Ready4Hire.MVVM.Models
{
    /// <summary>
    /// Entrevista completa con todas sus preguntas y respuestas
    /// Sincronizada con el backend Python
    /// </summary>
    public class Interview
    {
        [Key]
        public int Id { get; set; }
        
        /// <summary>
        /// ID de la entrevista en el sistema Python (UUID)
        /// </summary>
        [Required]
        [MaxLength(50)]
        public string InterviewId { get; set; } = string.Empty;
        
        // Relación con Usuario
        public int UserId { get; set; }
        public User User { get; set; } = null!;
        
        // Información básica
        [Required]
        [MaxLength(100)]
        public string Role { get; set; } = string.Empty;
        
        [MaxLength(50)]
        public string InterviewType { get; set; } = "technical"; // technical, soft_skills, mixed
        
        [MaxLength(20)]
        public string Mode { get; set; } = "practice"; // practice, exam
        
        [MaxLength(20)]
        public string SkillLevel { get; set; } = "junior"; // junior, mid, senior
        
        [MaxLength(20)]
        public string Status { get; set; } = "active"; // created, active, completed, cancelled
        
        // Fase actual
        [MaxLength(20)]
        public string CurrentPhase { get; set; } = "context"; // context, questions, completed
        
        public int ContextQuestionIndex { get; set; } = 0;
        
        // Fechas
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime? StartedAt { get; set; }
        public DateTime? CompletedAt { get; set; }
        
        // Métricas
        public double AverageScore { get; set; } = 0.0;
        public int TotalQuestions { get; set; } = 0;
        public int CorrectAnswers { get; set; } = 0;
        public int TotalHintsUsed { get; set; } = 0;
        public int TotalTimeSeconds { get; set; } = 0;
        public int CurrentStreak { get; set; } = 0;
        public int MaxStreak { get; set; } = 0;
        
        // Metadata adicional (JSON)
        [Column(TypeName = "jsonb")]
        public string? ContextAnswers { get; set; } // Lista de respuestas de contexto
        
        [Column(TypeName = "jsonb")]
        public string? Metadata { get; set; } // Metadata general
        
        // Relaciones
        public List<InterviewQuestion> Questions { get; set; } = new();
        public List<InterviewAnswer> Answers { get; set; } = new();
        public InterviewReport? Report { get; set; }
        public Certificate? Certificate { get; set; }
    }
    
    /// <summary>
    /// Pregunta dentro de una entrevista
    /// </summary>
    public class InterviewQuestion
    {
        [Key]
        public int Id { get; set; }
        
        // Relación con Interview
        public int InterviewId { get; set; }
        public Interview Interview { get; set; } = null!;
        
        // ID de la pregunta en el sistema Python
        [Required]
        [MaxLength(100)]
        public string QuestionId { get; set; } = string.Empty;
        
        // Contenido de la pregunta
        [Required]
        public string Text { get; set; } = string.Empty;
        
        [MaxLength(50)]
        public string Category { get; set; } = string.Empty;
        
        [MaxLength(20)]
        public string Difficulty { get; set; } = string.Empty;
        
        [MaxLength(100)]
        public string Topic { get; set; } = string.Empty;
        
        // Conceptos esperados (JSON array)
        [Column(TypeName = "jsonb")]
        public string? ExpectedConcepts { get; set; }
        
        // Keywords (JSON array)
        [Column(TypeName = "jsonb")]
        public string? Keywords { get; set; }
        
        public int OrderIndex { get; set; } = 0;
        public DateTime AskedAt { get; set; } = DateTime.UtcNow;
        
        // Relación con respuesta
        public InterviewAnswer? Answer { get; set; }
    }
    
    /// <summary>
    /// Respuesta del candidato con evaluación de IA
    /// </summary>
    public class InterviewAnswer
    {
        [Key]
        public int Id { get; set; }
        
        // Relación con Interview
        public int InterviewId { get; set; }
        public Interview Interview { get; set; } = null!;
        
        // Relación con Question
        public int QuestionId { get; set; }
        public InterviewQuestion Question { get; set; } = null!;
        
        // Respuesta del candidato
        [Required]
        public string AnswerText { get; set; } = string.Empty;
        
        // Evaluación de IA
        public double Score { get; set; } = 0.0;
        public bool IsCorrect { get; set; } = false;
        
        [MaxLength(20)]
        public string Emotion { get; set; } = "neutral"; // happy, sad, confident, nervous, neutral
        
        public double EmotionConfidence { get; set; } = 0.0;
        
        // Feedback detallado (puede ser largo)
        [Column(TypeName = "text")]
        public string? Feedback { get; set; }
        
        // Detalles de evaluación (JSON)
        [Column(TypeName = "jsonb")]
        public string? EvaluationDetails { get; set; } // strengths, improvements, concepts_covered, missing_concepts
        
        // Métricas
        public int TimeTakenSeconds { get; set; } = 0;
        public int HintsUsed { get; set; } = 0;
        public int AttemptsCount { get; set; } = 1;
        
        public DateTime AnsweredAt { get; set; } = DateTime.UtcNow;
    }
    
    /// <summary>
    /// Reporte generado de una entrevista completada
    /// </summary>
    public class InterviewReport
    {
        [Key]
        public int Id { get; set; }
        
        // Relación uno-a-uno con Interview
        public int InterviewId { get; set; }
        public Interview Interview { get; set; } = null!;
        
        // Métricas generales
        public double AverageScore { get; set; }
        public double SuccessRate { get; set; }
        public int Percentile { get; set; } = 50;
        
        // Performance por topic (JSON)
        [Column(TypeName = "jsonb")]
        public string? PerformanceByTopic { get; set; } // { "topic": score }
        
        // Score trend (JSON array)
        [Column(TypeName = "jsonb")]
        public string? ScoreTrend { get; set; } // [score1, score2, ...]
        
        // Tiempos por pregunta (JSON array)
        [Column(TypeName = "jsonb")]
        public string? TimePerQuestion { get; set; }
        
        // Fortalezas y áreas de mejora (JSON arrays)
        [Column(TypeName = "jsonb")]
        public string? Strengths { get; set; }
        
        [Column(TypeName = "jsonb")]
        public string? Improvements { get; set; }
        
        // Conceptos dominados y débiles (JSON arrays)
        [Column(TypeName = "jsonb")]
        public string? ConceptsMastered { get; set; }
        
        [Column(TypeName = "jsonb")]
        public string? ConceptsWeak { get; set; }
        
        // Recursos recomendados (JSON array)
        [Column(TypeName = "jsonb")]
        public string? RecommendedResources { get; set; }
        
        // URL compartible
        [MaxLength(500)]
        public string? ShareableUrl { get; set; }
        
        public DateTime GeneratedAt { get; set; } = DateTime.UtcNow;
    }
    
    /// <summary>
    /// Certificado generado para entrevista exitosa en modo examen
    /// </summary>
    public class Certificate
    {
        [Key]
        public int Id { get; set; }
        
        // Relación uno-a-uno con Interview
        public int InterviewId { get; set; }
        public Interview Interview { get; set; } = null!;
        
        // ID público del certificado (R4H-XXXXXXXXXXXX)
        [Required]
        [MaxLength(50)]
        public string CertificateId { get; set; } = string.Empty;
        
        // Información del candidato
        [Required]
        [MaxLength(200)]
        public string CandidateName { get; set; } = string.Empty;
        
        [Required]
        [MaxLength(100)]
        public string Role { get; set; } = string.Empty;
        
        public double Score { get; set; }
        public int Percentile { get; set; }
        
        // Nivel de certificación
        [MaxLength(50)]
        public string CertificationLevel { get; set; } = string.Empty; // EXCELENCIA, SOBRESALIENTE, etc.
        
        // URLs
        [MaxLength(500)]
        public string ValidationUrl { get; set; } = string.Empty;
        
        [MaxLength(500)]
        public string? DownloadUrl { get; set; }
        
        // Datos del certificado en SVG/PDF (puede ser grande)
        [Column(TypeName = "text")]
        public string? CertificateData { get; set; }
        
        public DateTime IssuedAt { get; set; } = DateTime.UtcNow;
        public bool IsValid { get; set; } = true;
        public DateTime? RevokedAt { get; set; }
    }
    
    /// <summary>
    /// Progreso del usuario por skill/topic
    /// Tracking detallado de evolución
    /// </summary>
    public class UserProgress
    {
        [Key]
        public int Id { get; set; }
        
        // Relación con Usuario
        public int UserId { get; set; }
        public User User { get; set; } = null!;
        
        [Required]
        [MaxLength(100)]
        public string SkillOrTopic { get; set; } = string.Empty;
        
        [MaxLength(20)]
        public string Type { get; set; } = "skill"; // skill, topic
        
        // Métricas de progreso
        public double CurrentLevel { get; set; } = 0.0; // 0-10
        public double MasteryLevel { get; set; } = 0.0; // 0-1
        public int TimesEncountered { get; set; } = 0;
        public int TimesSuccessful { get; set; } = 0;
        
        // Vector de habilidades (JSON) para ML
        [Column(TypeName = "jsonb")]
        public string? SkillVector { get; set; }
        
        // Histórico de scores (JSON array)
        [Column(TypeName = "jsonb")]
        public string? ScoreHistory { get; set; }
        
        public DateTime FirstEncountered { get; set; } = DateTime.UtcNow;
        public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
    }
}

