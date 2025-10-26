using Microsoft.EntityFrameworkCore;
using Ready4Hire.MVVM.Models;
using System.Text.Json;

namespace Ready4Hire.Data
{
    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

        public DbSet<User> Users => Set<User>();
        public DbSet<Chat> Chats => Set<Chat>();
        public DbSet<Message> Messages => Set<Message>();
        public DbSet<Badge> Badges => Set<Badge>();
        public DbSet<UserBadge> UserBadges => Set<UserBadge>();
        
        // Nuevos modelos para persistencia completa
        public DbSet<Interview> Interviews => Set<Interview>();
        public DbSet<InterviewQuestion> InterviewQuestions => Set<InterviewQuestion>();
        public DbSet<InterviewAnswer> InterviewAnswers => Set<InterviewAnswer>();
        public DbSet<InterviewReport> InterviewReports => Set<InterviewReport>();
        public DbSet<Certificate> Certificates => Set<Certificate>();
        public DbSet<UserProgress> UserProgress => Set<UserProgress>();

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            // Conversión de listas a JSON
            modelBuilder.Entity<User>()
                .Property(u => u.Skills)
                .HasConversion(
                    v => JsonSerializer.Serialize(v, (JsonSerializerOptions)null),
                    v => JsonSerializer.Deserialize<List<string>>(v, (JsonSerializerOptions)null));

            modelBuilder.Entity<User>()
                .Property(u => u.Softskills)
                .HasConversion(
                    v => JsonSerializer.Serialize(v, (JsonSerializerOptions)null),
                    v => JsonSerializer.Deserialize<List<string>>(v, (JsonSerializerOptions)null));

            modelBuilder.Entity<User>()
                .Property(u => u.Interests)
                .HasConversion(
                    v => JsonSerializer.Serialize(v, (JsonSerializerOptions)null),
                    v => JsonSerializer.Deserialize<List<string>>(v, (JsonSerializerOptions)null));

            // Configuración de relaciones UserBadge
            modelBuilder.Entity<UserBadge>()
                .HasOne(ub => ub.User)
                .WithMany(u => u.Badges)
                .HasForeignKey(ub => ub.UserId)
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<UserBadge>()
                .HasOne(ub => ub.Badge)
                .WithMany(b => b.UserBadges)
                .HasForeignKey(ub => ub.BadgeId)
                .OnDelete(DeleteBehavior.Cascade);

            // Índices para mejor performance
            modelBuilder.Entity<User>()
                .HasIndex(u => u.Email)
                .IsUnique();

            modelBuilder.Entity<User>()
                .HasIndex(u => u.TotalPoints);

            modelBuilder.Entity<User>()
                .HasIndex(u => u.Level);

            modelBuilder.Entity<Badge>()
                .HasIndex(b => b.Category);

            modelBuilder.Entity<UserBadge>()
                .HasIndex(ub => new { ub.UserId, ub.BadgeId })
                .IsUnique();
            
            // ============================================================================
            // Configuración de nuevos modelos de persistencia
            // ============================================================================
            
            // Interview
            modelBuilder.Entity<Interview>()
                .HasIndex(i => i.InterviewId)
                .IsUnique();
            
            modelBuilder.Entity<Interview>()
                .HasIndex(i => new { i.UserId, i.Status });
            
            modelBuilder.Entity<Interview>()
                .HasIndex(i => i.CreatedAt);
            
            modelBuilder.Entity<Interview>()
                .HasOne(i => i.User)
                .WithMany()
                .HasForeignKey(i => i.UserId)
                .OnDelete(DeleteBehavior.Cascade);
            
            // InterviewQuestion
            modelBuilder.Entity<InterviewQuestion>()
                .HasIndex(iq => iq.QuestionId);
            
            modelBuilder.Entity<InterviewQuestion>()
                .HasOne(iq => iq.Interview)
                .WithMany(i => i.Questions)
                .HasForeignKey(iq => iq.InterviewId)
                .OnDelete(DeleteBehavior.Cascade);
            
            // InterviewAnswer
            modelBuilder.Entity<InterviewAnswer>()
                .HasOne(ia => ia.Interview)
                .WithMany(i => i.Answers)
                .HasForeignKey(ia => ia.InterviewId)
                .OnDelete(DeleteBehavior.Cascade);
            
            modelBuilder.Entity<InterviewAnswer>()
                .HasOne(ia => ia.Question)
                .WithOne(iq => iq.Answer)
                .HasForeignKey<InterviewAnswer>(ia => ia.QuestionId)
                .OnDelete(DeleteBehavior.Cascade);
            
            modelBuilder.Entity<InterviewAnswer>()
                .HasIndex(ia => ia.Score);
            
            // InterviewReport
            modelBuilder.Entity<InterviewReport>()
                .HasOne(ir => ir.Interview)
                .WithOne(i => i.Report)
                .HasForeignKey<InterviewReport>(ir => ir.InterviewId)
                .OnDelete(DeleteBehavior.Cascade);
            
            // Certificate
            modelBuilder.Entity<Certificate>()
                .HasIndex(c => c.CertificateId)
                .IsUnique();
            
            modelBuilder.Entity<Certificate>()
                .HasOne(c => c.Interview)
                .WithOne(i => i.Certificate)
                .HasForeignKey<Certificate>(c => c.InterviewId)
                .OnDelete(DeleteBehavior.Cascade);
            
            // UserProgress
            modelBuilder.Entity<UserProgress>()
                .HasIndex(up => new { up.UserId, up.SkillOrTopic, up.Type })
                .IsUnique();
            
            modelBuilder.Entity<UserProgress>()
                .HasOne(up => up.User)
                .WithMany()
                .HasForeignKey(up => up.UserId)
                .OnDelete(DeleteBehavior.Cascade);
            
            modelBuilder.Entity<UserProgress>()
                .HasIndex(up => up.MasteryLevel);
        }
    }
}
