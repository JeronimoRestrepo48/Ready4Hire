using System;
using Microsoft.EntityFrameworkCore.Migrations;
using Npgsql.EntityFrameworkCore.PostgreSQL.Metadata;

#nullable disable

namespace Ready4Hire.Migrations
{
    /// <inheritdoc />
    public partial class AddCompletePersistenceModels : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "Interviews",
                columns: table => new
                {
                    Id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    InterviewId = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    UserId = table.Column<int>(type: "integer", nullable: false),
                    Role = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: false),
                    InterviewType = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    Mode = table.Column<string>(type: "character varying(20)", maxLength: 20, nullable: false),
                    SkillLevel = table.Column<string>(type: "character varying(20)", maxLength: 20, nullable: false),
                    Status = table.Column<string>(type: "character varying(20)", maxLength: 20, nullable: false),
                    CurrentPhase = table.Column<string>(type: "character varying(20)", maxLength: 20, nullable: false),
                    ContextQuestionIndex = table.Column<int>(type: "integer", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    StartedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: true),
                    CompletedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: true),
                    AverageScore = table.Column<double>(type: "double precision", nullable: false),
                    TotalQuestions = table.Column<int>(type: "integer", nullable: false),
                    CorrectAnswers = table.Column<int>(type: "integer", nullable: false),
                    TotalHintsUsed = table.Column<int>(type: "integer", nullable: false),
                    TotalTimeSeconds = table.Column<int>(type: "integer", nullable: false),
                    CurrentStreak = table.Column<int>(type: "integer", nullable: false),
                    MaxStreak = table.Column<int>(type: "integer", nullable: false),
                    ContextAnswers = table.Column<string>(type: "jsonb", nullable: true),
                    Metadata = table.Column<string>(type: "jsonb", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Interviews", x => x.Id);
                    table.ForeignKey(
                        name: "FK_Interviews_Users_UserId",
                        column: x => x.UserId,
                        principalTable: "Users",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "UserProgress",
                columns: table => new
                {
                    Id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    UserId = table.Column<int>(type: "integer", nullable: false),
                    SkillOrTopic = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: false),
                    Type = table.Column<string>(type: "character varying(20)", maxLength: 20, nullable: false),
                    CurrentLevel = table.Column<double>(type: "double precision", nullable: false),
                    MasteryLevel = table.Column<double>(type: "double precision", nullable: false),
                    TimesEncountered = table.Column<int>(type: "integer", nullable: false),
                    TimesSuccessful = table.Column<int>(type: "integer", nullable: false),
                    SkillVector = table.Column<string>(type: "jsonb", nullable: true),
                    ScoreHistory = table.Column<string>(type: "jsonb", nullable: true),
                    FirstEncountered = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    LastUpdated = table.Column<DateTime>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_UserProgress", x => x.Id);
                    table.ForeignKey(
                        name: "FK_UserProgress_Users_UserId",
                        column: x => x.UserId,
                        principalTable: "Users",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "Certificates",
                columns: table => new
                {
                    Id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    InterviewId = table.Column<int>(type: "integer", nullable: false),
                    CertificateId = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    CandidateName = table.Column<string>(type: "character varying(200)", maxLength: 200, nullable: false),
                    Role = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: false),
                    Score = table.Column<double>(type: "double precision", nullable: false),
                    Percentile = table.Column<int>(type: "integer", nullable: false),
                    CertificationLevel = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    ValidationUrl = table.Column<string>(type: "character varying(500)", maxLength: 500, nullable: false),
                    DownloadUrl = table.Column<string>(type: "character varying(500)", maxLength: 500, nullable: true),
                    CertificateData = table.Column<string>(type: "text", nullable: true),
                    IssuedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    IsValid = table.Column<bool>(type: "boolean", nullable: false),
                    RevokedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Certificates", x => x.Id);
                    table.ForeignKey(
                        name: "FK_Certificates_Interviews_InterviewId",
                        column: x => x.InterviewId,
                        principalTable: "Interviews",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "InterviewQuestions",
                columns: table => new
                {
                    Id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    InterviewId = table.Column<int>(type: "integer", nullable: false),
                    QuestionId = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: false),
                    Text = table.Column<string>(type: "text", nullable: false),
                    Category = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    Difficulty = table.Column<string>(type: "character varying(20)", maxLength: 20, nullable: false),
                    Topic = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: false),
                    ExpectedConcepts = table.Column<string>(type: "jsonb", nullable: true),
                    Keywords = table.Column<string>(type: "jsonb", nullable: true),
                    OrderIndex = table.Column<int>(type: "integer", nullable: false),
                    AskedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_InterviewQuestions", x => x.Id);
                    table.ForeignKey(
                        name: "FK_InterviewQuestions_Interviews_InterviewId",
                        column: x => x.InterviewId,
                        principalTable: "Interviews",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "InterviewReports",
                columns: table => new
                {
                    Id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    InterviewId = table.Column<int>(type: "integer", nullable: false),
                    AverageScore = table.Column<double>(type: "double precision", nullable: false),
                    SuccessRate = table.Column<double>(type: "double precision", nullable: false),
                    Percentile = table.Column<int>(type: "integer", nullable: false),
                    PerformanceByTopic = table.Column<string>(type: "jsonb", nullable: true),
                    ScoreTrend = table.Column<string>(type: "jsonb", nullable: true),
                    TimePerQuestion = table.Column<string>(type: "jsonb", nullable: true),
                    Strengths = table.Column<string>(type: "jsonb", nullable: true),
                    Improvements = table.Column<string>(type: "jsonb", nullable: true),
                    ConceptsMastered = table.Column<string>(type: "jsonb", nullable: true),
                    ConceptsWeak = table.Column<string>(type: "jsonb", nullable: true),
                    RecommendedResources = table.Column<string>(type: "jsonb", nullable: true),
                    ShareableUrl = table.Column<string>(type: "character varying(500)", maxLength: 500, nullable: true),
                    GeneratedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_InterviewReports", x => x.Id);
                    table.ForeignKey(
                        name: "FK_InterviewReports_Interviews_InterviewId",
                        column: x => x.InterviewId,
                        principalTable: "Interviews",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "InterviewAnswers",
                columns: table => new
                {
                    Id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    InterviewId = table.Column<int>(type: "integer", nullable: false),
                    QuestionId = table.Column<int>(type: "integer", nullable: false),
                    AnswerText = table.Column<string>(type: "text", nullable: false),
                    Score = table.Column<double>(type: "double precision", nullable: false),
                    IsCorrect = table.Column<bool>(type: "boolean", nullable: false),
                    Emotion = table.Column<string>(type: "character varying(20)", maxLength: 20, nullable: false),
                    EmotionConfidence = table.Column<double>(type: "double precision", nullable: false),
                    Feedback = table.Column<string>(type: "text", nullable: true),
                    EvaluationDetails = table.Column<string>(type: "jsonb", nullable: true),
                    TimeTakenSeconds = table.Column<int>(type: "integer", nullable: false),
                    HintsUsed = table.Column<int>(type: "integer", nullable: false),
                    AttemptsCount = table.Column<int>(type: "integer", nullable: false),
                    AnsweredAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_InterviewAnswers", x => x.Id);
                    table.ForeignKey(
                        name: "FK_InterviewAnswers_InterviewQuestions_QuestionId",
                        column: x => x.QuestionId,
                        principalTable: "InterviewQuestions",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_InterviewAnswers_Interviews_InterviewId",
                        column: x => x.InterviewId,
                        principalTable: "Interviews",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_Certificates_CertificateId",
                table: "Certificates",
                column: "CertificateId",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_Certificates_InterviewId",
                table: "Certificates",
                column: "InterviewId",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_InterviewAnswers_InterviewId",
                table: "InterviewAnswers",
                column: "InterviewId");

            migrationBuilder.CreateIndex(
                name: "IX_InterviewAnswers_QuestionId",
                table: "InterviewAnswers",
                column: "QuestionId",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_InterviewAnswers_Score",
                table: "InterviewAnswers",
                column: "Score");

            migrationBuilder.CreateIndex(
                name: "IX_InterviewQuestions_InterviewId",
                table: "InterviewQuestions",
                column: "InterviewId");

            migrationBuilder.CreateIndex(
                name: "IX_InterviewQuestions_QuestionId",
                table: "InterviewQuestions",
                column: "QuestionId");

            migrationBuilder.CreateIndex(
                name: "IX_InterviewReports_InterviewId",
                table: "InterviewReports",
                column: "InterviewId",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_Interviews_CreatedAt",
                table: "Interviews",
                column: "CreatedAt");

            migrationBuilder.CreateIndex(
                name: "IX_Interviews_InterviewId",
                table: "Interviews",
                column: "InterviewId",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_Interviews_UserId_Status",
                table: "Interviews",
                columns: new[] { "UserId", "Status" });

            migrationBuilder.CreateIndex(
                name: "IX_UserProgress_MasteryLevel",
                table: "UserProgress",
                column: "MasteryLevel");

            migrationBuilder.CreateIndex(
                name: "IX_UserProgress_UserId_SkillOrTopic_Type",
                table: "UserProgress",
                columns: new[] { "UserId", "SkillOrTopic", "Type" },
                unique: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "Certificates");

            migrationBuilder.DropTable(
                name: "InterviewAnswers");

            migrationBuilder.DropTable(
                name: "InterviewReports");

            migrationBuilder.DropTable(
                name: "UserProgress");

            migrationBuilder.DropTable(
                name: "InterviewQuestions");

            migrationBuilder.DropTable(
                name: "Interviews");
        }
    }
}
