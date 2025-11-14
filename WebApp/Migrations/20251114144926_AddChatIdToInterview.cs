using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace Ready4Hire.Migrations
{
    /// <inheritdoc />
    public partial class AddChatIdToInterview : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<int>(
                name: "ChatId",
                table: "Interviews",
                type: "integer",
                nullable: true);

            migrationBuilder.CreateIndex(
                name: "IX_Interviews_ChatId_Status",
                table: "Interviews",
                columns: new[] { "ChatId", "Status" });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "IX_Interviews_ChatId_Status",
                table: "Interviews");

            migrationBuilder.DropColumn(
                name: "ChatId",
                table: "Interviews");
        }
    }
}
