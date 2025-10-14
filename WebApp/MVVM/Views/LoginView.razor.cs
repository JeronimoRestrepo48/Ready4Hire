using Microsoft.AspNetCore.Components;
using Ready4Hire.MVVM.ViewModels;

namespace Ready4Hire.MVVM.Views
{
    public partial class LoginView : ComponentBase
    {
        private LoginViewModel vm = new LoginViewModel();

        private bool showRegisterModal = false;
        private int Step = 1;

        private string hardSkillSearch = "";
        private string softSkillSearch = "";
        private string interestSearch = "";

        // Step 1 registration fields
        private string registerEmail = "";
        private string registerPassword = "";
        private string registerConfirmPassword = "";

        // Step 2 registration fields
        private string registerName = "";
        private string registerLastName = "";
        private string registerCountry = "";
        private string registerJob = "";

        // Validation state step 1
        private bool isEmailInvalid = false;
        private bool isPasswordInvalid = false;
        private bool isConfirmPasswordInvalid = false;

        // Validation state step 2
        private bool isNameInvalid = false;
        private bool isLastNameInvalid = false;
        private bool isCountryInvalid = false;
        private bool isJobInvalid = false;

        // Validation state step 3
        private bool isHardskillsInvalid = false;
        private bool isSoftskillsInvalid = false;
        private bool isInterestsInvalid = false;

        #region Listas de habilidades e intereses
        private List<string> hardSkills = new() { "C#", "Java", "SQL", "JavaScript", "Python", "Docker", "Kubernetes" };
        private List<string> softSkills = new() { "Comunicación", "Trabajo en equipo", "Liderazgo", "Adaptabilidad", "Pensamiento crítico" };
        private List<string> interests = new() { "Inteligencia Artificial", "Desarrollo Web", "Videojuegos", "Ciberseguridad", "Cloud Computing" };
        #endregion

        private List<string> filteredHardSkills = new();
        private List<string> filteredSoftSkills = new();
        private List<string> filteredInterests = new();

        private List<string> selectedHardSkills = new();
        private List<string> selectedSoftSkills = new();
        private List<string> selectedInterests = new();

        void ShowRegisterModal()
        {
            showRegisterModal = true;
            Step = 1;
            ResetSearch();
            ResetValidation();
            registerEmail = "";
            registerPassword = "";
            registerConfirmPassword = "";
            registerName = "";
            registerLastName = "";
            registerCountry = "Colombia";
            registerJob = "";
        }

        void NextStep()
        {
            if (Step == 1)
            {
                // Validate email and password
                isEmailInvalid = !vm.ValidateEmail(registerEmail);
                isPasswordInvalid = !vm.ValidatePassword(registerPassword);
                isConfirmPasswordInvalid = registerPassword != registerConfirmPassword;

                if (isEmailInvalid || isPasswordInvalid || isConfirmPasswordInvalid)
                    return;
            }
            else if (Step == 2)
            {
                // Validate name and last name
                isNameInvalid = !vm.ValidateString(registerName);
                isLastNameInvalid = !vm.ValidateString(registerLastName);
                isCountryInvalid = !vm.ValidateString(registerCountry);
                isJobInvalid = !vm.ValidateString(registerJob);

                if (isNameInvalid || isLastNameInvalid || isCountryInvalid || isJobInvalid)
                    return;
            }
            if (Step == 3)
            {
                // Validate skills and interests
                isHardskillsInvalid = selectedHardSkills.Count == 0;
                isSoftskillsInvalid = selectedSoftSkills.Count == 0;
                isInterestsInvalid = selectedInterests.Count == 0;

                // Prevent continue if any required selection is missing
                if (isHardskillsInvalid || isSoftskillsInvalid || isInterestsInvalid)
                    return;
            }
            if (Step < 3)
                Step++;
            ResetSearch();
            ResetValidation();
        }

        void HideRegisterModal()
        {
            showRegisterModal = false;
            Step = 1;
            ResetSearch();
            ResetValidation();
        }

        void ResetValidation()
        {
            isEmailInvalid = false;
            isPasswordInvalid = false;
            isConfirmPasswordInvalid = false;
            isNameInvalid = false;
            isLastNameInvalid = false;
            isCountryInvalid = false;
            isJobInvalid = false;
            isHardskillsInvalid = false;
            isSoftskillsInvalid = false;
            isInterestsInvalid = false;
        }

        #region MetodosPaso3

        void ResetSearch()
        {
            hardSkillSearch = "";
            softSkillSearch = "";
            interestSearch = "";
            filteredHardSkills.Clear();
            filteredSoftSkills.Clear();
            filteredInterests.Clear();
        }

        void FilterHardSkills(ChangeEventArgs e)
        {
            hardSkillSearch = e.Value?.ToString() ?? "";
            filteredHardSkills = hardSkills
                .Where(s => s.Contains(hardSkillSearch, StringComparison.OrdinalIgnoreCase) && !selectedHardSkills.Contains(s))
                .ToList();
        }

        void FilterSoftSkills(ChangeEventArgs e)
        {
            softSkillSearch = e.Value?.ToString() ?? "";
            filteredSoftSkills = softSkills
                .Where(s => s.Contains(softSkillSearch, StringComparison.OrdinalIgnoreCase) && !selectedSoftSkills.Contains(s))
                .ToList();
        }

        void FilterInterests(ChangeEventArgs e)
        {
            interestSearch = e.Value?.ToString() ?? "";
            filteredInterests = interests
                .Where(s => s.Contains(interestSearch, StringComparison.OrdinalIgnoreCase) && !selectedInterests.Contains(s))
                .ToList();
        }

        void AddHardSkill(string skill)
        {
            selectedHardSkills.Add(skill);
            hardSkillSearch = "";
            filteredHardSkills.Clear();
        }

        void RemoveHardSkill(string skill)
        {
            selectedHardSkills.Remove(skill);
        }

        void AddSoftSkill(string skill)
        {
            selectedSoftSkills.Add(skill);
            softSkillSearch = "";
            filteredSoftSkills.Clear();
        }

        void RemoveSoftSkill(string skill)
        {
            selectedSoftSkills.Remove(skill);
        }

        void AddInterest(string interest)
        {
            selectedInterests.Add(interest);
            interestSearch = "";
            filteredInterests.Clear();
        }

        void RemoveInterest(string interest)
        {
            selectedInterests.Remove(interest);
        }
        #endregion

    }
}
