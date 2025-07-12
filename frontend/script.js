const FASTAPI_CLASSIFY_URL = "/api/classify";
const FASTAPI_RECOMMEND_JOBS_URL = "/api/recommend_jobs";
const FASTAPI_EXTRACT_PDF_URL = "/api/extract_pdf_text";

const radioPaste = document.getElementById("radioPaste");
const radioUpload = document.getElementById("radioUpload");
const pasteTextInput = document.getElementById("pasteTextInput");
const uploadPdfInput = document.getElementById("uploadPdfInput");
const resumeTextArea = document.getElementById("resumeTextArea");
const pdfFileInput = document.getElementById("pdfFileInput");
const analyzeButton = document.getElementById("analyzeButton");
const loadingSpinner = document.getElementById("loadingSpinner");
const messageDisplay = document.getElementById("messageDisplay");
const resultsSection = document.getElementById("resultsSection");
const predictedCategoryDisplay = document.getElementById(
  "predictedCategoryDisplay"
);
const jobList = document.getElementById("jobList");
const fileNameDisplay = document.getElementById("fileName");

let currentResumeText = "";

// Toggle input method
radioPaste.addEventListener("change", () => {
  pasteTextInput.classList.remove("hidden");
  uploadPdfInput.classList.add("hidden");
  resultsSection.classList.add("hidden");
});
radioUpload.addEventListener("change", () => {
  pasteTextInput.classList.add("hidden");
  uploadPdfInput.classList.remove("hidden");
  resultsSection.classList.add("hidden");
});

// Handle PDF upload
pdfFileInput.addEventListener("change", async () => {
  const file = pdfFileInput.files[0];
  if (!file || file.type !== "application/pdf") return;

  fileNameDisplay.textContent = `📎 Selected: ${file.name}`;
  showMessage("info", "⏳ Extracting text from PDF...");
  loadingSpinner.classList.remove("hidden");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(FASTAPI_EXTRACT_PDF_URL, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    currentResumeText = data.extracted_text || "";
    showMessage("success", "✅ Text extracted successfully.");
  } catch (e) {
    showMessage("error", "❌ PDF extraction failed.");
  } finally {
    loadingSpinner.classList.add("hidden");
  }
});

// Analyze Button
analyzeButton.addEventListener("click", async () => {
  currentResumeText = radioPaste.checked
    ? resumeTextArea.value
    : currentResumeText;
  if (!currentResumeText.trim()) {
    showMessage("warning", "⚠️ Please provide resume text.");
    return;
  }

  showMessage("info", "⏳ Analyzing resume...");
  loadingSpinner.classList.remove("hidden");
  resultsSection.classList.add("hidden");

  try {
    const classifyRes = await fetch(FASTAPI_CLASSIFY_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resume_content: currentResumeText }),
    });

    const classifyData = await classifyRes.json();
    const predictedField = classifyData.category;
    const experience = classifyData.years_experience;

    predictedCategoryDisplay.textContent = `Predicted Category: ${predictedField} | Experience: ${experience}`;

    const jobRes = await fetch(FASTAPI_RECOMMEND_JOBS_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        category: predictedField,
        candidate_years_experience: experience,
      }),
    });

    const jobs = await jobRes.json();
    jobList.innerHTML = "";
    resultsSection.classList.remove("hidden");

    if (jobs.length > 0) {
      jobs.forEach((job) => {
        const jobCard = document.createElement("div");
        jobCard.className = "job-card";
        jobCard.innerHTML = `
          <h4>${job.title}</h4>
          <p><strong>Company:</strong> ${job.company}</p>
          <p><strong>Location:</strong> ${job.location}</p>
          <p><strong>Description:</strong> ${job.description.slice(
            0,
            150
          )}...</p>
          <p><strong>Skills:</strong> ${job.required_skills}</p>
          <p><a href="${job.url}" target="_blank">View Job Details ➡️</a></p>
        `;
        jobList.appendChild(jobCard);
      });
    } else {
      showMessage("info", "ℹ️ No job recommendations available.");
    }
  } catch (err) {
    showMessage("error", `❌ Error: ${err.message}`);
  } finally {
    loadingSpinner.classList.add("hidden");
  }
});

function showMessage(type, message) {
  messageDisplay.innerHTML = `<div class="message ${type}">${message}</div>`;
}
document.getElementById("themeSwitcher").addEventListener("change", () => {
    document.body.classList.toggle("dark-mode");
    document.body.classList.toggle("light-mode");
});