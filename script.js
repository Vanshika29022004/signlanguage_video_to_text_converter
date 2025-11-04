const form = document.getElementById("uploadForm");
const videoInput = document.getElementById("videoInput");
const previewVideo = document.getElementById("previewVideo");
const resultText = document.getElementById("resultText");

videoInput.addEventListener("change", () => {
    const file = videoInput.files[0];
    if (file) {
        const url = URL.createObjectURL(file);
        previewVideo.src = url;
        previewVideo.style.display = "block";
    }
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = videoInput.files[0];
    if (!file) {
        alert("Please select a video file first!");
        return;
    }

    resultText.textContent = "Processing... ⏳";

    // Dummy output for now
    setTimeout(() => {
        resultText.textContent = "Hello (Predicted Text)";
    }, 3000);

    // Later replace with Flask backend fetch
});
