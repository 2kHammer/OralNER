import { uploadTrainingData } from "../../services/api.js"

let datasetNameInput = document.getElementById("datasetNameInput")
let datasetFileInput = document.getElementById("datasetFileInput")
let datasetUploadButton = document.getElementById("datasetUploadButton")
let datasetUploadForm = document.getElementById("datasetUploadForm")
let datasetUploadStatusLabel = document.getElementById("datasetUploadStatusLabel")
let datasetName = ""
let fileIsThere = false;

/**
 * Disables/enables the `datasetUploadButton` depending on the dataset name and the uploaded file
 */
function checkSetButtonActive(){
    if ((datasetName.length > 0)&& fileIsThere){
        datasetUploadButton.disabled = false;
    } else{
        datasetUploadButton.disabled = true;
    }
}

/**
 * Resets the upload dataset form
 * @param {bool} withLabel - should the `datasetUploadStatusLabel` also be reset 
 */
export function resetUploadDataset(withLabel=true){
    fileIsThere = true;
    datasetUploadForm.reset();
    if(withLabel){
        datasetUploadStatusLabel.textContent = ""
    }
}

/**
 * Dis/enable the file input, the dataset name input and the upload button
 * @param {boolean} disable 
 */
function disableEnableForm(disable){
     datasetFileInput.disabled = disable;
    datasetNameInput.disabled = disable;
    datasetUploadButton.disabled=disable;
}


/*
 * Event Listeners and applied functions 
 */

// Handling of the dataset name input
datasetNameInput.addEventListener('input', (event)=>{
    datasetName = datasetNameInput.value;
    checkSetButtonActive();
})

// Handling of the file input
datasetFileInput.addEventListener('input', (event)=>{
    let file = datasetFileInput.files[0];
    if (file){
        let filename = file.name.toLowerCase()
        if(filename.endsWith('.csv')){
            fileIsThere = true;
        } else {
            resetUploadDataset();
            alert("Nur csv-Dateien sind möglich");
        }
    }
    checkSetButtonActive();
})

// Handling of upload Button
datasetUploadButton.addEventListener("click", async () =>{
    datasetUploadStatusLabel.textContent = "Hochladen begonnen";
    disableEnableForm(true);
    if (await uploadTrainingData(datasetFileInput.files[0], datasetName)){
        datasetUploadStatusLabel.textContent = "Hochladen bzw. Konvertieren erfolgreich"
    } else{
        datasetUploadStatusLabel.textContent = "Hochladen bzw. Konvertieren nicht erfolgreich"
    }
    disableEnableForm(false);
    resetUploadDataset(false)
})