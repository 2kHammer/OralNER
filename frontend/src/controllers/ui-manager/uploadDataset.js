import { uploadTrainingData } from "../../services/api.js"

let datasetNameInput = document.getElementById("datasetNameInput")
let datasetFileInput = document.getElementById("datasetFileInput")
let datasetUploadButton = document.getElementById("datasetUploadButton")
let datasetUploadForm = document.getElementById("datasetUploadForm")
let datasetUploadStatusLabel = document.getElementById("datasetUploadStatusLabel")
let datasetName = ""
let fileIsThere = false;

function checkSetButtonActive(){
    if ((datasetName.length > 0)&& fileIsThere){
        datasetUploadButton.disabled = false;
    } else{
        datasetUploadButton.disabled = true;
    }
}

export function resetUploadDataset(withLabel=true){
    fileIsThere = true;
    datasetUploadForm.reset();
    if(withLabel){
        datasetUploadStatusLabel.textContent = ""
    }
}

function disableEnableForm(disable){
     datasetFileInput.disabled = disable;
    datasetNameInput.disabled = disable;
    datasetUploadButton.disabled=disable;
}

datasetNameInput.addEventListener('input', (event)=>{
    datasetName = datasetNameInput.value;
    checkSetButtonActive();
})

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

datasetUploadButton.addEventListener("click", async () =>{
    datasetUploadStatusLabel.textContent = "Hochladen begonnen";
    disableEnableForm(true);
    if (await uploadTrainingData(datasetFileInput.files[0], datasetName)){
        datasetUploadStatusLabel.textContent = "Hochladen erfolgreich"
    } else{
        datasetUploadStatusLabel.textContent = "Hochladen nicht erfolgreich"
    }
    disableEnableForm(false);
    resetUploadDataset(false)
})