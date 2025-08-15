import {createTable, createModelTableVals, modelColumns} from "../ui-manager/modelComparison.js"
import { getModel, getModels, getTrainingsData, startFinetune} from "../../services/api.js";

const keyModifiedModelId = "modifiedModelId"
const datasetColumns = ['ID', 'Name','Datum']

let selectedModelId = -1;
let selectedDatasetId = -1;
let modifiedModelName = "";
//let modifiedModelId = -1;

let timerId = undefined
let intervallDuration = 5 *1000;

let finetuningContainer = document.getElementById("finetuningContainer")
let modelSelectionContainer = document.getElementById("modelSelectionContainer")
let datasetSelectionContainer = document.getElementById("datasetSelectionContainer")
let buttonFinetuneModel = document.getElementById("buttonFinetuneModel")
let modelNameInput = document.getElementById("modelNameInput")
let statusHeading = document.getElementById("statusHeading")
let splitSentencesFinetuningCheckbox = document.getElementById("splitSentencesFinetuningCheckbox")

/**
 * Saves the selected model id of the model table in `modelId`
 * @param {number} modelId - selected id
 */
function handleClickModelComparison(modelId){
    selectedModelId = modelId;
    console.log('Ausgewähltes Modell ID:', selectedModelId);
    checkButtonFinetune();
}

/**
 * Saves the selected dataset id of the dataset table in `modelId`
 * @param {number} datasetId - selected id
 */
function handleClickDatasetComparison(datasetId){
    selectedDatasetId = datasetId;
    console.log("Ausgewählter Datensatz:", selectedDatasetId);
    checkButtonFinetune();
}

/**
 * Creates the data for createTable from the dataset metadata `datasets`
 * @param {Object[]} datasets - datasets metadata from the backend 
 * @returns {string[][]} - data in order for the table
 */
function createDatasetTableVals(datasets){
    let datasetVals = []
    datasets.forEach(dataset =>{
        datasetVals.push([dataset.id,dataset.name,dataset.upload_date])
    })
    return datasetVals;
}

/**
 * Checks whether fine-tuning can be started and then activates or deactivates the button
 */
function checkButtonFinetune(){
    if (selectedDatasetId != -1 && selectedModelId != -1 && modifiedModelName != ""){
        buttonFinetuneModel.disabled = false;
    } else {
        buttonFinetuneModel.disabled = true;
    }
}

/**
 * Start the finetuning-job
 */
async function finetune(){
    let split = splitSentencesFinetuningCheckbox.checked;
    let modelId = await startFinetune(selectedModelId, selectedDatasetId, modifiedModelName, split)
    if (modelId != undefined){
        localStorage.setItem(keyModifiedModelId, modelId)
        startCheckIfModelIsInFinetuning();
        buttonFinetuneModel.disabled = true; 
    }
}

/**
 * Checks if a finetuning job has already been started or was finished
 */
async function checkIfModelIsInFinetuning(){
    let modifiedModelId = localStorage.getItem(keyModifiedModelId)
    if (modifiedModelId){
        let res = await getModel(modifiedModelId)
        if (res["state"] == "IN_TRAINING"){
            statusHeading.innerHTML = "Das Feinanpassen von Model " + modifiedModelId + " wurde bereits gestartet";
            statusHeading.style.color = "red"
            finetuningContainer.classList.add("window")
            buttonFinetuneModel.disabled = true;
        } else{
            statusHeading.innerHTML = "Model " + modifiedModelId + " ist fertig feinangepasst"
            statusHeading.style.color = "green"
            finetuningContainer.classList.remove("window")
            localStorage.removeItem(keyModifiedModelId)
            endCheckIfModelIsInFinetuning();
            buttonFinetuneModel.disabled = false;
        }
    } else{
        statusHeading.innerHTML = ""
    }
}

/**
 * Start the interval for checking the finetuning-job
 */
function startCheckIfModelIsInFinetuning(){
    timerId = setInterval(checkIfModelIsInFinetuning, intervallDuration)
}

/**
 * Ends the interval for checking the finetuning-job
 */
function endCheckIfModelIsInFinetuning(){
   clearInterval(timerId) 
}

/**
 * Inits the finetuning window: rertrieves the model and dataset metadata, creates the tables, check if an model is already in finetuning
 */
export async function initFinetuningWindow() {
    try{
        let models = await getModels();
        if (models != undefined){
            let modelVals = createModelTableVals(models, false)
            createTable(modelVals, modelColumns, modelSelectionContainer, handleClickModelComparison)
        }

        let datasets = await getTrainingsData()
        if (datasets != undefined){
            let datasetTableVals = createDatasetTableVals(datasets)
            createTable(datasetTableVals,datasetColumns,datasetSelectionContainer, handleClickDatasetComparison)
        }

        checkIfModelIsInFinetuning()
        if (localStorage.getItem(keyModifiedModelId)){
            startCheckIfModelIsInFinetuning();
        }
    } catch(e){
        console.error("No Server connection for model finetuning, reset local storage")
        //reset the finetuning job if the server connection is lost
        localStorage.clear()

    }
}

/*
 * Event Listeners and applied functions 
 */

/**
 * handles the input for the modified model name
 */
modelNameInput.addEventListener("input", ()=>{
    modifiedModelName = modelNameInput.value;
    console.log(modifiedModelName)
    checkButtonFinetune();
})

buttonFinetuneModel.onclick = finetune
await initFinetuningWindow();






