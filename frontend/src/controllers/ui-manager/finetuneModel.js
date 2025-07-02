import {createTable, createModelTableVals, modelColumns} from "../ui-manager/modelComparison.js"
import { getModel, getModels, getTrainingsData, startFinetune} from "../../services/api.js";

const keyModifiedModelId = "modifiedModelId"
const datasetColumns = ['ID', 'Name','Datum']

let selectedModelId = -1;
let selectedDatasetId = -1;
let modifiedModelName = "";
//let modifiedModelId = -1;

let timerId = undefined
let intervallDuration = 10 *1000;

let finetuningContainer = document.getElementById("finetuningContainer")
let modelSelectionContainer = document.getElementById("modelSelectionContainer")
let datasetSelectionContainer = document.getElementById("datasetSelectionContainer")
let buttonFinetuneModel = document.getElementById("buttonFinetuneModel")
let modelNameInput = document.getElementById("modelNameInput")
let statusHeading = document.getElementById("statusHeading")

function handleClickModelComparison(modelId){
    selectedModelId = modelId;
    console.log('Ausgewähltes Modell ID:', selectedModelId);
    checkButtonFinetune();
}

function handleClickDatasetComparison(datasetId){
    selectedDatasetId = datasetId;
    console.log("Ausgewählter Datensatz:", selectedDatasetId);
    checkButtonFinetune();
}

function createDatasetTableVals(datasets){
    let datasetVals = []
    datasets.forEach(dataset =>{
        datasetVals.push([dataset.id,dataset.name,dataset.upload_date])
    })
    return datasetVals;
}

function checkButtonFinetune(){
    if (selectedDatasetId != -1 && selectedModelId != -1 && modifiedModelName != ""){
        buttonFinetuneModel.disabled = false;
    } else {
        buttonFinetuneModel.disabled = true;
    }
}

async function finetune(){
    let res = await startFinetune(selectedModelId, selectedDatasetId, modifiedModelName)
    if (res != undefined){
        let modelId = res["modified_model_id"]
        localStorage.setItem(keyModifiedModelId, modelId)
        startCheckIfModelIsInFinetuning();
    }
}

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
        }
    } else{
        statusHeading.innerHTML = ""
    }
}

function startCheckIfModelIsInFinetuning(){
    timerId = setInterval(checkIfModelIsInFinetuning, intervallDuration)
}

function endCheckIfModelIsInFinetuning(){
   clearInterval(timerId) 
}

export async function initFinetuningWindow() {
    let models = await getModels();
    if (models != undefined){
        let modelVals = createModelTableVals(models)
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
}

modelNameInput.addEventListener("input", ()=>{
    modifiedModelName = modelNameInput.value;
    console.log(modifiedModelName)
    checkButtonFinetune();
})

buttonFinetuneModel.onclick = finetune
await initFinetuningWindow();






