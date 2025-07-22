import { applyNERText, applyNERFile, getNERResults } from "../../services/api.js";

const colors = [
  "#e6194b", // red
  "#3cb44b", // green
  "#ffe119", // yellow
  "#0082c8", // blue
  "#f58231", // orange
  "#911eb4", // violet 
  "#46f0f0", // turquoise
  "#f032e6",  // pink
  // only 8 possible entity types, rest is ssubstitue
  "#d2f53c", // lime green
  "#fabebe", // light pink
  "#008080", // dark turquoise
  "#e6beff"  // light violet
];

let buttonApplyNER = document.getElementById("buttonApplyNER")
let textarea = document.getElementById("userText")
let upload = document.getElementById("uploadForm")
let input = document.getElementById("fileInput")
let buttonExportsResults = document.getElementById("exportResults")
let labelNERState = document.getElementById("nerState")
let entityLegend = document.getElementById("entityLegend")
let applyNERTextStatus = document.getElementById("applyNERTextStatus")
let splitSentencesCheckbox = document.getElementById("splitSentencesCheckbox")
let buttonUploadNERFile = document.getElementById("buttonUploadNERFile")

let tokens = undefined
let labels = undefined

let timerApplyNER = undefined
let intervallDuration = 3 * 1000;


/**
 * Visualize the extracted entities in the text field `textarea`
 * @param {string[]} tokens 
 * @param {string[]} labels 
 */
function visualizeEntities(tokens, labels){
    let entityToColor = {}
    let colorIndex = 0
    textarea.innerHTML = ""
    for(let i = 0; i < labels.length;i++){
        for(let u = 0; u < labels[i].length; u++)
        if (labels[i][u] != "O"){
            let entity = labels[i][u].slice(2)
            //check if entity in dict
            let color = undefined
            if (!entityToColor.hasOwnProperty(entity)){
                entityToColor[entity] = colors[colorIndex]
                colorIndex++;
            }
            color = entityToColor[entity]
            let span = document.createElement("span");
            span.textContent = tokens[i][u] + " "
            span.style.backgroundColor = color;
            textarea.appendChild(span);
        } else{
            let text = document.createTextNode(tokens[i][u] + " ")
            textarea.appendChild(text);
        }
        textarea.appendChild(document.createElement("br"));
    }
    createEntityLegend(entityToColor, entityLegend)
}

/**
 * Creates the legend that shows, which color stands for which entity type
 * @param {Object} entityToColor - with the entity types as proberties and the related colors
 * @param {HTMLElement} documentLegend - elem which should show the legend
 */
function createEntityLegend(entityToColor, documentLegend){
    documentLegend.innerHTML = "Legende:";
    documentLegend.classList.add("legend")

    for(const [label, color] of Object.entries(entityToColor)){
        let li = document.createElement("div");
        li.classList.add("legend-element")

        let colorBox = document.createElement("span");
        colorBox.classList.add("legend-element-color")
        colorBox.style.backgroundColor = color;

        li.appendChild(colorBox);
        li.appendChild(document.createTextNode(label));
        documentLegend.appendChild(li);
    }
}

/**
 * Resets the legend and the textfield (`textarea`)
 */
function resetVisualizingEntities(){
    entityLegend.innerHTML = "";
    textarea.innerHTML = "";
}

/**
 * Dis/enables the html elements for applying NER to a file
 * @param {boolean} disable 
 */
function disEnableApplyNERFile(disable){
    buttonExportsResults.disabled = disable;
    buttonUploadNERFile.disabled = disable;
    textarea.contentEditable = "false";
    if (disable == false){
        textarea.contentEditable = "true"
        //only enable if text is in textartea
        if (textarea.innerHTML != ""){
            buttonApplyNER.disabled = disable;
        }
    } else {
        textarea.contentEditable = "false"
        buttonApplyNER.disabled = disable;
    }
}

/**
 * Dis/enables the html elements for applying NER to text in `textarea`
 * @param {boolean} disable 
 */
function disEnableApplyNERText(disable){
    let elements = upload.elements;
    buttonApplyNER.disabled = disable;
    for (let i = 0; i < elements.length; i++){
        if((elements[i] == buttonExportsResults) && !disable){
            console.log(tokens)
            if (tokens != undefined && labels != undefined){
                elements[i].disabled = disable;
            } 
        } else{
            //disable or enable all elements
            elements[i].disabled = disable;
        }
    }
}

/**
 * Creating the export file if NER has been applied to a file
 */
function createExportFile(){
    let exportFile = ""
    for(let i =0; i < tokens.length; i++){
        for(let u =0; u < tokens[i].length; u++){
            exportFile += `${tokens[i][u]}\t${labels[i][u]}\n`;
        }
    }
    const blob = new Blob([exportFile], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'ner_output.txt'; 
    document.body.appendChild(a);  
    a.click();
    document.body.removeChild(a); 
    URL.revokeObjectURL(url);
    upload.reset();
    tokens = undefined
    labels = undefined
    buttonExportsResults.disabled = true;
}

/**
 * Starts the intevall for checking if NER is finished
 * @param {boolean} withLabels - true if NER was applied on a file, else false
 */
async function startApplyNERInterval(withLabels){
    timerApplyNER = setInterval(async ()=> {
                let reset = await checkNERResults(withLabels);
                if (reset){
                    clearInterval(timerApplyNER);
                    localStorage.removeItem("job_id")
                }
            }, intervallDuration)
}

/**
 * Check if the NER-Job is finished and the resuls are available
 * @param {boolean} withLabels - true if NER was applied on a file, else false
 * @returns {boolean} - true: if the job is finished (whether successful or not), else false 
 */
async function checkNERResults(withLabels){ 
    let jobId = localStorage.getItem("job_id")
    let reset = false;
    if (jobId){
        let res = await getNERResults(jobId)
        let state = res["state"]
        if(withLabels){
            reset = handleNERResultsFile(state,res)
        } else {
            reset = handleNERResultsText(state, res);
        }
    return reset;
    } else {
        console.error("No JobId")
    }
}

/**
 * Is responsible for displaying the respective status of the NER execution on a file
 * @param {boolean} state - is the NER-job finished
 * @param {{state: boolean, result: {string[],string[],object}}} res - the ner results from the backend
 * @returns {boolean} - true: if the job is finished (whether successful or not), else false 
 */
function handleNERResultsFile(state, res){
    let reset = true;
    if(state){
        let result = res["result"][2]
        labelNERState.innerHTML = `NER abgeschlossen - F1: ${result["f1"]}, Precision: ${result["precision"]}, Recall: ${result["recall"]}, Genauigkeit: ${result["accuracy"]}`;
        labelNERState.style.color = "green"
        tokens = res["result"][0];
        labels = res["result"][1];
        buttonExportsResults.disabled = false;
        disEnableApplyNERFile(false);
    } else if (state == false){
        labelNERState.innerHTML = "NER wird durchgeführt"
        labelNERState.style.color = "red"
        reset = false;
    } else{
        labelNERState.innerHTML = "Fehlerhafte JobId"
        disEnableApplyNERFile(false);
    }
    return reset;
}

/**
 * Is responsible for displaying the respective status of the NER execution on text
 * @param {boolean} state - is the NER-job finished
 * @param {{state: boolean, result: {string[],string[],object}}} res - the ner results from the backend
 * @returns {boolean} - - true: if the job is finished (whether successful or not), else false 
 */
function handleNERResultsText(state, res){
    let reset = true;
    if (state == true){
        let result = res["result"]
        visualizeEntities(result[0], result[1])
        applyNERTextStatus.textContent = "";
        disEnableApplyNERText(false);
    } else if (state == false) {
        applyNERTextStatus.textContent = "NER wird durchgeführt"
        applyNERTextStatus.style.color ="red"
        reset = false;
    } else {
        applyNERTextStatus.textContent = "Fehlerhafte JobId"
        disEnableApplyNERText(false);
    }
    return reset;
}

/**
 * Checks after reload if a NER-job is running, displays the corresponging elements
 */
async function checkIfNerIsRunning() {
    let jobId = localStorage.getItem("job_id")
    if (jobId){
        let labels = localStorage.getItem("labels")
        if (labels == "false"){
            startApplyNERInterval(false)
            disEnableApplyNERText(true);
        } else {
            startApplyNERInterval(true)
            disEnableApplyNERFile(true)
        }
    }
    
}


/*
 * Event Listeners and applied functions 
 */

/*
 * handles the upload for a file and starting the NER-job on it
 */
upload.addEventListener('submit', async(e) => {
    e.preventDefault();
    const file = input.files[0];

    if (!file) {
      alert("Bitte eine csv-Datei auswählen");
      return;
    }
    if (!file.name.endsWith(".csv")){
        alert("Bitte eine csv-Datei auswählen")
        return;
    }
  
    const formData = new FormData();
    formData.append("file", file);

    resetVisualizingEntities();
    let splitSentences = splitSentencesCheckbox.checked;
    formData.append("split_sentences",splitSentences)
    let jobId = await applyNERFile(formData)
    if (jobId == undefined){
            labelNERState.textContent = "Fehler beim Starten von NER"
        } else{
            localStorage.setItem("job_id",jobId)
            localStorage.setItem("labels", true)
            startApplyNERInterval(true);
            disEnableApplyNERFile(true);
        }
})

/*
 * activates/deactivates the button depending on whether there is text in the input field
 */
textarea.addEventListener('input', () =>{
    if (textarea.innerText.trim().length >0){
        buttonApplyNER.disabled = false;
    } else{
        buttonApplyNER.disabled = true;
    }
})

/*
 * handles and start the NER-job for text
 */
buttonApplyNER.onclick = async () =>{
    if (!localStorage.getItem("job_id")){
        let jobId = await applyNERText(textarea.innerText)
        if (jobId == undefined){
            applyNERTextStatus.textContent = "Fehler beim Starten von NER"
        } else{
            //reset Results from file
            labelNERState.innerHTML = ""

            localStorage.setItem("job_id",jobId)
            localStorage.setItem("labels",false)
            startApplyNERInterval(false);
            disEnableApplyNERText(true);
        }
    }
}


buttonExportsResults.onclick = createExportFile

//check if NER is running after reload
checkIfNerIsRunning();

