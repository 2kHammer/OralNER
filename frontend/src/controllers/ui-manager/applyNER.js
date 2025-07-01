import { applyNERText, applyNERFile } from "../../services/api.js";

const colors = [
  "#e6194b", // Rot
  "#3cb44b", // Grün
  "#ffe119", // Gelb
  "#0082c8", // Blau
  "#f58231", // Orange
  "#911eb4", // Violett
  "#46f0f0", // Türkis
  "#f032e6"  // Pink
];

let buttonApplyNER = document.getElementById("buttonApplyNER")
let textarea = document.getElementById("userText")
let upload = document.getElementById("uploadForm")
let input = document.getElementById("fileInput")
let buttonExportsResults = document.getElementById("exportResults")
let labelNERState = document.getElementById("nerState")
let entityLegend = document.getElementById("entityLegend")

let tokens = undefined
let labels = undefined

//check if text is in textarea
textarea.addEventListener('input', () =>{
    if (textarea.innerText.trim().length >0){
        buttonApplyNER.disabled = false;
    } else{
        buttonApplyNER.disabled = true;
    }
})

buttonApplyNER.onclick = async () =>{
    let res = await applyNERText(textarea.innerText)
    visualizeEntities(res[0],res[1])
    console.log(res)
}

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

function createEntityLegend(entityToColor, documentLegend){
    documentLegend.innerHTML = "Legende:"
    for(const [label, color] of Object.entries(entityToColor)){
        let li = document.createElement("li");
        li.style.display = "flex";
        li.style.alignItems = "center";
        li.style.marginBottom = "5px";

        let colorBox = document.createElement("span");
        colorBox.style.display = "inline-block";
        colorBox.style.width = "20px";
        colorBox.style.height = "20px";
        colorBox.style.marginRight = "8px";
        colorBox.style.borderRadius = "3px";
        colorBox.style.backgroundColor = color;

        li.appendChild(colorBox)
        li.appendChild(document.createTextNode(label))
        documentLegend.appendChild(li)
    }
}

/*
 * Muss dies noch abändern das es über einen Job läuft und nachgefragt wird, ob verfügbar
 */

//listener if file is uploaded
upload.addEventListener('submit', async(e) => {
    e.preventDefault();
    const file = input.files[0];

    if (!file) {
      alert("Bitte eine Datei auswählen");
      return;
    }
  
    const formData = new FormData();
    formData.append("file", file);

    labelNERState.innerHTML = "NER wird angewandt";
    labelNERState.style.color = "red"
    let res = await applyNERFile(formData)
    labelNERState.innerHTML = `NER abgeschlossen - F1: ${res[2].f1.toFixed(2)}, Precision: ${res[2].precision.toFixed(2)}, Recall: ${res[2].recall.toFixed(2)}, Genauigkeit: ${res[2].accuracy.toFixed(2)}`;
    labelNERState.style.color = "green"
    tokens = res[0];
    labels = res[1];
    buttonExportsResults.disabled = false;
})

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
    buttonExportsResults.disabled = true;
}

buttonExportsResults.onclick = createExportFile