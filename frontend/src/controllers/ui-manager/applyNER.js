import { applyNERText, applyNERFile } from "../../services/api.js";

let buttonApplyNER = document.getElementById("buttonApplyNER")
let textarea = document.getElementById("userText")
let upload = document.getElementById("uploadForm")
let input = document.getElementById("fileInput")
let buttonExportsResults = document.getElementById("exportResults")
let labelNERState = document.getElementById("nerState")

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
    showEntities(res[0],res[1])
    console.log(res)
}

function showEntities(tokens, labels){
    textarea.innerHTML = ""
    for(let i = 0; i < labels.length;i++){
        for(let u = 0; u < labels[i].length; u++)
        if (labels[i][u] != "O"){
            let span = document.createElement("span");
            span.textContent = tokens[i][u] + " "
            span.style.backgroundColor = "yellow";
            textarea.appendChild(span);
        } else{
            let text = document.createTextNode(tokens[i][u] + " ")
            textarea.appendChild(text);
        }
        textarea.appendChild(document.createElement("br"));
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