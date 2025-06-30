import { applyNERText, applyNERFile } from "../../services/api.js";

let buttonApplyNER = document.getElementById("buttonApplyNER")
let textarea = document.getElementById("userText")
let upload = document.getElementById("uploadForm")
let input = document.getElementById("fileInput")
let buttonExportsResults = document.getElementById("exportResults")

let tokens = undefined
let labels = undefined

//check if text is in textarea
textarea.addEventListener('input', () =>{
    if (textarea.value.trim().length >0){
        buttonApplyNER.disabled = false;
    } else{
        buttonApplyNER.disabled = true;
    }
})

buttonApplyNER.onclick = () =>{
    applyNERText(textarea.value)
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

    let res = await applyNERFile(formData)
    tokens = res[0];
    labels = res[1];
    print(tokens)
    buttonExportsResults.disabled = false;
})

function createExportFile(){
    let exportFile = ""
    for(let i =0; i < tokens.length; i++){
        for(let u =0; u < tokens[i].length; u++){
            exportFile += `${tokens[i]}\t${labels[i]}\n`;
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
}

buttonExportsResults.onclick = createExportFile