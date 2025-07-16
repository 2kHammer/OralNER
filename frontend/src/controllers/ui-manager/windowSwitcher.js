import  {getActiveModel} from "../../services/api.js"
import { initComparisonWindow } from "./modelComparison.js";
import { initFinetuningWindow } from "./finetuneModel.js";
import { resetUploadDataset } from "./uploadDataset.js";

async function switchWindow(id) {
    // hide every window
    document.querySelectorAll('.window').forEach(div => {
        div.classList.remove('active')
    });
    // view selected window
    const active = document.getElementById(id);
    if (active) {
      active.classList.add('active');
    }

    if (id == "home"){
      initHome();
      resetUploadDataset();
    } else if (id == "finetuning"){
      await initFinetuningWindow();
    } else if (id == "comparison"){
      await initComparisonWindow();
    }
  }

async function initHome(){
  //set active Model Name
  let activeModel = await getActiveModel()
  let activeModelName = activeModel.name;
  let activeModelText = ""
  if (activeModelName != undefined){
    activeModelText = activeModelName
  }
  document.getElementById("actualModel").innerText = activeModelText
}

//global 
window.switchWindow = switchWindow

//test if server is available
try{
  await getActiveModel();
  
  //first init Home, if no error in server connection
  initHome();
} catch(exception){
  await switchWindow("noserver")
}
  