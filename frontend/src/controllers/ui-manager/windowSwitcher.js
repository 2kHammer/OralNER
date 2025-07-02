import { getActiveModelname} from "../service-manager/modelManagement.js";
import { initComparisonWindow } from "./modelComparison.js";
import { initFinetuningWindow } from "./finetuneModel.js";

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
    } else if (id == "finetuning"){
      await initFinetuningWindow();
    } else if (id == "comparison"){
      await initComparisonWindow();
    }
  }

async function initHome(){
  //set active Model Name
  let activeModelName = await getActiveModelname()
  let activeModelText = "Keine Server Verbindung"
  if (activeModelName != undefined){
    activeModelText = "Aktuelles Modell: "  + activeModelName
  }
  document.getElementById("actualModel").innerText = activeModelText
}

//global 
window.switchWindow = switchWindow
//first init Home
initHome();
  