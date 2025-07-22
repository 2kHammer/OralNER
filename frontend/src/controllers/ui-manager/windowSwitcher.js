import  {getActiveModel} from "../../services/api.js"
import { initComparisonWindow } from "./modelComparison.js";
import { initFinetuningWindow } from "./finetuneModel.js";
import { resetUploadDataset } from "./uploadDataset.js";

/**
 * Switches to the window with `id`
 * @param {string} id - the id of the window to switch
 */
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

/**
 * Init the home window, check if a backend connection is possible
 */
async function initHome(){
  //set active Model Name
  let activeModel = await getActiveModel()
  let activeModelText = ""
  if (activeModel != undefined){
    activeModelText = getActiveModelText(activeModel)
  }
  document.getElementById("actualModel").innerHTML = activeModelText
}

/**
 * Returns a description of the current active model
 * @param {Object} model - the metadata for the current model
 * @returns {string}
 */
function getActiveModelText(model){
  let name = model.name;
  let framework = model.framework_name;
  if(model.trainings.length >0){
    let training = model.trainings[model.trainings.length-1]
    return `${name} <strong>|</strong> ${framework} <strong>|</strong> trainiert auf ${training.dataset_name} 
    <br><strong>| F1: </strong> ${training.metrics.f1.toFixed(2)} <strong>Recall:</strong> ${training.metrics.recall.toFixed(2)} <strong>Precision:</strong> ${training.metrics.precision.toFixed(2)}`

  } else {
    return `${name} <strong>|</strong> ${framework} <strong>|</strong> Basismodell`
  }
}

//global 
window.switchWindow = switchWindow

//test if server is available
let test = await getActiveModel();
if (test != undefined){
  //init Home, if no error in server connection
  initHome();
} else {
  await switchWindow("noserver")
}
  