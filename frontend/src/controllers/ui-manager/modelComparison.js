import { getModels, setActiveModel } from "../../services/api.js";

export const modelColumns = ['ID', 'Name', 'Framework','Basis Modell', 'Trainingsdatum','Trainingsdaten', 'F1','Recall','Precision','Genauigkeit','Laufzeit [s]']

let selectedId = undefined
let buttonSetActive = document.getElementById("buttonSetModelActive")
let labelFeedbackSetModelActive = document.getElementById("labelFeedbackSetModelActive")

/**
 * Creates the table to compare and select data (models, datasets)
 * 
 * @param {string[][]} data - data in the order in which it is displayed in the table 
 * @param {string[]} columns - the column names for the data
 * @param {HTMLElement} container - the html element where the table is saved
 * @param {(id)=> void} handleClickModel - the function which defines how an element selection is handled
 }}
 */
export function createTable(data,columns, container, handleClickModel) {
    const table = document.createElement('table');
    table.style.width = '100%';
    table.style.borderCollapse = 'collapse';
    
    // heading
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    columns.forEach(text => {
      const th = document.createElement('th');
      th.innerText = text;
      th.style.border = '1px solid #ccc';
      th.style.padding = '8px';
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
  
    // Body
    const tbody = document.createElement('tbody');
    data.forEach(row => {
      const tr = document.createElement('tr');
      tr.style.cursor = 'pointer';
  
      tr.addEventListener('click', () => {
        tbody.querySelectorAll('tr').forEach(row => row.style.backgroundColor = '');
        tr.style.backgroundColor = '#cce5ff';
        //id has to be on position 0
        handleClickModel(row[0]);
      });
  
      for (let i =0; i < row.length; i++){
        const td = document.createElement('td');
        td.innerText = row[i];
        td.style.border = '1px solid #ccc';
        td.style.padding = '8px';
        tr.appendChild(td);
      };
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
  
    container.innerHTML = '';
    container.appendChild(table);
}



/*
 * for Model Comparison Window
 */


const container = document.getElementById("modelComparisonContainer");
    
/**
 * Creates the data for createTable from the model metadata `models`
 * @param {Object[]} models - model metadata from the backend 
 * @returns {string[][]} - data in order for the table
 */
export function createModelTableVals(models){
  let modelVals = []
  models.forEach(model => {
      if(model.state != "IN_TRAINING"){
      let rowData = [model.id, model.name, model.framework_name, model.base_model_name]
      let trainingsData = undefined
      if (model.trainings.length > 0){
          let lastTrainingsIndex = model.trainings.length -1;
          let metrics = model.trainings[lastTrainingsIndex].metrics;
          trainingsData = [model.trainings[lastTrainingsIndex].date, model.trainings[lastTrainingsIndex].dataset_name, 
          roundNumberReturnSpace(metrics.f1),
          roundNumberReturnSpace(metrics.recall), 
          roundNumberReturnSpace(metrics.precision), 
          roundNumberReturnSpace(metrics.accuracy),
          roundNumberReturnSpace(metrics.duration)]
      } else {
          trainingsData = ["","","","","","",""]
      }
      rowData = rowData.concat(trainingsData)
      modelVals.push(rowData)
    }
  });
  return modelVals
}

/**
 * Returns "" if `toCheck` is no number, else returns it rounded
 * @param {any} toCheck 
 * @returns {number | string} - fixed number or ""
 */
function roundNumberReturnSpace(toCheck){
  if (Number.isFinite(toCheck)){
    return toCheck.toFixed(2)
  } else {
    ""
  }
}

/**
 * Saves the id for the selected model in `selectedId`
 * @param {number} modelId 
 */
function handleClickModelComparison(modelId){
    selectedId = modelId;
    console.log('Ausgewählte ID:', selectedId);
    buttonSetActive.disabled = false;
}

/**
 * Tries to set the selected model as active and displays the result
 */
async function setModelActive(){
  let now = new Date();
  let timeStr = now.toLocaleTimeString();
  if (await setActiveModel(selectedId)){
    labelFeedbackSetModelActive.innerHTML = `${timeStr}: Model ${selectedId} ist jetzt aktiv`
  } else {
    labelFeedbackSetModelActive.innerHTML = `${timeStr}: Fehler beim Setzen des aktiven Models`
  }
}

/**
 * Inits the comparison window: retrieves models and creates the table
 */
export async function initComparisonWindow(){
  try{
    let models = await getModels();
    if (models != undefined){
      let modelData = createModelTableVals(models);
      createTable(modelData,modelColumns, container, handleClickModelComparison);
    } 
  }catch(e){
    console.error("No Server connection for model comparison")
  }
}


/*
 * Event Listeners and applied functions 
 */

buttonSetActive.onclick = setModelActive
await initComparisonWindow();