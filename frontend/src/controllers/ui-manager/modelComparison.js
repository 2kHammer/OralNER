import { getModels, setActiveModel } from "../../services/api.js";

export const modelColumns = ['ID', 'Name', 'Framework','Basis Modell', 'Trainingsdatum','Trainingsdaten', 'F1','Recall','Precision','Genauigkeit','Laufzeit [s]']

let selectedId = undefined
let buttonSetActive = document.getElementById("buttonSetModelActive")
let labelFeedbackSetModelActive = document.getElementById("labelFeedbackSetModelActive")

/*
 * general function
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

export function createModelTableVals(models){
  let modelVals = []
  models.forEach(model => {
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
  });
  return modelVals
}

function roundNumberReturnSpace(toCheck){
  if (Number.isFinite(toCheck)){
    return toCheck.toFixed(2)
  } else {
    ""
  }
}

/*
 * for Model Comparison Window
 */
const container = document.getElementById("modelComparisonContainer");
    

function handleClickModelComparison(modelId){
    selectedId = modelId;
    console.log('Ausgewählte ID:', selectedId);
    buttonSetActive.disabled = false;
}

async function setModelActive(){
  let now = new Date();
  let timeStr = now.toLocaleTimeString();
  if (await setActiveModel(selectedId)){
    labelFeedbackSetModelActive.innerHTML = `${timeStr}: Model ${selectedId} ist jetzt aktiv`
  } else {
    labelFeedbackSetModelActive.innerHTML = `${timeStr}: Fehler beim Setzen des aktiven Models`
  }
}

export async function initComparisonWindow(){
  let models = await getModels();
  if (models != undefined){
    let modelData = createModelTableVals(models);
    createTable(modelData,modelColumns, container, handleClickModelComparison);
  }
}

buttonSetActive.onclick = setModelActive
await initComparisonWindow();