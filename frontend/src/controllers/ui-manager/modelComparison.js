import { getModels, setActiveModel } from "../../services/api.js";

let selectedId = undefined
let buttonSetActive = document.getElementById("buttonSetModelActive")
let labelFeedbackSetModelActive = document.getElementById("labelFeedbackSetModelActive")

function createTable(data) {
    const container = document.getElementById('table-container');
    
    const table = document.createElement('table');
    table.style.width = '100%';
    table.style.borderCollapse = 'collapse';
    
    // heading
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    ['ID', 'Name', 'Framework','Basis Modell', 'Trainingsdatum','Trainingsdaten', 'F1','Recall','Precision','Genauigkeit','Laufzeit [s]'].forEach(text => {
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
    models.forEach(model => {
      const tr = document.createElement('tr');
      tr.style.cursor = 'pointer';
  
      tr.addEventListener('click', () => {
        tbody.querySelectorAll('tr').forEach(row => row.style.backgroundColor = '');
        tr.style.backgroundColor = '#cce5ff';
        selectedId = model.id;
        console.log('Ausgewählte ID:', model.id);
        buttonSetActive.disabled = false;
      });
  
      let modelValues = [model.id, model.name, model.framework_name, model.base_model_name]
      let trainingsData = undefined
      if (model.trainings.length > 0){
          let lastTrainingsIndex = model.trainings.length -1;
          trainingsData = [model.trainings[lastTrainingsIndex].date, model.trainings[lastTrainingsIndex].dataset_name, model.trainings[lastTrainingsIndex].metrics.f1.toFixed(2),
          model.trainings[lastTrainingsIndex].metrics.recall.toFixed(2), model.trainings[lastTrainingsIndex].metrics.precision.toFixed(2), model.trainings[lastTrainingsIndex].metrics.accuracy.toFixed(2),
          model.trainings[lastTrainingsIndex].metrics.duration.toFixed(2)]
      } else {
          trainingsData = ["","","","","","",""]
      }
      modelValues =modelValues.concat(trainingsData)
      modelValues.forEach(value => {
        const td = document.createElement('td');
        td.innerText = value;
        td.style.border = '1px solid #ccc';
        td.style.padding = '8px';
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
  
    container.innerHTML = '';
    container.appendChild(table);
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
  
let models = await getModels();
if (models != undefined){
  createTable(models);
}

buttonSetActive.onclick = setModelActive