const API_BASE_URL = 'http://127.0.0.1:5000'

export async function getModels(){
    const res = await fetch(`${API_BASE_URL}/models`)
    if(res.ok){
        return res.json()
    } else {
        return undefined
    }
}

export async function getModel(id){
    const res = await fetch(`${API_BASE_URL}/models/${id}`)
    if(res.ok){
        return res.json()
    } else {
        return undefined
    }
}

export async function getActiveModel() {
    const res = await fetch(`${API_BASE_URL}/models/active`)
    if(res.ok){
        return res.json()
    } else {
        return undefined
    }
}

export async function setActiveModel(id){
    const res = await fetch(`${API_BASE_URL}/models/active/${id}`,{method:'PUT'})
    if (res.status != 204){
        return false;
    } else{
        return true;
    }
}

export async function applyNERText(text){
    const res = await fetch(`${API_BASE_URL}/ner`,{method:'POST',headers: { "Content-Type": "application/json" },body: JSON.stringify({text})})
    if (res.ok){
        return res.json()
    } else{
        return undefined;
    }
}

export async function applyNERFile(file){
    const res = await fetch(`${API_BASE_URL}/ner`,{method:'POST',body:file}) 
    if (res.ok){
        return res.json()
    } else {
        return undefined;
    }
}

export async function getTrainingsData(){
    const res = await fetch(`${API_BASE_URL}/trainingdata`);
    if (res.ok){
        return res.json();
    } else{
        return undefined;
    }
}

export async function startFinetune(modelId, datasetId, name){
    let body =  JSON.stringify({
        "model_id":modelId,
        "dataset_id":datasetId,
        "parameters":{
            "new_model_name":name
        }
    })
    const res = await fetch(`${API_BASE_URL}/ner/finetune`, {method:'POST',headers: { "Content-Type": "application/json" }, body:body})
    if (res.status == 202){
        return res.json();
    } else {
        return undefined;
    }
}
