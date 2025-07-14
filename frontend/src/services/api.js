const API_BASE_URL = 'http://127.0.0.1:5000'

export async function getModels(){
    const res = await fetch(`${API_BASE_URL}/models`)
    if(res.ok){
        return await res.json()
    } else {
        return undefined
    }
}

export async function getModel(id){
    const res = await fetch(`${API_BASE_URL}/models/${id}`)
    if(res.ok){
        return await res.json()
    } else {
        return undefined
    }
}

export async function getActiveModel() {
    const res = await fetch(`${API_BASE_URL}/models/active`)
    if(res.ok){
        return await res.json()
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
    if (res.status == 200){
        let responseData = await res.json()
        return responseData["job_id"]
    } else{
        return undefined;
    }
}

export async function applyNERFile(formData){
    const res = await fetch(`${API_BASE_URL}/ner`,{method:'POST',body:formData}) 
    if (res.status == 200){
        let responseData = await res.json()
        return responseData.job_id
    } else {
        console.log(res.status)
        return undefined;
    }
}

export async function getNERResults(job_id){
    const res = await fetch(`${API_BASE_URL}/ner/${job_id}`,{method:'GET'})
    if (res.status == 200 || res.status  == 202){
        let responseData = await res.json();
        if (responseData["status"] == "done"){
            console.log(responseData)
            return {"state":true, "result":responseData["result"]}
        } else if (responseData["status"] == "processing"){
            return {"state":false}
        }
    } else if (res.status == 404){
        return {"state":undefined}
    }

}

export async function getTrainingsData(){
    const res = await fetch(`${API_BASE_URL}/trainingdata`);
    if (res.ok){
        return await res.json();
    } else{
        return undefined;
    }
}

export async function startFinetune(modelId, datasetId, name, splitSentences){
    let body =  JSON.stringify({
        "model_id":modelId,
        "dataset_id":datasetId,
        "parameters":{
            "new_model_name":name,
            "split_sentences":splitSentences
        }
    })
    const res = await fetch(`${API_BASE_URL}/ner/finetune`, {method:'POST',headers: { "Content-Type": "application/json" }, body:body})
    if (res.status == 202){
        return await res.json();
    } else {
        return undefined;
    }
}

export async function uploadTrainingData(file, datasetName){
    let formData = new FormData();
    formData.append("file", file)
    formData.append("dataset_name", datasetName)

    const res =await fetch(`${API_BASE_URL}/trainingdata`, {method: 'POST', body: formData});
    if (res.status != 201){
        return false;
    } else {
        return true;
    }
}
