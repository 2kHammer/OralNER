const API_BASE_URL = 'http://127.0.0.1:5000'

export async function getModels(){
    const res = await fetch(`${API_BASE_URL}/models`)
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
    return res.json()
}

export async function applyNERFile(file){
    const res = await fetch(`${API_BASE_URL}/ner`,{method:'POST',body:file}) 
    return res.json()
}
