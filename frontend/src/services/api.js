//default backend url
let API_BASE_URL = 'http://127.0.0.1:5000'

/**
 * loads the url of the backend, use default url if not avaiable
 */
async function loadConfig(){
    let res = await fetch("/config/config.json")
    if(res.ok){
        let conf = await res.json();
        API_BASE_URL = conf.backendURL
        console.log(conf.backendURL)
    } else{
        console.log("Couldn't load config, use defeault url: "+API_BASE_URL)
    }
}

await loadConfig();

/**
 * Get the models metadata
 * @returns {Object | undefined}
 */
export async function getModels(){
    const res = await fetch(`${API_BASE_URL}/models`)
    if(res.ok){
        return await res.json()
    } else {
        return undefined
    }
}

/**
 * Get the model metadata for the `id`
 * @param {number} id 
 * @returns {Object | undefined}
 */
export async function getModel(id){
    try{
        const res = await fetch(`${API_BASE_URL}/models/${id}`)
        if(res.ok){
            return await res.json()
        } else {
            return undefined
        }
    } catch (error){
        return undefined;
    }
}

/**
 * Returns the metadata of the active model
 * @returns {Object | undefined}
 */
export async function getActiveModel() {
    //needs try and catch because this function checks if the backend is available
    try{
        const res = await fetch(`${API_BASE_URL}/models/active`)
        console.log(res)
        if(res.ok){
            return await res.json()
        } else {
            return undefined
        }
    }
    catch(error){
        return undefined;
    }
}

/**
 * Sets the model with `id` as active
 * @param {number} id 
 * @returns {boolean}
 */
export async function setActiveModel(id){
    const res = await fetch(`${API_BASE_URL}/models/active/${id}`,{method:'PUT'})
    if (res.status != 204){
        return false;
    } else{
        return true;
    }
}

/**
 * Starts a NER-job with text
 * @param {string} text 
 * @returns {string | undefined} - the job-id or undefined
 */
export async function applyNERText(text){
    const res = await fetch(`${API_BASE_URL}/ner`,{method:'POST',headers: { "Content-Type": "application/json" },body: JSON.stringify({text})})
    if (res.status == 200){
        let responseData = await res.json()
        return responseData["job_id"]
    } else{
        return undefined;
    }
}

/**
 * Starts a NER-job with a file
 * @param {{file: file, split_sentences: boolean}} formData 
 * @returns {string | undefined} - the job-id or undefined
 */
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

/**
 * Gets the status or results of a NER-Job
 * @param {string} job_id 
 * @returns {{status: string, result:Object}}
 */
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

/**
 * Gets the trainingsdatasets metadata
 * @returns {Object | undefined}
 */
export async function getTrainingsData(){
    const res = await fetch(`${API_BASE_URL}/trainingdata`);
    if (res.ok){
        return await res.json();
    } else{
        return undefined;
    }
}

/**
 * Starts the model finetuning
 * @param {number} modelId - id of the model which should be finetuned
 * @param {number} datasetId -id of the dataset which should be used for finetuning
 * @param {string} name - name of the new model
 * @param {boolean} splitSentences - should the trainingdata be split into sentences 
 * @returns {number} - modified model id
 */
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
        let responseData = await res.json();
        return responseData["modified_model_id"]
    } else {
        return undefined;
    }
}

/**
 * Uploads a trainingsdataset
 * @param {file} file 
 * @param {string} datasetName 
 * @returns {boolean}
 */
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
