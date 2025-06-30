 import { getActiveModel, getModels } from "../../services/api.js";

 export async function getActiveModelname(){
    let activeModel = await getActiveModel();
    if(activeModel){
        return activeModel.name;
    } else{
        return undefined;
    }
 }
