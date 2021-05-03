import axios from "axios";


const api = axios.create({
    baseURL:"http://localhost:8000"
});


export function classifyImage(image: File) {
    var formData = new FormData();
    formData.append("file", image);
    let promise = api.post('/classifyImage', formData, {headers: {
      'Content-Type': 'multipart/form-data'}
    });
    const dataPromise = promise.then((response) => response.data);
    console.log(dataPromise);
    return dataPromise;
}