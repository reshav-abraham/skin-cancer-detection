import React, { useState } from 'react';
import ImageIcon from '@material-ui/icons/Image';
import Button from '@material-ui/core/Button';
import { classifyImage } from '../api';

//https://www.youtube.com/watch?v=Z5iWr6Srsj8&ab_channel=BenAwad

interface Props {
    text: string
    ok?: boolean //optional prop
}

export const ImageUpload: React.FC<Props> = () => {
    const [imgSrc, setImgSrc] = useState<File | null>();
    const [prediction, setPrediction] = useState<{"encoded_img":""}>({encoded_img:""});


    // function selectFile(text:string): void {
    //     setImgSrc(text);
    //   }

      function selectFile(e: React.ChangeEvent<HTMLInputElement>): void { 
        console.log(e.target.innerHTML);
        if(e.target.files){
            setImgSrc(e.target.files[0]);
        }
    } 

    return(
        <div>
            <Button
                variant="contained"
                component="label"
                >
                <ImageIcon fontSize="large" />
                Upload File
                <input
                type="file"
                onChange={selectFile}
                id="lesion-image" name="lesion-image"
                accept="image/png, image/jpeg"
            />
            </Button>
            { imgSrc ? <img id="target" src={URL.createObjectURL(imgSrc)} /> : ""}
            { imgSrc ? <Button 
                            onClick={()=>{
                                            classifyImage(imgSrc).then(data => {setPrediction(data)}).catch(err => console.log(err))
                        }}> 
                            Classify image 
                        </Button> : ""}
            {console.log("prediction", prediction)}
            {prediction ? <h3>{prediction.encoded_img}</h3>:""}
        </div>
        );
}