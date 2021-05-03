// https://github.com/TypeStrong/ts-loader/issues/595
// document!.getElementById('root')!.textContent = 'hello World';
import React from "react";
import ReactDom from "react-dom";
import { ImageUpload } from "./components/ImageUpload";

const App: React.FC = () => {
    return (
        <div>
            <ImageUpload text=''/>
        </div>
    );
};


ReactDom.render(<App />, document.getElementById('root'));