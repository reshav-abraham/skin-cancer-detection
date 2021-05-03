// https://github.com/TypeStrong/ts-loader/issues/595
// document!.getElementById('root')!.textContent = 'hello World';
import React from "react";
import ReactDom from "react-dom";
import { ImageUpload } from "./components/ImageUpload";

const App: React.FC = () => {
    return (
        <div>
            <h3>Hello</h3>
            <ImageUpload text='hello'/>
        </div>
    );
};


ReactDom.render(<App />, document.getElementById('root'));