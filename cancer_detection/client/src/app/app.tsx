// https://github.com/TypeStrong/ts-loader/issues/595
// document!.getElementById('root')!.textContent = 'hello World';
import React from "react";
import ReactDom from "react-dom";

ReactDom.render(<div>Hello</div>, document.getElementById('root'))