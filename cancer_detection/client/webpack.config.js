module.exports = {
    entry: './src/app/App.tsx',
    output: {
        path: __dirname + '/public',
        filename: 'build/bundle.js'
    },
    resolve: {
        extensions: ['.ts', '.tsx', '.js']
    },
    module: {
        rules: [
            { test: /\.tsx?$/, loader: 'ts-loader' }
        ]
    }
}