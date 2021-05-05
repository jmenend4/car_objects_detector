const webpack = require("webpack");
const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const CopyPlugin = require("copy-webpack-plugin");

process.env.NODE_ENV = "development";

module.exports = {
  mode: "development",
  target: "web",
  devtool: "cheap-module-source-map",
  entry: "./src/index",
  output: {
    path: path.resolve(__dirname, "build"),
    publicPath: "/",
    filename: "bundle.js"
  },
  devServer: {
    stats: "minimal",
    overlay: true,
    historyApiFallback: true,
    disableHostCheck: true,
    headers: { "Access-Control-Allow-Origin": "*" },
    https: false
  },
  plugins: [
    // new webpack.DefinePlugin({
    //   "process.env.API_URL": JSON.stringify("http://localhost:3001")
    // }),
    new HtmlWebpackPlugin({
      template: "src/index.html",
      favicon: "src/favicon.ico"
    }),
    new CopyPlugin([{ from: "src/assets", to: "assets" }])
  ],
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: ["babel-loader", "eslint-loader"]
      },
      {
        test: /(\.css)$/,
        use: ["style-loader", "css-loader"]
      },
      {
        test: /\.(jpe?g|jfif|png|gif|woff|woff2|eot|ttf|svg|mp4)(\?[a-z0-9=.]+)?$/,
        use: "url-loader?limit=100000"
      }
      // ,
      // {
      //   test: /\.wasm$/i,
      //   type: "javascript/auto",
      //   use: [
      //     {
      //       loader: "file-loader"
      //     }
      //   ]
      // }
    ]
  }
};
