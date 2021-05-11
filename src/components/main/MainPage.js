import React, { useEffect, useState, useRef } from "react";
import * as yolo from "../model/Yolo";
import * as tf from "@tensorflow/tfjs";
import wasmSimdPath from "@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm";
import wasmSimdThreadedPath from "@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm";
import wasmPath from "@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import "./main-page.css";
import videoFile from "../../assets/video_perilla_2.mp4";

const MainPage = () => {
  const [model, setModel] = useState(null);
  const [buttonLegend, setButtonLegend] = useState("Play");
  const testVideo = useRef(null);
  const canvas = useRef(null);
  const playing = useRef(false);
  const ctx = useRef(null);

  useEffect(() => {
    if (model == null) {
      tf.ready().then(async () => {
        console.log(navigator);
        setWasmPaths({
          "tfjs-backend-wasm.wasm": wasmPath,
          "tfjs-backend-wasm-simd.wasm": wasmSimdPath,
          "tfjs-backend-wasm-threaded-simd.wasm": wasmSimdThreadedPath
        });
        await tf.setBackend("wasm");
        console.log("Using backend: " + tf.getBackend());
        await loadModel();
      });
    } else {
      const intervalId = predictVideo();
      return () => clearInterval(intervalId);
    }
  }, [model]);

  const predictVideo = () => {
    const interval = setInterval(async () => {
      if (playing.current) {
        const initTime = new Date();
        ctx.current.clearRect(
          0,
          0,
          canvas.current.width,
          canvas.current.height
        );
        ctx.current.drawImage(testVideo.current, 0, 0);
        const _boxes = tf.tidy(() => {
          return yolo.predict(canvas.current, model);
        });

        ctx.current.strokeStyle = "#0000FF";
        _boxes.forEach((box) => {
          ctx.current.strokeRect(box[0], box[1], box[2], box[3]);
        });
        // console.log(tf.memory().numTensors);
        const endTime = new Date();
        const fps = 1 / ((endTime - initTime) / 1000);
        console.log("fps: " + fps);
      }
    }, 500);
    return interval;
  };

  const loadModel = async () => {
    // const _model = await yolo.getTrainedModel();
    const _model = await tf.loadGraphModel(
      "../assets/models/mobilenet/model.json"
    );
    console.log("Model loaded");
    let init = new Date();
    // yolo.predict(testVideo.current, _model);
    await _model.executeAsync(
      tf.cast(tf.browser.fromPixels(testVideo.current), "float32").expandDims(0)
    );
    let end = new Date();
    let delta = end - init;
    console.log("First prediction issued in: " + delta + "ms");
    init = new Date();
    // yolo.predict(testVideo.current, _model);
    const prediction = await _model.executeAsync(
      tf.cast(tf.browser.fromPixels(testVideo.current), "float32").expandDims(0)
    );
    end = new Date();
    delta = end - init;
    console.log("Second prediction issued in: " + delta + "ms");
    console.log(prediction);
    setModel(_model);
  };

  const play = () => {
    playing.current = true;
    setButtonLegend("Pause");
  };

  const pause = () => {
    playing.current = false;
    setButtonLegend("Play");
  };

  const setCanvasParams = () => {
    canvas.current.width = testVideo.current.videoWidth;
    canvas.current.height = testVideo.current.videoHeight;
    ctx.current = canvas.current.getContext("2d");
  };

  return (
    <div>
      <video
        ref={testVideo}
        src={videoFile}
        // controls={true}
        playsInline
        // style={{ display: "none" }}
        onPlay={() => play()}
        onPause={() => pause()}
        onLoadedData={() => setCanvasParams()}
      ></video>
      <canvas ref={canvas}></canvas>
      {model != null && (
        <button
          className="boton"
          onClick={() => {
            if (playing.current) {
              testVideo.current.pause();
            } else {
              testVideo.current.play();
            }
          }}
        >
          {buttonLegend}
        </button>
      )}
    </div>
  );
};

export default MainPage;
