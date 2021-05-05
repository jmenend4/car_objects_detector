import React, { useEffect, useState, useRef } from "react";
import PropTypes from "prop-types";
import { connect } from "react-redux";
import * as modelActions from "../../redux/actions/modelActions";
import * as yolo from "../model/Yolo";
import * as tf from "@tensorflow/tfjs";
import "./main-page.css";

const MainPage = () => {
  const [model, setModel] = useState(null);
  // const [boxes, setBoxes] = useState([]);
  const play = useRef(false);
  // const toPredict = "image";
  const testVideo = useRef(null);
  const canvas = useRef(null);
  const [ctx, setCtx] = useState(null);

  useEffect(() => {
    if (model == null) {
      tf.ready().then(() => {
        loadModel();
      });
    } else {
      const intervalId = setInterval(predictVideo);
      return () => {
        clearInterval(intervalId);
      };
    }
  }, [model]);

  useEffect(() => {
    if (testVideo.current != null && canvas.current != null) {
      testVideo.current.controls = true;
      // testVideo.current.loop = true;
      testVideo.current.addEventListener(
        "loadeddata",
        () => {
          console.log("loaded event fired");
          canvas.current.width = testVideo.current.videoWidth;
          canvas.current.height = testVideo.current.videoHeight;
          setCtx(canvas.current.getContext("2d"));
        },
        false
      );
      testVideo.current.addEventListener(
        "ended",
        () => {
          console.log("ended event fired");
          play.current = false;
        },
        false
      );
      // testVideo.current.addEventListener(
      //   "play",
      //   () => {
      //     play.current = true;
      //   },
      //   false
      // );
    }
  }, [testVideo, canvas]);

  const predictVideo = () => {
    // const div = document.getElementById("div0");
    // const canvas = document.createElement("canvas");
    // const ctx = canvas.getContext("2d");

    const interval = setInterval(async () => {
      if (play.current) {
        const _boxes = tf.tidy(() => {
          return yolo.predict(testVideo.current, model);
        });
        ctx.clearRect(0, 0, canvas.current.width, canvas.current.height);
        ctx.drawImage(testVideo.current, 0, 0);
        ctx.strokeStyle = "#0000FF";
        _boxes.forEach((box) => {
          ctx.strokeRect(box[0], box[1], box[2], box[3]);
        });
        console.log(tf.memory().numTensors);
      }
    }, 200);
    return interval;
  };

  const loadModel = async () => {
    // const _model = yolo.getNewYolo(416); // just to keep reference but no longer used
    const _model = await yolo.getTrainedModel();
    setModel(_model);
  };

  return (
    <div id="div0">
      {/* <img id="image19" src="../../assets/Archivo_019.jpeg"></img> */}
      <video
        ref={testVideo}
        src="../../assets/video_perilla_2.mp4"
        // style={{ display: "none" }}
      ></video>
      <canvas ref={canvas}></canvas>
      {/* {boxes.map((box, idx) => (
        <div
          key={idx}
          className="box"
          style={{
            "--top": box[1] + "px",
            "--left": box[0] + "px",
            "--height": box[3] + "px",
            "--width": box[2] + "px"
          }}
        ></div>
      ))} */}
      {model != null && (
        <button
          className="boton"
          onClick={() => {
            if (play.current) {
              // testVideo.current.pause();
            } else {
              testVideo.current.play();
            }
            console.log(testVideo.current.readyState);
            play.current = !play.current;
          }}
        >
          Play
        </button>
      )}
    </div>
  );
};

// MainPage.propTypes = {
//   model: PropTypes.object,
//   setModel: PropTypes.func.isRequired
// };

// const mapStateToProps = (state) => {
//   return { model: state.model };
// };

// const mapDispatchToProps = {
//   setModel: modelActions.setModel
// };

// export default connect(mapStateToProps, mapDispatchToProps)(MainPage);

export default MainPage;
