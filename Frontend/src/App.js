import React, { useRef, useState, useEffect, useCallback } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import { Holistic } from "@mediapipe/holistic";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { Camera } from "@mediapipe/camera_utils";

const CAPTURE_FRAMES = 30;

const holistic = new Holistic({
  locateFile: (file) =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
});

holistic.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  refineFaceLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [sequence, setSequence] = useState([]);
  const [sentence, setSentence] = useState([]);
  const [prediction, setPrediction] = useState("");
  const [speaking, setSpeaking] = useState(false);

  const extractKeypoints = (results) => {
    if (!results) return null;
    const pose = results.poseLandmarks
      ? results.poseLandmarks.flatMap((lm) => [
          lm.x,
          lm.y,
          lm.z,
          lm.visibility || 0,
        ])
      : Array(33 * 4).fill(0);
    const face = results.faceLandmarks
      ? results.faceLandmarks.flatMap((lm) => [lm.x, lm.y, lm.z])
      : Array(468 * 3).fill(0);
    const leftHand = results.leftHandLandmarks
      ? results.leftHandLandmarks.flatMap((lm) => [lm.x, lm.y, lm.z])
      : Array(21 * 3).fill(0);
    const rightHand = results.rightHandLandmarks
      ? results.rightHandLandmarks.flatMap((lm) => [lm.x, lm.y, lm.z])
      : Array(21 * 3).fill(0);
    return [...pose, ...face, ...leftHand, ...rightHand];
  };

  const onResults = useCallback(
    (results) => {
      const ctx = canvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      ctx.drawImage(
        results.image,
        0,
        0,
        canvasRef.current.width,
        canvasRef.current.height
      );

      if (results.poseLandmarks) {
        drawConnectors(ctx, results.poseLandmarks, Holistic.POSE_CONNECTIONS, {
          color: "#00FF00",
          lineWidth: 4,
        });
        drawLandmarks(ctx, results.poseLandmarks, {
          color: "#FF0000",
          lineWidth: 2,
        });
      }
      if (results.faceLandmarks) {
        drawConnectors(
          ctx,
          results.faceLandmarks,
          Holistic.FACE_MESH_TESSELATION,
          {
            color: "#C0C0C070",
            lineWidth: 1,
          }
        );
      }
      if (results.leftHandLandmarks) {
        drawConnectors(
          ctx,
          results.leftHandLandmarks,
          Holistic.HAND_CONNECTIONS,
          {
            color: "#CC0000",
            lineWidth: 5,
          }
        );
        drawLandmarks(ctx, results.leftHandLandmarks, {
          color: "#00FF00",
          lineWidth: 2,
        });
      }
      if (results.rightHandLandmarks) {
        drawConnectors(
          ctx,
          results.rightHandLandmarks,
          Holistic.HAND_CONNECTIONS,
          {
            color: "#00CC00",
            lineWidth: 5,
          }
        );
        drawLandmarks(ctx, results.rightHandLandmarks, {
          color: "#FF0000",
          lineWidth: 2,
        });
      }

      const keypoints = extractKeypoints(results);
      if (keypoints) {
        setSequence((seq) => {
          const newSeq = [...seq, keypoints];
          if (newSeq.length > CAPTURE_FRAMES) newSeq.shift();
          return newSeq;
        });
      }
    },
    [setSequence, setSentence]
  );

  useEffect(() => {
    holistic.onResults(onResults);

    if (!webcamRef.current || !webcamRef.current.video) return;

    const camera = new Camera(webcamRef.current.video, {
      onFrame: async () =>
        await holistic.send({ image: webcamRef.current.video }),
      width: 640,
      height: 480,
    });
    camera.start();

    return () => {
      camera.stop();
      holistic.close();
    };
  }, [onResults]);

  useEffect(() => {
    if (sequence.length === CAPTURE_FRAMES) {
      axios
        .post("http://127.0.0.1:5000/predict", { sequence })
        .then((res) => {
          const pred = res.data.prediction;
          setPrediction(pred);
          setSentence((s) =>
            s.length === 0 || s[s.length - 1] !== pred ? [...s, pred] : s
          );
        })
        .catch(console.error);
    }
  }, [sequence]);

  const clearText = () => {
    setSentence([]);
    setPrediction("");
  };

  const speakText = () => {
    if (!("speechSynthesis" in window)) {
      alert("Speech synthesis not supported");
      return;
    }
    const utterance = new SpeechSynthesisUtterance(sentence.join(" "));
    setSpeaking(true);
    utterance.onend = () => setSpeaking(false);
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div style={{ textAlign: "center" }}>
      <h1>Sign Language to Text</h1>
      <Webcam ref={webcamRef} mirrored width={640} height={480} />
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        style={{ position: "absolute", left: 0, top: 0, zIndex: 1 }}
      />
      <div style={{ marginTop: 20 }}>
        <h2>Prediction: {prediction}</h2>
        <p>Sentence: {sentence.join(" ")}</p>
        <button onClick={clearText}>Clear</button>
        <button
          onClick={speakText}
          disabled={speaking || sentence.length === 0}
        >
          {speaking ? "Speaking..." : "Speak"}
        </button>
      </div>
    </div>
  );
}
 
export default App;
