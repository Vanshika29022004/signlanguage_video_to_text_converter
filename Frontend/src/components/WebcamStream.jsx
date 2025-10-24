import React, { useRef, useCallback } from "react";
import Webcam from "react-webcam";
import axios from "axios";

const videoConstraints = {
  width: 320,
  height: 240,
  facingMode: "user",
};

export default function WebcamStream({ onPrediction }) {
  const webcamRef = useRef(null);

  const captureFrame = useCallback(async () => {
    if (!webcamRef.current) return;
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    try {
      const res = await axios.post("http://127.0.0.1:5000/predict", { image: imageSrc });
      onPrediction(res.data.prediction);
    } catch (err) {
      console.error("Prediction error:", err);
    }
  }, [onPrediction]);

  React.useEffect(() => {
    const interval = setInterval(captureFrame, 1000); // every second
    return () => clearInterval(interval);
  }, [captureFrame]);

  return (
    <div className="border-4 border-indigo-500 rounded-lg p-2">
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        videoConstraints={videoConstraints}
        className="rounded-lg"
      />
    </div>
  );
}
