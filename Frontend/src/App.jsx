import React, { useState } from "react";
import WebcamStream from "./components/WebcamStream";
import PredictionDisplay from "./components/PredictionDisplay";

export default function App() {
  const [prediction, setPrediction] = useState("Waiting for sign...");

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-900 text-white">
      <h1 className="text-3xl font-bold mb-4">🖐️ ISL Video to Text Converter</h1>
      <WebcamStream onPrediction={setPrediction} />
      <PredictionDisplay prediction={prediction} />
    </div>
  );
}
