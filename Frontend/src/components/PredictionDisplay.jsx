import React from "react";

export default function PredictionDisplay({ prediction }) {
  return (
    <div className="mt-6 text-center">
      <h2 className="text-lg text-gray-400">Predicted Text:</h2>
      <p className="text-2xl font-semibold text-green-400 mt-2">{prediction}</p>
    </div>
  );
}
