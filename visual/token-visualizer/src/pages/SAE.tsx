import React, { useState, useEffect, useMemo } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

// Tipo dati per il file delle feature SAE
type SingleActivation = [number, number]; // [feature_id, value]
type TokenActivation = {
  token_idx: number;
  token_str: string;
  activations: SingleActivation[];
};
type Sample = {
  sample_index: number;
  tokens: TokenActivation[];
};


const FEATURE_FILE = "sae/token_activations.json"; // Cambia con il nome reale del file

function getColorForActivation(value: number, min: number, max: number): string {
  // Più alto il valore, più verde. Più basso, più trasparente.
  if (value <= min) return 'transparent';
  const perc = (value - min) / (max - min + 1e-8); // normalizza
  // Da bianco a verde
  return `hsl(120, 80%, ${100 - perc * 50}%)`;
}

const SAEPage: React.FC = () => {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [featureCount, setFeatureCount] = useState<number>(0);
  const [selectedFeature, setSelectedFeature] = useState<number>(0);
  const [minMax, setMinMax] = useState<{min: number, max: number}>({min: 0, max: 1});

  // Carica il file json
  useEffect(() => {
    async function loadData() {
      try {
        const mod = await import(/* @vite-ignore */ `../assets/${FEATURE_FILE}`);
        const data = mod.default as Sample[];
        setSamples(data);
        // Trova il massimo feature_id
        let maxFeature = 0;
        for (const sample of data) {
          for (const t of sample.tokens) {
            for (const fid of t.activations) {
              if (fid[0] > maxFeature) maxFeature = fid[0];
            }
          }
        }
        setFeatureCount(maxFeature + 1);
      } catch (e) {
        setSamples([]);
        setFeatureCount(0);
      }
    }
    loadData();
  }, []);

  // Calcola min/max per la feature selezionata
  useEffect(() => {
    if (samples.length === 0 || featureCount === 0) {
      setMinMax({min: 0, max: 1});
      return;
    }
    let min = Infinity, max = -Infinity, found = false;
    for (const sample of samples) {
      for (const t of sample.tokens) {
        const foundFeature = t.activations.find((fid) => fid[0] === selectedFeature) || [];
        if (foundFeature) {
          const v = foundFeature[1];
          if (v < min) min = v;
          if (v > max) max = v;
          found = true;
        }
      }
    }
    setMinMax(found ? {min, max} : {min: 0, max: 1});
  }, [samples, selectedFeature, featureCount]);

  // Funzione per cambiare feature
  const goPrev = () => setSelectedFeature(f => Math.max(0, f - 1));
  const goNext = () => setSelectedFeature(f => Math.min(featureCount - 1, f + 1));

  return (
    <div className="flex flex-col items-center p-4 gap-4">
      <Card className="w-full max-w-5xl">
        <CardHeader>
          <CardTitle>Visualizzazione attivazioni SAE</CardTitle>
          <CardDescription>
            Seleziona la feature e osserva i token più attivi nei vari samples.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4 mb-4">
            <button
              className="px-2 py-1 border rounded disabled:opacity-50"
              onClick={goPrev}
              disabled={selectedFeature <= 0}
            >
              Feature precedente
            </button>
            <span className="font-mono text-lg">
              Feature{" "}
              <input
              type="number"
              min={1}
              max={featureCount}
              value={selectedFeature + 1}
              onChange={e => {
                let val = Number(e.target.value);
                if (isNaN(val)) return;
                val = Math.max(1, Math.min(featureCount, val));
                setSelectedFeature(val - 1);
              }}
              className="w-16 px-1 border rounded text-center"
              style={{ width: "3.5rem" }}
              />{" "}
              / {featureCount}
            </span>
            <button
              className="px-2 py-1 border rounded disabled:opacity-50"
              onClick={goNext}
              disabled={selectedFeature >= featureCount - 1}
            >
              Feature successiva
            </button>
          </div>
          <div className="flex flex-col gap-4">
            {samples.map(sample => (
              <div key={sample.sample_index} className="border rounded p-2 bg-gray-50">
                <div className="mb-1 text-xs text-gray-500">Sample {sample.sample_index}</div>
                <div className="flex flex-wrap gap-1">
                  {sample.tokens.map((tok, idx) => {
                    // console.log(tok, idx);
                    const foundTuple = tok.activations.find((fid) => {console.log(fid[0], selectedFeature); return fid[0] === selectedFeature;});
                    const value = foundTuple ? foundTuple[1] : 0;
                    return (
                      <span
                        key={tok.token_idx}
                        className="font-mono px-2 py-1 rounded"
                        style={{
                          backgroundColor: getColorForActivation(value, minMax.min, minMax.max),
                          border: value === minMax.max && value !== 0 ? '2px solid #16a34a' : undefined,
                        }}
                        title={`Attivazione: ${value.toFixed(4)}`}
                      >
                        {tok.token_str}
                        <span className="text-xs text-gray-500"> ({value.toFixed(2)})</span>
                      </span>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SAEPage;
