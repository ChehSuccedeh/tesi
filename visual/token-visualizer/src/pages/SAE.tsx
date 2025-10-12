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
type SamplesString = string[][];
type TokenFeatureDict = { [feature_id: number]: number };
type SamplesTokenFeatureDict = TokenFeatureDict[][];


// const FEATURE_FILE = "sae/tokens_activations_12288.json"; // Cambia con il nome reale del file
const FEATURE_FILE = "sae/token_activations_24576.json"

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
  const [samplesString, setSamplesString] = useState<SamplesString>([]);
  const [samplesActivations, setSamplesActivations] = useState<SamplesTokenFeatureDict>([]);

  // Carica il file json
  useEffect(() => {
    async function loadData() {
      try {
        const mod = await import(/* @vite-ignore */ `../assets/${FEATURE_FILE}`);
        const data = mod.default as Sample[];
        const samplesStr = [] as SamplesString;
        const activations = [] as SamplesTokenFeatureDict;
        let maxFeature = 0;
        setSamples(data);

        // Parsing data
        for (const sample of data) {
          const tokenStrs = sample.tokens.map(t => t.token_str);
          const tokenActivations = sample.tokens.map(t => {
            const activations: TokenFeatureDict = {};
            for (const [fid, val] of t.activations) {
              activations[fid] = val;
              if (fid > maxFeature) maxFeature = fid;
            }
            return activations;
          });
          samplesStr.push(tokenStrs);
          activations.push(tokenActivations);
        }

        // Saving parsed data
        setSamplesString(samplesStr);
        setSamplesActivations(activations);

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
    let max = -Infinity, found = false;
    for (const s of samplesActivations) {
      for (const t of s) {
        const foundFeature = t[selectedFeature] || [];
        if (foundFeature) {
          const v = foundFeature;
          if (v > max) max = v;
          found = true;
        }
      }
    }
    setMinMax(found ? {min: 0, max} : {min: 0, max: -1});
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
              className="px-2 py-1 border rounded disabled:opacity-50 hover:bg-gray-200 transition-colors"
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
              className="px-2 py-1 border rounded disabled:opacity-50 hover:bg-gray-200 transition-colors"
              onClick={goNext}
              disabled={selectedFeature >= featureCount - 1}
            >
              Feature successiva
            </button>
          </div>
          <div className="flex flex-col gap-4">
            {
              (() => {
                const sampleElements = [];
                for (let sampleIdx = 0; sampleIdx < samplesString.length; sampleIdx++) {
                  const tokens = samplesString[sampleIdx];
                  let hasNonZeroActivation = false;
                  const tokenElements = [];
                  for (let tokenIdx = 0; tokenIdx < tokens.length; tokenIdx++) {
                    const token = tokens[tokenIdx];
                    const activation =
                      samplesActivations[sampleIdx]?.[tokenIdx]?.[selectedFeature] ?? 0;
                    if (activation !== 0) hasNonZeroActivation = true;
                    const bg = getColorForActivation(activation, minMax.min, minMax.max);
                    tokenElements.push(
                      <span
                        key={tokenIdx}
                        className="px-2 py-1 rounded text-sm font-mono"
                        style={{
                          background: bg,
                          border: "1px solid #e5e7eb",
                          transition: "background 0.2s",
                        }}
                        title={`Attivazione: ${activation}`}
                      >
                        {token}
                      </span>
                    );
                  }
                  if (!hasNonZeroActivation) continue;
                  sampleElements.push(
                    <div key={sampleIdx} className="flex flex-col gap-1 border rounded p-2">
                      <div className="font-bold mb-1">Sample {sampleIdx + 1}</div>
                      <div className="flex flex-wrap gap-1">{tokenElements}</div>
                    </div>
                  );
                }
                return sampleElements;
              })()
            }
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SAEPage;
