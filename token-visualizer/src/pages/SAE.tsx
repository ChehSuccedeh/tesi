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
type SampleBase = {
  sample_index: number;
  tokens: TokenActivation[];
};

type SampleWithClass = SampleBase & {
  class: string | null;
};

type SampleWithTruePred = SampleBase & {
  true_class?: string | null;
  pred_class?: string | null;
};

type Sample = SampleWithClass | SampleWithTruePred;
type SamplesString = string[][];
type TokenFeatureDict = { [feature_id: number]: number };
type SamplesTokenFeatureDict = TokenFeatureDict[][];


// const FEATURE_FILE = "sae/tokens_activations_12288.json"; // Cambia con il nome reale del file
// const FEATURE_FILE = "sae/token_activations_24576.json"
// const FEATURE_FILE = "sae/token_activations_6144.json";
// const FEATURE_FILE = "sae/tokens_with_activations_sparse_24576_full.json";
// const FEATURE_FILE = "sae/feature_visualizations_hiddim_24576_standard.json";
const FEATURE_FILE = "sae/feature_visualizations_hiddim_24576_layer_12_new.json";

function getColorForActivation(value: number, min: number, max: number): string {
  // Più alto il valore, più verde. Più basso, più trasparente.
  if (value <= min) return 'transparent';
  const perc = (value - min) / (max - min + 1e-8); // normalizza
  // Da bianco a verde
  return `hsl(120, 80%, ${100 - perc * 50}%)`;
}

const SAEPage: React.FC = () => {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [featureList, setFeatureList] = useState<number[]>([]);
  const [selectedFeature, setSelectedFeature] = useState<number>(0);
  const [minMax, setMinMax] = useState<{min: number, max: number}>({min: 0, max: 1});
  const [samplesString, setSamplesString] = useState<SamplesString>([]);
  const [samplesActivations, setSamplesActivations] = useState<SamplesTokenFeatureDict>([]);
  const [classes, setClasses] = useState<string[]>([]);

  // Carica il file json
  useEffect(() => {
    async function loadData() {
      try {
        const mod = await import(/* @vite-ignore */ `../assets/${FEATURE_FILE}`);
        const data = mod.default as Sample[];
        const samplesStr = [] as SamplesString;
        const activations = [] as SamplesTokenFeatureDict;
        const featuresList: number[] = []; // elenco (con duplicati) delle feature incontrate
        const class_list: string[] = []

        setSamples(data);

        // Parsing data
        for (const sample of data) {

          const tokenStrs = sample.tokens.map(t => t.token_str);
          const tokenActivations = sample.tokens.map(t => {
            const activations: TokenFeatureDict = {};
            for (const [fid, val] of t.activations) {
              console.log(fid);
              activations[fid] = val;
              if (!featuresList.includes(fid)) {
                featuresList.push(fid); // aggiunge fid solo se non presente
              }
            }
            return activations;
          });
          samplesStr.push(tokenStrs);
          activations.push(tokenActivations);
          // Support either a single `class` field or `true_class` + `pred_class` pair.
          let displayClass: string;
          if ("class" in sample) {
            // SampleWithClass
            displayClass = sample.class ?? "-";
          } else {
            // SampleWithTruePred
            const s = sample as SampleWithTruePred;
            const tc = s.true_class ?? "-";
            const pc = s.pred_class ?? "-";
            displayClass = `True: ${tc} / Pred: ${pc}`;
          }
          class_list.push(displayClass);
        }

        // Saving parsed data
        setSamplesString(samplesStr);
        setSamplesActivations(activations);
        setClasses(class_list);

        const sortedFeaturesList = featuresList.sort((a, b) => a - b);
        setFeatureList(sortedFeaturesList);


      } catch (e) {
        setSamples([]);
        setFeatureList([]);
      }
    }
    loadData();
  }, []);

  // Calcola min/max per la feature selezionata
  useEffect(() => {
    if (samples.length === 0 || featureList.length === 0) {
      setMinMax({min: 0, max: 1});
      return;
    }
    let max = -Infinity, found = false;
    for (const s of samplesActivations) {
      for (const t of s) {
        const val = t[selectedFeature];
        if (typeof val === "number") {
          if (val > max) max = val;
          found = true;
        }
      }
    }
    setMinMax(found ? {min: 0, max} : {min: 0, max: -1});
  }, [samples, selectedFeature, featureList]);

  const featureCount = featureList.length;
  // Funzione per cambiare feature
  const goPrev = () => {
    if (featureList.length === 0) return;
    const idx = Math.max(0, featureList.indexOf(selectedFeature));
    const newIdx = Math.max(0, idx - 1);
    setSelectedFeature(featureList[newIdx]);
  };

  const goNext = () => {
    if (featureList.length === 0) return;
    const idx = Math.max(0, featureList.indexOf(selectedFeature));
    const newIdx = Math.min(featureList.length - 1, idx + 1);
    setSelectedFeature(featureList[newIdx]);
  };

  console.log(samplesActivations);
  console.log(samplesString);

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
              {" "}
              <select
                value={selectedFeature}
                onChange={(e) => {setSelectedFeature(Number(e.target.value)); console.log(e.target.value)}}
                className="w-48 px-1 border rounded text-center"
                disabled={featureList.length === 0}
              >
                {featureList.map((fid, idx) => (
                  <option key={idx} value={fid}>
                    {idx + 1} - Feature {fid}
                  </option>
                ))}
              </select>
              {" "}
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
                      <div className="text-sm text-gray-600 text-right">
                        Classe: <span className="font-mono font-semibold">{classes[sampleIdx] ?? "-"}</span>
                      </div>
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
