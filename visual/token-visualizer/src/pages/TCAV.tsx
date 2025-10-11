import React, { useState, useMemo, useEffect } from 'react';
import { Combobox } from '../components/ui/combobox';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const ASSET_JSON_FILES = [
  "code_classification.json",
  "packet_inspection.json",
];

type DataType = {
  concept: string;
  bottleneck: number;
  sample_index: number;
  token_sensitivities: [string, number][];
};

function getColorForValue(value: number, min_max: [number, number]): string {
  // Usa un range morto tra -0.0000001 e 0.0000001: valori in questo range sono trasparenti
  if (value >= -0.0000001 && value <= 0.0000001) {
    return 'transparent';
  }

  const [min, max] = min_max;

  // Colori HSV: valori positivi -> verde, negativi -> rosso, 0 -> trasparente
  // Hue: 120 (verde) per valori positivi, 0 (rosso) per negativi
  if (value > 0) {
    const perc = (value / max) * 100;
    // Saturazione e valore pieni, hue 120 (verde)
    return `hsl(120, ${100}%, ${100-perc}%)`;
  } else {
    const perc = (value / min) * 100;
    // Saturazione e valore pieni, hue 0 (rosso)
    return `hsl(0, ${100}%, ${100-perc}%)`;
  }
}

const HomePage: React.FC = () => {
  // Lista dei file JSON disponibili
  const fileOptions = ASSET_JSON_FILES.map(f => ({ value: f, label: f }));
  const [selectedFile, setSelectedFile] = useState(ASSET_JSON_FILES[0]);
  const [dataset, setDataset] = useState<DataType[]>([]);
  const [minMax, setMinMax] = useState<[number, number]>([-1, 1]);

  // Caricamento dinamico del file selezionato
  useEffect(() => {
    async function loadData() {
      if (!selectedFile) return;
      try {
        const mod = await import(/* @vite-ignore */ `../assets/${selectedFile}`);
        const data = mod.default as DataType[];
        setDataset(data);

        // Trova min e max tra tutte le sensitivities dei token
        let min = Infinity;
        let max = -Infinity;
        data.forEach(item => {
          item.token_sensitivities.forEach(([_, value]) => {
            if (value < min) min = value;
            if (value > max) max = value;
          });
        });
        setMinMax([min, max]);
      } catch (e) {
        console.error("Error loading data: ", e);
        setDataset([]);
        setMinMax([-1, 1]);
      }
    }
    loadData();
  }, [selectedFile]);

  // Estrai valori unici per concept, bottleneck e sample_index
  const concepts = useMemo(() => Array.from(new Set(dataset.map(d => d.concept))), [dataset]);
  const bottlenecks = useMemo(() => Array.from(new Set(dataset.map(d => d.bottleneck))), [dataset]);
  const sampleIndices = useMemo(() => Array.from(new Set(dataset.map(d => d.sample_index))), [dataset]);

  const [selectedConcept, setSelectedConcept] = useState(() => concepts[0] ?? "");
  const [selectedBottleneck, _setSelectedBottleneck] = useState(() => bottlenecks.length > 0 ? String(bottlenecks[0]) : "");
  const [selectedSampleIndex, setSelectedSampleIndex] = useState(() => sampleIndices.length > 0 ? String(sampleIndices[0]) : "");

  // Aggiorna i valori selezionati quando cambiano le opzioni
  useEffect(() => {
    setSelectedConcept(concepts[0] ?? "");
  }, [concepts]);

  useEffect(() => {
    _setSelectedBottleneck(bottlenecks.length > 0 ? String(bottlenecks[0]) : "");
  }, [bottlenecks]);

  useEffect(() => {
    setSelectedSampleIndex(sampleIndices.length > 0 ? String(sampleIndices[0]) : "");
  }, [sampleIndices]);

  function setSelectedBottleneck(value: string) {
    _setSelectedBottleneck(value);
  }

  useEffect(() => {
    setSelectedConcept("");
    setSelectedBottleneck("");
    setSelectedSampleIndex("");
  }, [selectedFile]);

  // Trova l'elemento selezionato
  const selectedItem = useMemo(() => {
    if (!selectedConcept || !selectedBottleneck || !selectedSampleIndex) return undefined;
    return dataset.find(
      (item: DataType) =>
        item.concept == selectedConcept &&
        item.bottleneck == Number(selectedBottleneck) &&
        item.sample_index == Number(selectedSampleIndex)
    );
  }, [dataset, selectedConcept, selectedBottleneck, selectedSampleIndex]);

  return (
    <div className="flex flex-col items-center p-4 gap-4">
      <Card className="w-full max-w-5xl">
        <CardHeader>
          <CardTitle>Seleziona Filtri</CardTitle>
          <CardDescription>
            Scegli il file, concept, bottleneck e sample index per filtrare i dati.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4">
            <div>
              <h2 className="text-lg font-semibold mb-1">File</h2>
              <Combobox
                options={fileOptions}
                value={selectedFile}
                onChange={setSelectedFile}
                placeholder="Seleziona file..."
                className="w-full"
              />
            </div>
            <div>
              <h2 className="text-lg font-semibold mb-1">Concept</h2>
              <Combobox
                options={concepts.map(c => ({ value: c, label: c }))}
                value={selectedConcept}
                onChange={setSelectedConcept}
                placeholder="Seleziona concept..."
                className="w-full"
              />
            </div>
            <div>
              <h2 className="text-lg font-semibold mb-1">Bottleneck</h2>
              <div className="flex items-center gap-2">
              <button
                className="px-2 py-1 border rounded disabled:opacity-50"
                onClick={() => {
                const idx = bottlenecks.findIndex(b => String(b) === selectedBottleneck);
                if (idx > 0) setSelectedBottleneck(String(bottlenecks[idx - 1]));
                }}
                disabled={
                bottlenecks.length === 0 ||
                bottlenecks.findIndex(b => String(b) === selectedBottleneck) <= 0
                }
                aria-label="Previous bottleneck"
              >
                Previous
              </button>
              <Combobox
                options={bottlenecks.map(b => ({ value: String(b), label: String(b) }))}
                value={selectedBottleneck}
                onChange={setSelectedBottleneck}
                placeholder="Seleziona bottleneck..."
                className="flex-grow"
              />
              <button
                className="px-2 py-1 border rounded disabled:opacity-50"
                onClick={() => {
                const idx = bottlenecks.findIndex(b => String(b) === selectedBottleneck);
                if (idx < bottlenecks.length - 1 && idx !== -1) setSelectedBottleneck(String(bottlenecks[idx + 1]));
                }}
                disabled={
                bottlenecks.length === 0 ||
                bottlenecks.findIndex(b => String(b) === selectedBottleneck) === -1 ||
                bottlenecks.findIndex(b => String(b) === selectedBottleneck) >= bottlenecks.length - 1
                }
                aria-label="Next bottleneck"
              >
                Next
              </button>
              </div>
            </div>
            <div>
              <h2 className="text-lg font-semibold mb-1">Sample Index</h2>
              <div className="flex items-center gap-2">
                <button
                  className="px-2 py-1 border rounded disabled:opacity-50"
                  onClick={() => {
                    const idx = sampleIndices.findIndex(s => String(s) === selectedSampleIndex);
                    if (idx > 0) setSelectedSampleIndex(String(sampleIndices[idx - 1]));
                  }}
                  disabled={
                    sampleIndices.length === 0 ||
                    sampleIndices.findIndex(s => String(s) === selectedSampleIndex) <= 0
                  }
                  aria-label="Previous sample"
                >
                  Previous
                </button>
                <Combobox
                  options={sampleIndices.map(s => ({ value: String(s), label: String(s) }))}
                  value={selectedSampleIndex}
                  onChange={setSelectedSampleIndex}
                  placeholder="Seleziona sample index..."
                  className="flex-grow"
                />
                <button
                  className="px-2 py-1 border rounded disabled:opacity-50"
                  onClick={() => {
                    const idx = sampleIndices.findIndex(s => String(s) === selectedSampleIndex);
                    if (idx < sampleIndices.length - 1 && idx !== -1) setSelectedSampleIndex(String(sampleIndices[idx + 1]));
                  }}
                  disabled={
                    sampleIndices.length === 0 ||
                    sampleIndices.findIndex(s => String(s) === selectedSampleIndex) === -1 ||
                    sampleIndices.findIndex(s => String(s) === selectedSampleIndex) >= sampleIndices.length - 1
                  }
                  aria-label="Next sample"
                >
                  Next
                </button>
              </div>
            </div>
          </div>
        </CardContent>
        <CardContent>
          <div className="mt-4">
            <h3 className="font-semibold mb-2">Top 5 Token Sensitivities (positivi & negativi)</h3>
            {selectedConcept && selectedBottleneck ? (
              (() => {
          // Filtra tutti i sample per concept e bottleneck selezionati
          const filtered = dataset.filter(
            d =>
              d.concept === selectedConcept &&
              Number(d.bottleneck) === Number(selectedBottleneck)
          );
          if (filtered.length === 0) {
            return <p className="text-sm text-gray-500">Nessun dato per questa combinazione.</p>;
          }
          // Raccogli tutte le tuple (token, sensitivity, sample_index) dei sample filtrati
          const tokenTuples: [string, number, number][] = [];
          filtered.forEach(item => {
            item.token_sensitivities.forEach(([token, sensitivity]) => {
              tokenTuples.push([token, sensitivity, item.sample_index]);
            });
          });

          // Top 5 positivi
          const topPositives = tokenTuples
            .filter(([_, val]) => val > 0)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5);

          // Top 5 negativi
          const topNegatives = tokenTuples
            .filter(([_, val]) => val < 0)
            .sort((a, b) => a[1] - b[1])
            .slice(0, 5);

          return (
            <div className="flex flex-col gap-2">
              <div>
                <span className="font-semibold text-green-700 mr-2">Top 5 positivi:</span>
                {topPositives.length > 0 ? (
            topPositives.map(([token, val, sampleIdx], idx) => (
              <span
                key={idx}
                className="font-mono px-2 py-1 rounded"
                style={{
                  backgroundColor: getColorForValue(val, minMax),
                }}
                title={`Sensitivity: ${val.toFixed(4)} | Sample index: ${sampleIdx}`}
              >
                {token} <span className="text-xs text-gray-500">({val.toFixed(4)})</span>
              </span>
            ))
                ) : (
            <span className="text-sm text-gray-500">Nessun token positivo.</span>
                )}
              </div>
              <div>
                <span className="font-semibold text-red-700 mr-2">Top 5 negativi:</span>
                {topNegatives.length > 0 ? (
            topNegatives.map(([token, val, sampleIdx], idx) => (
              <span
                key={idx}
                className="font-mono px-2 py-1 rounded"
                style={{
                  backgroundColor: getColorForValue(val, minMax),
                }}
                title={`Sensitivity: ${val.toFixed(4)} | Sample index: ${sampleIdx}`}
              >
                {token} <span className="text-xs text-gray-500">({val.toFixed(4)})</span>
              </span>
            ))
                ) : (
            <span className="text-sm text-gray-500">Nessun token negativo.</span>
                )}
              </div>
            </div>
          );
              })()
            ) : (
              <p className="text-sm text-gray-500">Seleziona concept e bottleneck per vedere i top token.</p>
            )}
          </div>
        </CardContent>
      </Card>

      <Card className="w-full max-w-5xl">
        <CardHeader>
          <CardTitle>Selezione Corrente</CardTitle>
        </CardHeader>
        <CardContent>
          {selectedItem && (
            <>
              <h3 className="font-semibold mb-2">Token Sensitivities</h3>
              <p className="mb-2">
                {selectedItem.token_sensitivities.map(([token, sensitivity], idx) => (
                    <span
                    key={idx}
                    className="font-mono"
                    style={{
                      backgroundColor: getColorForValue(sensitivity, minMax),
                      padding: '2px',
                      margin: '2px',
                      display: 'inline-block'
                    }}
                    title={sensitivity.toFixed(4)}
                    >
                    {token}{" "}
                    </span>
                ))}
              </p>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default HomePage;
