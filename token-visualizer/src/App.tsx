import { useState, useMemo, useEffect } from 'react';
import { Combobox } from './components/ui/combobox';
// ...existing code...
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"


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


export default function App() {
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
        const mod = await import(/* @vite-ignore */ `./assets/${selectedFile}`);
        console.log("Loaded data: ", selectedFile);
        console.log(`Length: ${mod.default.length}`);
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
        console.log("Min-Max sensitivities: ", min, max);
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
    console.log("Setting bottleneck to: ", value);
    _setSelectedBottleneck(value);
  }

  useEffect(() => {
    setSelectedConcept("");
    setSelectedBottleneck("");
    setSelectedSampleIndex("");
  }, [selectedFile]);

  // Trova l'elemento selezionato

  console.log("selectedConcept: ", selectedConcept);
  console.log("selectedBottleneck: ", selectedBottleneck);
  console.log("selectedSampleIndex: ", selectedSampleIndex);
  console.log("dataset length: ", dataset.length);
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
      <Card className="w-full max-w-xl">
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
              <Combobox
                options={bottlenecks.map(b => ({ value: String(b), label: String(b) }))}
                value={selectedBottleneck}
                onChange={setSelectedBottleneck}
                placeholder="Seleziona bottleneck..."
                className="w-full"
              />
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
      </Card>

      <Card className="w-full max-w-xl">
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
}
