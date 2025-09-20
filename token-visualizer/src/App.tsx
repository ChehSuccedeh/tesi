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

function getColorForValue(value: number): string {
  // Usa un range morto tra -0.01 e 0.01: valori in questo range sono trasparenti
  if (value >= -0.01 && value <= 0.01) {
    return 'transparent';
  }

  // Clamp value tra -1 e 1
  let clamped = Math.max(-1, Math.min(1, value));
  let intensity = Math.abs(clamped);

  if (clamped > 0) {
    // Verde: più intenso più è grande il valore
    const green = 255;
    const red = Math.round(255 * (1 - intensity));
    const blue = Math.round(255 * (1 - intensity));
    return `rgb(${red},${green},${blue})`;
  } else {
    // Rosso: più intenso più è vicino a -1
    const red = 255;
    const green = Math.round(255 * (1 - intensity));
    const blue = Math.round(255 * (1 - intensity));
    return `rgb(${red},${green},${blue})`;
  }
}


export default function App() {
  // Lista dei file JSON disponibili
  const fileOptions = ASSET_JSON_FILES.map(f => ({ value: f, label: f }));
  const [selectedFile, setSelectedFile] = useState(ASSET_JSON_FILES[0]);
  const [dataset, setDataset] = useState<DataType[]>([]);

  // Caricamento dinamico del file selezionato
  useEffect(() => {
    async function loadData() {
      if (!selectedFile) return;
      try {
        const mod = await import(/* @vite-ignore */ `./assets/${selectedFile}`);
        console.log("Loaded data: ", selectedFile);
        console.log(`Length: ${mod.default.length}`);
        setDataset(mod.default as DataType[]);
      } catch (e) {
        setDataset([]);
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
                      backgroundColor: getColorForValue(sensitivity),
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
