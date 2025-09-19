import React, { useState, useEffect } from 'react';
import TokenVisualizer from './TokenVisualizer';
import './App.css';
// Carica dinamicamente i file dalla cartella 'data' usando require.context (solo in ambiente webpack)
const dataContext = require.context('./data', false, /\.(json|jsonl)$/);
const dataFiles = dataContext.keys().map(file => file.replace('./', 'data/'));

const App = () => {
  const [data, setData] = useState([]);
  const [selectedFile, setSelectedFile] = useState(dataFiles[0]); // Imposta il primo file come default
  const [selectedConcept, setSelectedConcept] = useState('');
  const [selectedBottleneck, setSelectedBottleneck] = useState('');
  const [selectedSample, setSelectedSample] = useState(0);


  useEffect(() => {
    // Importa dinamicamente il file usando 'require'
    const rawData = require(`./${selectedFile}`);

    // console.log("Raw data loaded:", rawData);

    setData(rawData);
    if (rawData.length > 0) {
      setSelectedConcept(rawData[0].concept);
      setSelectedBottleneck(rawData[0].bottleneck);
      setSelectedSample(rawData[0].sample_index);
    }
}, [selectedFile]);

  // Usa uno stato per l'indice del sample corrente
  const [sampleIndex, setSampleIndex] = useState(0);
  
  // Trova tutti i sample unici per le selezioni attuali
  const allSamples = [...new Set(data
    .filter(item => item.concept === selectedConcept && item.bottleneck === selectedBottleneck)
    .map(item => item.sample_index)
  )].sort((a, b) => a - b);
  
  // Sincronizza l'indice dello stato con il sample index corretto
  useEffect(() => {
    const currentSamplePosition = allSamples.indexOf(selectedSample);
    setSampleIndex(currentSamplePosition);
  }, [selectedSample, allSamples]);

  // Gestori per i pulsanti Previous e Next
  const handlePrevious = () => {
    if (sampleIndex > 0) {
      const previousSample = allSamples[sampleIndex - 1];
      setSelectedSample(previousSample);
    }
  };

  const handleNext = () => {
    if (sampleIndex < allSamples.length - 1) {
      const nextSample = allSamples[sampleIndex + 1];
      setSelectedSample(nextSample);
    }
  };

  const concepts = [...new Set(data.map(item => item.concept))];
  const bottlenecks = [...new Set(data.map(item => item.bottleneck))].sort();

  const filteredData = data.find(item =>
    item.concept === selectedConcept &&
    item.bottleneck === selectedBottleneck &&
    item.sample_index === selectedSample
  );

  // Funzione di utilità per ottenere solo il nome del file senza cartella ed estensione
  const getFileName = (filePath) => {
    // filePath è tipo "data/qualcosa.json"
    return filePath.replace(/^.*[\\/]/, '').replace(/\.json$/, '');
  };

  return (
    <div className="app-container">
      <h1>Visualizzatore di Token</h1>
      <div className="controls-container">
        <label>
          Seleziona un file:
          <select value={selectedFile} onChange={e => setSelectedFile(e.target.value)}>
            {dataFiles.map(file => (
              <option key={file} value={file}>{getFileName(file)}</option>
            ))}
          </select>
        </label>
        <label>
          Seleziona un concetto:
          <select value={selectedConcept} onChange={e => setSelectedConcept(e.target.value)}>
            {concepts.map(concept => (
              <option key={concept} value={concept}>{concept}</option>
            ))}
          </select>
        </label>
        <label>
          Seleziona un bottleneck:
          <select value={selectedBottleneck} onChange={e => setSelectedBottleneck(e.target.value)}>
            {bottlenecks.map(bottleneck => (
              <option key={bottleneck} value={bottleneck}>{bottleneck}</option>
            ))}
          </select>
        </label>
        
        {/* Usiamo un div per raggruppare il select e i pulsanti */}
        <div style={{display: 'flex', alignItems: 'flex-end', gap: '10px'}}>
          <label>
            Seleziona un sample index:
            <select value={selectedSample} onChange={e => setSelectedSample(parseInt(e.target.value))}>
              {allSamples.map(sample => (
                <option key={sample} value={sample}>{sample}</option>
              ))}
            </select>
          </label>
          <button onClick={handlePrevious} disabled={sampleIndex <= 0}>Previous</button>
          <button onClick={handleNext} disabled={sampleIndex >= allSamples.length - 1}>Next</button>
        </div>
      </div>

      {filteredData ? (
        <TokenVisualizer tokenData={filteredData.token_sensitivities} />
      ) : (
        <p>Nessun dato trovato per la selezione.</p>
      )}
    </div>
  );
};

export default App;