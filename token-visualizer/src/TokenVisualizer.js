import React from 'react';

const getColor = (value) => {
    if (value > 0) {
        // Verde per valori positivi
        const greenValue = Math.min(255, Math.floor(255 * (value * 20)));
        return `rgb(${255 - greenValue}, 255, ${255 - greenValue})`;
    } else {
        // Rosso per valori negativi
        const redValue = Math.min(255, Math.floor(255 * (-value * 20)));
        return `rgb(255, ${255 - redValue}, ${255 - redValue})`;
    }
};

const TokenVisualizer = ({ tokenData }) => {
  return (
    <div style={{ padding: '20px', backgroundColor: '#f0f2f5', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)', overflowWrap: 'break-word' }}>
      {tokenData.map(([token, value], index) => {
        const color = getColor(value);
        const formattedValue = value.toFixed(3);
        
        return (
          <span
            key={index}
            title={`${formattedValue}`}
            style={{
              "background-color": color,
              padding: '4px 6px',
              borderRadius: '3px',
              margin: '2px',
              display: 'inline-block',
              cursor: 'pointer',
              whiteSpace: 'pre-wrap'
            }}
          >
            {token}
          </span>
        );
      })}
    </div>
  );
};

export default TokenVisualizer;