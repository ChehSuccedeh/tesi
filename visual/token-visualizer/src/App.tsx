import React, { useState } from "react";
import TCAV from "./pages/TCAV";
import SAE from "./pages/SAE";

export default function App(){
  const [page, setPage] = useState<'TCAV Validation' | 'SAE Validation'>('TCAV Validation');

  return (
    <div>
      <nav style={{ marginBottom: 20 }}>
        <div style={{ marginTop: 24, marginLeft: 24 }}>
          <button 
            className={`px-2 py-1 border rounded disabled:opacity-50 mr-2 ${page === 'TCAV Validation' ? 'bg-black text-white' : 'bg-white text-black'}`}
            onClick={() => setPage('TCAV Validation')}>
          TCAV Validation
          </button>
          <button 
            className={`px-2 py-1 border rounded disabled:opacity-50 mr-2 ${page === 'SAE Validation' ? 'bg-black text-white' : 'bg-white text-black'}`}
            onClick={() => setPage('SAE Validation')}>
            SAE Validation
          </button>
        </div>
      </nav>
      {page === 'TCAV Validation' && <TCAV />}
      {page === 'SAE Validation' && <SAE />}
    </div>
  );
};

