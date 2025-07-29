import { useState } from 'react';

function App() {
    const [F1, setF1] = useState(0.04);
    const [F2, setF2] = useState(0.08);
    const [K1, setK1] = useState(0.056);
    const [K2, setK2] = useState(0.074);
    const [Du1, setDu1] = useState(0.7);
    const [Du2, setDu2] = useState(0.7);
    const [Dv1, setDv1] = useState(0.25);
    const [Dv2, setDv2] = useState(0.25);
    const [imgSrc, setImgSrc] = useState<string>("/placeholder.png");
    const [loading, setLoading] = useState(false);

  const generatePattern = async () => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/generate?F1=${F1}&F2=${F2}&K1=${K1}&K2=${K2}&Du1=${Du1}&Du2=${Du2}&Dv1=${Dv1}&Dv2=${Dv2}`);
      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setImgSrc(imageUrl);
    } catch (err) {
      console.error('Failed to fetch pattern:', err);
    }
    setLoading(false);
  };


  return (
    
    <div className="max-w-4xl mx-auto p-5">
      <h1 className="text-3xl font-bold text-blue-500">Gray-Scott Turing Pattern Generator</h1>
      <div className="grid grid-cols-3 grid-rows-3 gap-4 place-items-center">
        
        {/* Top sliders (span image width) */}
        <div className="col-span-3 flex flex-col items-end space-y-2">
          <div>
            <label>Du (top): {Du1.toFixed(4)}</label>
            <input type="range" min="0.01" max="1" step="0.001" value={Du1} onChange={(e) => setDu1(parseFloat(e.target.value))} />
          </div>
          <div>
            <label>Dv (top): {Dv1.toFixed(4)}</label>
            <input type="range" min="0.01" max="1" step="0.001" value={Dv1} onChange={(e) => setDv1(parseFloat(e.target.value))} />
          </div>
        </div>


        {/* Left sliders */}
        <div className="flex flex-col items-end space-y-2">
          <div>
            <label>F (left): {F1.toFixed(4)}</label>
            <input type="range" min="0.01" max="0.1" step="0.001" value={F1} onChange={(e) => setF1(parseFloat(e.target.value))} />
          </div>
          <div>
            <label>K (left): {K1.toFixed(4)}</label>
            <input type="range" min="0.01" max="0.1" step="0.001" value={K1} onChange={(e) => setK1(parseFloat(e.target.value))} />
          </div>
        </div>

        {/* Center: Image */}
        <div>
          {imgSrc && <img src={imgSrc} alt="Turing pattern" className="w-[512px] h-auto border" width={512} height={512} />}
        </div>

        {/* Right sliders */}
        <div className="flex flex-col items-start space-y-2">
          <div>
            <label>F (right): {F2.toFixed(4)}</label>
            <input type="range" min="0.01" max="0.1" step="0.001" value={F2} onChange={(e) => setF2(parseFloat(e.target.value))} />
          </div>
          <div>
            <label>K (right): {K2.toFixed(4)}</label>
            <input type="range" min="0.01" max="0.1" step="0.001" value={K2} onChange={(e) => setK2(parseFloat(e.target.value))} />
          </div>
        </div>

        {/* Bottom sliders (span image width) */}
        <div className="col-span-3 flex flex-col items-end space-y-2">
          <div>
            <label>Du (bottom): {Du2.toFixed(4)}</label>
            <input type="range" min="0.1" max="1" step="0.001" value={Du2} onChange={(e) => setDu2(parseFloat(e.target.value))} />
          </div>
          <div>
            <label>Dv (bottom): {Dv2.toFixed(4)}</label>
            <input type="range" min="0.1" max="1" step="0.001" value={Dv2} onChange={(e) => setDv2(parseFloat(e.target.value))} />
          </div>
        </div>

        {/* Button */}
        <div className="col-span-3 flex flex-col place-items-center space-y-2">
          <div>
            <button className="mb-2 px-4 py-2 bg-blue-600 text-white rounded" onClick={generatePattern}>
              {loading ? "Generating..." : "Generate"}
            </button>
          </div>
          <div>
            <a href={imgSrc} download="pattern.png" className="mt-4 inline-block px-4 py-2 bg-blue-500 text-white rounded">
              Download Pattern
            </a>
          </div>
        </div>

        {/* Bottom Right: spacing */}
        <div></div>
      </div>
    </div>
  );
}

export default App;