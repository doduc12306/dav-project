"use client";

import { useEffect, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import { Database, FileText, BarChart2, CheckCircle2, AlertCircle, Play, Pause } from "lucide-react";
import dynamic from "next/dynamic";

// Dynamically import Plotly
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export default function DemoPage() {
  const reduce = useReducedMotion();
  const [skeletons, setSkeletons] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>("");
  const [prediction, setPrediction] = useState<any>(null);
  const [jointsData, setJointsData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  
  // Animation state
  const [skeletonFrames, setSkeletonFrames] = useState<any[]>([]);
  const [connections, setConnections] = useState<any[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    fetch("/api/skeletons")
      .then((res) => res.json())
      .then((data) => {
        if (data.files && data.files.length > 0) {
          setSkeletons(data.files);
          setSelectedFile(data.files[0]);
        }
      })
      .catch((err) => setError("Failed to fetch skeletons from server."));
  }, []);

  useEffect(() => {
    if (!selectedFile) return;
    
    setLoading(true);
    setPrediction(null);
    setError("");

    fetch(`/api/predict/${selectedFile}`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch prediction");
        return res.json();
      })
      .then((data) => {
        setPrediction(data.prediction);
        setSkeletonFrames(data.skeleton);
        setConnections(data.connections);
        setCurrentFrame(0);
        setIsPlaying(true);
        formatPlotData(data.skeleton[0], data.connections);
        setLoading(false);
      })
      .catch((err) => {
        setError("Error processing skeleton file.");
        setLoading(false);
      });
  }, [selectedFile]);

  // Update plot when frame changes
  useEffect(() => {
    if (skeletonFrames.length > 0 && connections.length > 0) {
      formatPlotData(skeletonFrames[currentFrame], connections);
    }
  }, [currentFrame, skeletonFrames, connections]);

  // Playback loop
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying && skeletonFrames.length > 0) {
      interval = setInterval(() => {
        setCurrentFrame((prev) => (prev + 1) % skeletonFrames.length);
      }, 50); // ~20fps
    }
    return () => clearInterval(interval);
  }, [isPlaying, skeletonFrames.length]);

  // Format Plotly data for Light Mode (single frame)
  const formatPlotData = (frameData: any[], connections: any[]) => {
    if (!frameData || frameData.length === 0) return;
    const body1 = frameData[0]; 
    if (!body1) return;

    const x = body1.map((j: any[]) => j[0]);
    const y = body1.map((j: any[]) => j[2]); 
    const z = body1.map((j: any[]) => j[1]); 

    const traces: any[] = [];
    
    // Joints (Red dots for visibility)
    traces.push({
      x, y, z,
      mode: "markers",
      type: "scatter3d",
      marker: {
        size: 5,
        color: "#dc2626", // red-600
        opacity: 0.9,
        line: { width: 1, color: "#7f1d1d" }
      },
      name: "Joints"
    });

    // Bones (Dark blue/black lines)
    connections.forEach((conn: [number, number]) => {
      const idx1 = conn[0];
      const idx2 = conn[1];
      if (idx1 < x.length && idx2 < x.length) {
        traces.push({
          x: [x[idx1], x[idx2]],
          y: [y[idx1], y[idx2]],
          z: [z[idx1], z[idx2]],
          mode: "lines",
          type: "scatter3d",
          line: {
            color: "#1e293b", // slate-800
            width: 3
          },
          showlegend: false,
          hoverinfo: "none"
        });
      }
    });

    setJointsData(traces);
  };

  return (
    <div className="min-h-screen bg-[#fcfcfc] text-zinc-900 font-sans">
      
      {/* Academic-style Header */}
      <header className="border-b border-zinc-200 bg-white">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <div className="text-xs font-mono text-zinc-500 uppercase tracking-widest mb-3 border-b border-zinc-200 inline-block pb-1">
            Research Live Demo
          </div>
          <h1 className="text-3xl md:text-4xl font-serif text-zinc-900 leading-tight mb-2">
            Kinematic Analysis & Explainability of S-JEPA
          </h1>
          <p className="text-sm text-zinc-600 max-w-3xl">
            Interactive visualization of the Spatio-temporal Joint-Embedding Predictive Architecture on the NTU RGB+D dataset. Select a sequence to observe spatial coordinate transformations and inference outputs.
          </p>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-10 grid grid-cols-1 lg:grid-cols-[1fr_1.5fr] gap-10">
        
        {/* Left Column: Data Selection & Results */}
        <div className="flex flex-col gap-8">
          
          {/* Data Loader Panel */}
          <section className="bg-white border border-zinc-200 rounded-sm p-6 shadow-sm">
            <h2 className="text-sm font-bold uppercase tracking-wide text-zinc-800 flex items-center gap-2 mb-4">
              <Database className="w-4 h-4" />
              1. Data Source Selection
            </h2>
            <div className="space-y-2">
              <label className="text-xs text-zinc-600 font-medium">SKELETON FILE (.skeleton)</label>
              <select 
                value={selectedFile}
                onChange={(e) => setSelectedFile(e.target.value)}
                className="w-full bg-zinc-50 border border-zinc-200 text-sm px-3 py-2 focus:outline-none focus:ring-1 focus:ring-zinc-400 focus:border-zinc-400 font-mono disabled:opacity-50 transition-colors"
                disabled={loading || skeletons.length === 0}
              >
                {skeletons.length === 0 && <option>Fetching records...</option>}
                {skeletons.map(f => <option key={f} value={f}>{f}</option>)}
              </select>
            </div>
            {error && <p className="text-red-600 text-xs mt-3 flex items-center gap-1.5"><AlertCircle className="w-3.5 h-3.5" />{error}</p>}
          </section>

          {/* Inference Results Panel */}
          <section className="bg-white border border-zinc-200 rounded-sm shadow-sm flex flex-col">
            <div className="border-b border-zinc-200 p-4 bg-zinc-50">
              <h2 className="text-sm font-bold uppercase tracking-wide text-zinc-800 flex items-center gap-2">
                <BarChart2 className="w-4 h-4" />
                2. S-JEPA Inference Output
              </h2>
            </div>
            
            <div className="p-6 flex flex-col gap-6 flex-1 justify-center">
              <div>
                <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Predicted Action</p>
                <div className="text-xl md:text-2xl font-serif text-zinc-900 flex items-center gap-2">
                  {loading ? <span className="animate-pulse">Computing...</span> : prediction?.top_1 || "N/A"}
                  {!loading && prediction && <CheckCircle2 className="w-5 h-5 text-green-600" />}
                </div>
              </div>

              <div>
                <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Model Confidence</p>
                <div className="flex items-center gap-4">
                  <span className="text-2xl font-mono font-medium text-zinc-800">
                    {loading ? "--" : prediction ? `${(prediction.confidence * 100).toFixed(2)}%` : "--"}
                  </span>
                  {prediction && !loading && (
                    <div className="h-1.5 flex-1 bg-zinc-100 border border-zinc-200 rounded-none overflow-hidden">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${prediction.confidence * 100}%` }}
                        transition={{ duration: 0.8, ease: "easeOut" }}
                        className="h-full bg-zinc-800"
                      />
                    </div>
                  )}
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* Right Column: Plotly 3D Figure */}
        <section className="bg-white border border-zinc-200 rounded-sm shadow-sm flex flex-col min-h-[500px] lg:min-h-[600px]">
          <div className="border-b border-zinc-200 p-4 flex justify-between items-center bg-zinc-50">
            <h2 className="text-sm font-bold uppercase tracking-wide text-zinc-800 flex items-center gap-2">
              <FileText className="w-4 h-4" />
              Figure 1: Spatial Kinematic Structure
            </h2>
            <div className="text-xs text-zinc-500 font-mono">
              (Interactive 3D View)
            </div>
          </div>
          
          <div className="flex-1 relative bg-zinc-50/50">
            {loading ? (
              <div className="absolute inset-0 flex items-center justify-center text-zinc-400 text-sm font-mono">
                [ Rendering Spatial Data... ]
              </div>
            ) : jointsData.length > 0 ? (
               <Plot
                data={jointsData}
                revision={currentFrame}
                layout={{
                  autosize: true,
                  margin: { l: 0, r: 0, b: 0, t: 0 },
                  paper_bgcolor: "transparent",
                  scene: {
                    aspectmode: "cube",
                    camera: { eye: { x: 2.2, y: -2.2, z: 0.8 } },
                    xaxis: { range: [-1, 1], showbackground: false, showgrid: true, gridcolor: "#e4e4e7", zeroline: true, zerolinecolor: "#a1a1aa", title: "X", tickfont: {size: 10} },
                    yaxis: { range: [-1, 1], showbackground: false, showgrid: true, gridcolor: "#e4e4e7", zeroline: true, zerolinecolor: "#a1a1aa", title: "Depth", tickfont: {size: 10} },
                    zaxis: { range: [-1, 1.5], showbackground: false, showgrid: true, gridcolor: "#e4e4e7", zeroline: true, zerolinecolor: "#a1a1aa", title: "Height", tickfont: {size: 10} },
                  }
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%", height: "100%" }}
                useResizeHandler={true}
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-zinc-400 text-sm italic">
                No data loaded.
              </div>
            )}
          </div>

          {/* Animation Controls */}
          {skeletonFrames.length > 0 && !loading && (
            <div className="border-t border-zinc-200 p-4 bg-zinc-50 flex items-center gap-4">
              <button 
                onClick={() => setIsPlaying(!isPlaying)}
                className="w-8 h-8 flex items-center justify-center bg-zinc-200 hover:bg-zinc-300 text-zinc-800 rounded-sm transition-colors"
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              </button>
              <input 
                type="range" 
                min="0" 
                max={skeletonFrames.length - 1} 
                value={currentFrame}
                onChange={(e) => {
                  setIsPlaying(false);
                  setCurrentFrame(parseInt(e.target.value));
                }}
                className="flex-1 h-1.5 bg-zinc-200 rounded-lg appearance-none cursor-pointer accent-zinc-800"
              />
              <div className="text-xs font-mono text-zinc-500 w-16 text-right">
                {currentFrame + 1} / {skeletonFrames.length}
              </div>
            </div>
          )}
        </section>

      </main>
    </div>
  );
}
