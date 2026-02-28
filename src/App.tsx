import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Camera, 
  Upload, 
  History, 
  Settings, 
  BarChart3, 
  Zap, 
  Volume2, 
  VolumeX, 
  Shield, 
  Search, 
  AlertCircle,
  CheckCircle2,
  Trash2,
  Download,
  Play,
  Pause,
  Maximize2,
  Mic,
  Eye,
  Activity,
  ChevronRight,
  Loader2,
  Info,
  Cpu,
  Monitor
} from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  AreaChart, 
  Area 
} from 'recharts';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { AppView, Detection, MediaAnalysis, VoiceSettings, RiskLevel, VisualMode } from './types';
import { cn, speak, formatTimestamp } from './utils';
import { analyzeMedia } from './services/ai';

const MOCK_HISTORY: MediaAnalysis[] = [
  {
    id: '1',
    type: 'image',
    url: 'https://picsum.photos/seed/vision1/800/600',
    name: 'security_cam_01.jpg',
    detections: [
      { id: 'd1', label: 'Person', confidence: 0.98, timestamp: Date.now() - 3600000, type: 'Object' },
      { id: 'd2', label: 'Vehicle', confidence: 0.92, timestamp: Date.now() - 3600000, type: 'Object' }
    ],
    summary: 'A person is walking near a parked vehicle in a residential area.',
    timestamp: Date.now() - 3600000
  }
];

const MOCK_ANALYTICS_DATA = [
  { time: '08:00', objects: 12, threats: 1 },
  { time: '09:00', objects: 24, threats: 0 },
  { time: '10:00', objects: 18, threats: 2 },
  { time: '11:00', objects: 32, threats: 1 },
  { time: '12:00', objects: 45, threats: 3 },
  { time: '13:00', objects: 28, threats: 0 },
  { time: '14:00', objects: 36, threats: 1 },
];

export default function App() {
  const [view, setView] = useState<AppView>('Live');
  const [history, setHistory] = useState<MediaAnalysis[]>(MOCK_HISTORY);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [voiceSettings, setVoiceSettings] = useState<VoiceSettings>({
    enabled: true,
    voice: 'default',
    pitch: 1,
    rate: 1
  });
  const [liveDetections, setLiveDetections] = useState<Detection[]>([]);
  const [visualMode, setVisualMode] = useState<VisualMode>('RGB');
  const [steeringGuidance, setSteeringGuidance] = useState<string>('CENTER');
  const [isRecording, setIsRecording] = useState(false);
  const [sessionTime, setSessionTime] = useState(0);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const detectionHistoryRef = useRef<Record<string, number[]>>({});
  const requestRef = useRef<number>(null);

  // Load Model
  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready();
        const loadedModel = await cocoSsd.load();
        setModel(loadedModel);
        setIsModelLoading(false);
      } catch (err) {
        console.error("Model loading failed", err);
      }
    };
    loadModel();
  }, []);

  // Real-world heights (meters) for distance estimation
  const OBJECT_DIMENSIONS: Record<string, number> = {
    'person': 1.7,
    'car': 1.5,
    'truck': 2.5,
    'bus': 3.0,
    'bicycle': 1.0,
    'motorcycle': 1.0,
    'laptop': 0.25,
    'cell phone': 0.15,
    'backpack': 0.5,
    'handbag': 0.3
  };

  // Focal length estimation (pixels) - typical for 720p webcam
  const FOCAL_LENGTH = 700; 

  // Temporal Smoothing (Moving Average)
  const smoothDistance = (label: string, newDistance: number) => {
    if (!detectionHistoryRef.current[label]) {
      detectionHistoryRef.current[label] = [];
    }
    const history = detectionHistoryRef.current[label];
    history.push(newDistance);
    if (history.length > 10) history.shift();
    
    const sum = history.reduce((a, b) => a + b, 0);
    return parseFloat((sum / history.length).toFixed(2));
  };

  const [isCameraActive, setIsCameraActive] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Voice Alert Logic
  const triggerVoiceAlert = useCallback((text: string) => {
    if (voiceSettings.enabled) {
      speak(text, { pitch: voiceSettings.pitch, rate: voiceSettings.rate });
    }
  }, [voiceSettings]);

  // Session Timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRecording) {
      interval = setInterval(() => setSessionTime(prev => prev + 1), 1000);
    } else {
      setSessionTime(0);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  const detectFrame = useCallback(async () => {
    if (!model || !videoRef.current || !isCameraActive) return;

    // Ensure video is ready and has valid dimensions to prevent [0x0] texture error
    if (videoRef.current.readyState < 2 || videoRef.current.videoWidth === 0 || videoRef.current.videoHeight === 0) {
      requestRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    try {
      const predictions = await model.detect(videoRef.current);
      const videoWidth = videoRef.current.videoWidth;
      
      let leftRisk = false;
      let rightRisk = false;
      let centerRisk = false;

      const newDetections: Detection[] = predictions.map((pred, idx) => {
        const label = pred.class;
        const confidence = pred.score;
        const [x, y, w, h] = pred.bbox;

        // Distance Estimation
        const realHeight = OBJECT_DIMENSIONS[label] || 0.5;
        const rawDistance = (realHeight * FOCAL_LENGTH) / h;
        const distance = smoothDistance(label, rawDistance);

        let risk: RiskLevel = 'SAFE';
        if (distance < 1.5) risk = 'DANGER';
        else if (distance < 3.5) risk = 'WARNING';

        // Steering Logic
        const centerX = x + w / 2;
        if (risk !== 'SAFE') {
          if (centerX < videoWidth * 0.33) leftRisk = true;
          else if (centerX > videoWidth * 0.66) rightRisk = true;
          else centerRisk = true;
        }

        return {
          id: `${label}-${idx}`,
          label,
          confidence,
          timestamp: Date.now(),
          type: 'Object',
          distance,
          risk,
          box: { x, y, w, h }
        };
      });

      // Guidance Decision
      if (centerRisk) setSteeringGuidance('STOP');
      else if (leftRisk && !rightRisk) setSteeringGuidance('STEER RIGHT');
      else if (rightRisk && !leftRisk) setSteeringGuidance('STEER LEFT');
      else setSteeringGuidance('CENTER');

      setLiveDetections(newDetections);

      // Voice Alert for Danger
      const dangerObj = newDetections.find(d => d.risk === 'DANGER');
      if (dangerObj && Date.now() % 3000 < 100) { // Throttle voice alerts
        triggerVoiceAlert(`Danger! ${dangerObj.label} at ${dangerObj.distance} meters.`);
      }

      requestRef.current = requestAnimationFrame(detectFrame);
    } catch (err) {
      console.error("Detection error", err);
    }
  }, [model, isCameraActive, triggerVoiceAlert]);

  useEffect(() => {
    if (isCameraActive && model) {
      requestRef.current = requestAnimationFrame(detectFrame);
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isCameraActive, model, detectFrame]);

  const toggleCamera = async () => {
    if (!isCameraActive) {
      try {
        const newStream = await navigator.mediaDevices.getUserMedia({ 
          video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } } 
        });
        setStream(newStream);
        setIsCameraActive(true);
      } catch (err) {
        console.error("Camera access failed", err);
        alert("Failed to access camera. Please ensure permissions are granted.");
      }
    } else {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      setStream(null);
      setIsCameraActive(false);
    }
  };

  useEffect(() => {
    if (isCameraActive && stream && videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(err => console.error("Video play failed", err));
    }
  }, [isCameraActive, stream]);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setIsAnalyzing(true);
    const type = file.type.startsWith('video') ? 'video' : 'image';
    const url = URL.createObjectURL(file);

    try {
      const result = await analyzeMedia(file, type);
      const newAnalysis: MediaAnalysis = {
        id: Math.random().toString(36).substr(2, 9),
        type,
        url,
        name: file.name,
        detections: result.detections.map((d: any) => ({
          ...d,
          id: Math.random().toString(36).substr(2, 9),
          timestamp: Date.now(),
          type: 'Object'
        })),
        summary: result.summary,
        timestamp: Date.now()
      };

      setHistory(prev => [newAnalysis, ...prev]);
      setView('History');
      triggerVoiceAlert(`Analysis complete for ${file.name}. ${result.summary}`);
    } catch (err) {
      console.error("Analysis failed", err);
    } finally {
      setIsAnalyzing(false);
    }
  }, [triggerVoiceAlert]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp'],
      'video/*': ['.mp4', '.webm', '.mov']
    },
    multiple: false
  });

  return (
    <div className="flex h-screen bg-[#0A0A0A] text-zinc-300 font-sans selection:bg-indigo-500/30 overflow-hidden">
      {/* Sidebar Navigation */}
      <aside className="w-64 border-r border-white/5 bg-black/40 backdrop-blur-xl flex flex-col p-6 z-50">
        <div className="flex items-center gap-3 mb-12">
          <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Shield className="w-6 h-6 text-white fill-current" />
          </div>
          <div>
            <h1 className="font-bold text-white tracking-tight text-lg">AEGIS AI</h1>
            <p className="text-[10px] text-zinc-500 uppercase tracking-widest font-bold">Vision Intelligence</p>
          </div>
        </div>

        <nav className="flex-1 space-y-2">
          {[
            { id: 'Live', icon: Camera, label: 'Live Sentinel' },
            { id: 'Upload', icon: Upload, label: 'Media Lab' },
            { id: 'History', icon: History, label: 'Intelligence Log' },
            { id: 'Analytics', icon: BarChart3, label: 'Insights' },
            { id: 'Settings', icon: Settings, label: 'System Config' },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => setView(item.id as AppView)}
              className={cn(
                "w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all group",
                view === item.id 
                  ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/20" 
                  : "hover:bg-white/5 text-zinc-500 hover:text-zinc-300"
              )}
            >
              <item.icon className={cn("w-5 h-5", view === item.id ? "text-white" : "text-zinc-500 group-hover:text-zinc-300")} />
              {item.label}
              {item.id === 'Live' && isCameraActive && (
                <div className="ml-auto w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              )}
            </button>
          ))}
        </nav>

        <div className="mt-auto pt-6 border-t border-white/5 space-y-4">
          <div className="bg-zinc-900/50 rounded-xl p-4 border border-white/5">
            <div className="flex items-center gap-2 mb-3">
              <Cpu className="w-4 h-4 text-indigo-500" />
              <span className="text-[10px] font-bold text-white uppercase tracking-widest">System Health</span>
            </div>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-[8px] text-zinc-500 uppercase mb-1">
                  <span>Neural Load</span>
                  <span className="text-emerald-500">24%</span>
                </div>
                <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                  <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: '24%' }}
                    className="h-full bg-emerald-500" 
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-[8px] text-zinc-500 uppercase mb-1">
                  <span>Memory</span>
                  <span className="text-indigo-500">1.2 GB</span>
                </div>
                <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                  <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: '45%' }}
                    className="h-full bg-indigo-500" 
                  />
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-zinc-900/50 rounded-xl p-4 border border-white/5">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="w-4 h-4 text-amber-400" />
              <span className="text-[10px] font-bold text-white uppercase tracking-widest">Neural Status</span>
            </div>
            <div className="flex items-center justify-between text-[10px] text-zinc-500">
              <span>Engine v4.2</span>
              <span className="text-emerald-500">Optimal</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 relative overflow-y-auto custom-scrollbar">
        <header className="h-20 border-b border-white/5 flex items-center justify-between px-8 sticky top-0 bg-[#0A0A0A]/80 backdrop-blur-md z-40">
          <div className="flex items-center gap-4">
            <h2 className="text-xl font-bold text-white">{view}</h2>
            <div className="h-4 w-px bg-white/10" />
            <div className="flex items-center gap-2 text-xs text-zinc-500">
              <Activity className="w-4 h-4 text-indigo-500" />
              <span>Real-time processing active</span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <button 
              onClick={() => setVoiceSettings(prev => ({ ...prev, enabled: !prev.enabled }))}
              className={cn(
                "p-2 rounded-full transition-all",
                voiceSettings.enabled ? "bg-indigo-500/10 text-indigo-500" : "bg-zinc-800 text-zinc-500"
              )}
            >
              {voiceSettings.enabled ? <Volume2 className="w-5 h-5" /> : <VolumeX className="w-5 h-5" />}
            </button>
            <div className="w-10 h-10 rounded-full bg-zinc-800 border border-white/5 flex items-center justify-center overflow-hidden">
              <img src="https://api.dicebear.com/7.x/avataaars/svg?seed=Aegis" alt="User" className="w-full h-full object-cover" />
            </div>
          </div>
        </header>

        <div className="p-8">
          <AnimatePresence mode="wait">
            {view === 'Live' && (
              <motion.div 
                key="live"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-6"
              >
                <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
                  <div className="lg:col-span-3 space-y-6">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Primary Input</span>
                        <h3 className="text-xl font-bold text-white">Live Feed Analysis</h3>
                      </div>
                      <div className="flex items-center gap-2 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                        <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-widest">Model Ready</span>
                      </div>
                    </div>

                    <div className={cn(
                      "aspect-video bg-zinc-900 rounded-lg border border-white/10 overflow-hidden relative group",
                      visualMode === 'GRAYSCALE' && "grayscale",
                      visualMode === 'NIGHT' && "brightness-150 contrast-125 sepia hue-rotate-[90deg] saturate-200",
                      visualMode === 'THERMAL' && "invert hue-rotate-[180deg] saturate-200 contrast-150"
                    )}>
                      {isModelLoading && (
                        <div className="absolute inset-0 z-50 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center gap-4">
                          <Loader2 className="w-10 h-10 text-indigo-500 animate-spin" />
                          <p className="text-zinc-400 text-sm font-bold uppercase tracking-widest">Loading Neural Engine...</p>
                        </div>
                      )}
                      
                      {!isCameraActive ? (
                        <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
                          <div className="w-20 h-20 bg-zinc-800 rounded-full flex items-center justify-center">
                            <Camera className="w-10 h-10 text-zinc-500" />
                          </div>
                          <p className="text-zinc-500 text-sm">Camera feed inactive</p>
                          <button 
                            onClick={toggleCamera}
                            className="px-6 py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl font-bold text-sm transition-all shadow-lg shadow-indigo-600/20"
                          >
                            Initialize Sentinel
                          </button>
                        </div>
                      ) : (
                        <>
                          <video 
                            ref={videoRef} 
                            autoPlay 
                            playsInline 
                            muted
                            className="w-full h-full object-cover" 
                          />

                          <div className="absolute top-6 left-6 flex items-center gap-3">
                            <div className="bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 flex items-center gap-2">
                              <div className={cn("w-2 h-2 rounded-full animate-pulse", isRecording ? "bg-rose-500" : "bg-zinc-500")} />
                              <span className="text-[10px] font-bold text-white uppercase tracking-widest">
                                {isRecording ? `REC ${new Date(sessionTime * 1000).toISOString().substr(14, 5)}` : 'STANDBY'}
                              </span>
                            </div>
                            <div className="bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 text-[10px] font-bold text-white uppercase tracking-widest">
                              {visualMode} MODE
                            </div>
                          </div>

                          <div className="absolute bottom-6 right-6 flex gap-3">
                            <button 
                              onClick={() => setIsRecording(!isRecording)}
                              className={cn(
                                "p-3 rounded-xl transition-all shadow-lg pointer-events-auto",
                                isRecording ? "bg-rose-600 text-white shadow-rose-600/20" : "bg-zinc-800 text-zinc-400 hover:text-white"
                              )}
                              title={isRecording ? "Stop Recording" : "Start Recording"}
                            >
                              <Play className={cn("w-5 h-5", isRecording && "fill-current")} />
                            </button>
                            <button 
                              onClick={toggleCamera}
                              className="p-3 bg-zinc-800 hover:bg-rose-600 text-zinc-400 hover:text-white rounded-xl transition-all shadow-lg pointer-events-auto"
                              title="Stop Sentinel"
                            >
                              <VolumeX className="w-5 h-5" />
                            </button>
                          </div>
                          
                          {/* HUD Overlays */}
                          <div className="absolute inset-0 pointer-events-none">
                            {/* Lane Lines */}
                            <svg className="absolute inset-0 w-full h-full opacity-30">
                              <path d="M 300 720 L 500 400" stroke="cyan" strokeWidth="2" fill="none" />
                              <path d="M 980 720 L 780 400" stroke="cyan" strokeWidth="2" fill="none" />
                            </svg>

                            {/* Center Reticle */}
                            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-12 h-12 border border-white/20 rounded-full flex items-center justify-center">
                              <div className="w-1 h-1 bg-white/40 rounded-full" />
                            </div>

                            {/* Steering Guidance HUD */}
                            <div className="absolute top-1/2 left-12 -translate-y-1/2 space-y-4">
                              <div className={cn(
                                "w-1 h-16 rounded-full transition-all duration-500",
                                steeringGuidance === 'STEER LEFT' ? "bg-indigo-500 shadow-[0_0_15px_rgba(99,102,241,0.8)] scale-y-125" : "bg-white/10"
                              )} />
                            </div>
                            <div className="absolute top-1/2 right-12 -translate-y-1/2 space-y-4">
                              <div className={cn(
                                "w-1 h-16 rounded-full transition-all duration-500",
                                steeringGuidance === 'STEER RIGHT' ? "bg-indigo-500 shadow-[0_0_15px_rgba(99,102,241,0.8)] scale-y-125" : "bg-white/10"
                              )} />
                            </div>

                            {/* Speedometer & Compass HUD */}
                            <div className="absolute bottom-6 left-6 flex gap-4">
                              <div className="bg-black/60 backdrop-blur-md p-3 rounded-xl border border-white/10 flex flex-col items-center">
                                <span className="text-[8px] font-bold text-zinc-500 uppercase tracking-widest mb-1">Speed</span>
                                <span className="text-xl font-black text-white italic">42 <span className="text-[10px] not-italic text-zinc-500">KM/H</span></span>
                              </div>
                              <div className="bg-black/60 backdrop-blur-md p-3 rounded-xl border border-white/10 flex flex-col items-center">
                                <span className="text-[8px] font-bold text-zinc-500 uppercase tracking-widest mb-1">Compass</span>
                                <span className="text-xl font-black text-white italic">NW <span className="text-[10px] not-italic text-zinc-500">312°</span></span>
                              </div>
                            </div>

                            {/* Guidance Text */}
                            <div className="absolute bottom-24 left-1/2 -translate-x-1/2">
                              <motion.div 
                                key={steeringGuidance}
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className={cn(
                                  "px-6 py-2 rounded-full border backdrop-blur-md text-xs font-black uppercase tracking-[0.2em]",
                                  steeringGuidance === 'STOP' ? "bg-rose-500/20 border-rose-500 text-rose-500" :
                                  steeringGuidance !== 'CENTER' ? "bg-amber-500/20 border-amber-500 text-amber-500" :
                                  "bg-emerald-500/20 border-emerald-500 text-emerald-500"
                                )}
                              >
                                {steeringGuidance}
                              </motion.div>
                            </div>
                          </div>

                          {/* Bounding Box Overlays */}
                          {liveDetections.map((det) => {
                            const videoWidth = videoRef.current?.videoWidth || 1280;
                            const videoHeight = videoRef.current?.videoHeight || 720;
                            
                            return (
                              <div 
                                key={det.id}
                                className={cn(
                                  "absolute border-2 transition-all duration-75",
                                  det.risk === 'DANGER' ? "border-rose-500 bg-rose-500/10" : 
                                  det.risk === 'WARNING' ? "border-amber-500 bg-amber-500/10" : "border-emerald-500/30 bg-emerald-500/5"
                                )}
                                style={{
                                  left: `${((det.box?.x || 0) / videoWidth) * 100}%`,
                                  top: `${((det.box?.y || 0) / videoHeight) * 100}%`,
                                  width: `${((det.box?.w || 0) / videoWidth) * 100}%`,
                                  height: `${((det.box?.h || 0) / videoHeight) * 100}%`,
                                }}
                              >
                              <div className={cn(
                                "absolute -top-6 left-0 px-2 py-0.5 text-[10px] font-bold uppercase tracking-tighter whitespace-nowrap",
                                det.risk === 'DANGER' ? "bg-rose-500 text-white" : 
                                det.risk === 'WARNING' ? "bg-amber-500 text-black" : "bg-emerald-500/50 text-white"
                              )}>
                                {det.label} | {(det.confidence * 100).toFixed(0)}% | {det.distance}m
                              </div>
                              {det.risk === 'DANGER' && (
                                <div className="absolute inset-0 flex items-center justify-center">
                                  <span className="text-[10px] font-black text-rose-500 uppercase animate-pulse">Obstacle Detected</span>
                                </div>
                              )}
                            </div>
                          );
                        })}

                          {/* Scanning Effect */}
                          <div className="absolute inset-0 overflow-hidden pointer-events-none">
                            <motion.div 
                              initial={{ top: '-10%' }}
                              animate={{ top: '110%' }}
                              transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                              className="absolute left-0 right-0 h-1 bg-indigo-500/20 shadow-[0_0_15px_rgba(99,102,241,0.4)] z-10"
                            />
                          </div>
                        </>
                      )}
                    </div>

                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-rose-600/90 p-4 rounded-lg border border-rose-500/20">
                        <div className="text-[10px] font-bold text-white uppercase tracking-widest mb-1">Danger Zone</div>
                        <div className="flex items-center justify-between">
                          <span className="text-3xl font-black text-white italic">{liveDetections.filter(d => d.risk === 'DANGER').length}</span>
                          <AlertCircle className="w-5 h-5 text-white/50" />
                        </div>
                      </div>
                      <div className="bg-amber-500/90 p-4 rounded-lg border border-amber-400/20">
                        <div className="text-[10px] font-bold text-black uppercase tracking-widest mb-1">Warning Zone</div>
                        <div className="flex items-center justify-between">
                          <span className="text-3xl font-black text-black italic">{liveDetections.filter(d => d.risk === 'WARNING').length}</span>
                          <Info className="w-5 h-5 text-black/50" />
                        </div>
                      </div>
                      <div className="bg-zinc-800/50 p-4 rounded-lg border border-white/5">
                        <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-1">Safe Objects</div>
                        <div className="flex items-center justify-between">
                          <span className="text-3xl font-black text-white italic">{liveDetections.filter(d => d.risk === 'SAFE').length}</span>
                          <Shield className="w-5 h-5 text-zinc-500" />
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-6">
                    <div className="bg-zinc-900/50 rounded-lg border border-white/5 p-6 flex flex-col h-[400px]">
                      <div className="flex items-center gap-2 mb-6">
                        <Activity className="w-4 h-4 text-indigo-500" />
                        <h3 className="text-sm font-bold text-white uppercase tracking-widest">Real-Time Telemetry</h3>
                      </div>
                      
                      <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
                        <table className="w-full text-[10px] text-left">
                          <thead className="text-zinc-500 uppercase tracking-widest border-b border-white/5">
                            <tr>
                              <th className="pb-2 font-bold">Object</th>
                              <th className="pb-2 font-bold">Conf.</th>
                              <th className="pb-2 font-bold">Dist.</th>
                              <th className="pb-2 font-bold">Status</th>
                            </tr>
                          </thead>
                          <tbody className="text-zinc-300">
                            {liveDetections.map((det) => (
                              <tr key={det.id} className="border-b border-white/5">
                                <td className="py-3 font-bold">{det.label}</td>
                                <td className="py-3">{(det.confidence * 100).toFixed(0)}%</td>
                                <td className="py-3">{det.distance}m</td>
                                <td className={cn(
                                  "py-3 font-bold",
                                  det.risk === 'DANGER' ? "text-rose-500" : 
                                  det.risk === 'WARNING' ? "text-amber-500" : "text-emerald-500"
                                )}>
                                  {det.risk}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    <div className="bg-zinc-900/50 rounded-lg border border-white/5 p-6">
                      <div className="flex items-center gap-2 mb-4">
                        <Monitor className="w-4 h-4 text-indigo-500" />
                        <h3 className="text-sm font-bold text-white uppercase tracking-widest">Visual Configuration</h3>
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        {(['RGB', 'GRAYSCALE', 'NIGHT', 'THERMAL'] as VisualMode[]).map((mode) => (
                          <button
                            key={mode}
                            onClick={() => setVisualMode(mode)}
                            className={cn(
                              "px-3 py-2 rounded-lg text-[10px] font-bold uppercase tracking-widest transition-all border",
                              visualMode === mode 
                                ? "bg-indigo-600 border-indigo-500 text-white" 
                                : "bg-zinc-800 border-white/5 text-zinc-500 hover:text-zinc-300"
                            )}
                          >
                            {mode}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="bg-zinc-900/50 rounded-lg border border-white/5 p-6">
                      <div className="flex items-center gap-2 mb-4">
                        <Info className="w-4 h-4 text-indigo-500" />
                        <h3 className="text-sm font-bold text-white uppercase tracking-widest">Technical Specifications</h3>
                      </div>
                      
                      <div className="space-y-4">
                        <div>
                          <div className="text-[8px] font-bold text-zinc-500 uppercase tracking-widest mb-1">Mathematical Model</div>
                          <p className="text-[10px] text-zinc-400 leading-relaxed">
                            This system utilizes the <span className="italic">Triangle Similarity</span> theorem for monocular distance estimation.
                          </p>
                          <div className="mt-2 p-3 bg-black/40 rounded border border-white/5 font-mono text-[10px] text-white">
                            D = (W x F) / P
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-2 text-[8px]">
                          <div className="text-zinc-500"><span className="font-bold text-white">D:</span> Distance to object</div>
                          <div className="text-zinc-500"><span className="font-bold text-white">W:</span> Real-world width</div>
                          <div className="text-zinc-500"><span className="font-bold text-white">F:</span> Focal length</div>
                          <div className="text-zinc-500"><span className="font-bold text-white">P:</span> Perceived width</div>
                        </div>

                        <div className="pt-4 border-t border-white/5">
                          <div className="text-[8px] font-bold text-zinc-500 uppercase tracking-widest mb-2">Hardware Requirements</div>
                          <div className="grid grid-cols-2 gap-4">
                            <div className="p-2 bg-white/5 rounded border border-white/5">
                              <div className="text-[10px] font-bold text-white">Edge Device</div>
                              <div className="text-[8px] text-zinc-500">Jetson Nano / RPi 5</div>
                            </div>
                            <div className="p-2 bg-white/5 rounded border border-white/5">
                              <div className="text-[10px] font-bold text-white">Web Browser</div>
                              <div className="text-[8px] text-zinc-500">Chrome / Edge (WASM)</div>
                            </div>
                          </div>
                        </div>

                        <div className="pt-4 border-t border-white/5">
                          <div className="text-[8px] font-bold text-zinc-500 uppercase tracking-widest mb-2">Accuracy Techniques</div>
                          <ul className="text-[10px] text-zinc-400 space-y-1">
                            <li className="flex items-center gap-2">
                              <div className="w-1 h-1 rounded-full bg-indigo-500" />
                              Temporal smoothing (Moving Average)
                            </li>
                            <li className="flex items-center gap-2">
                              <div className="w-1 h-1 rounded-full bg-indigo-500" />
                              Object-specific width profiles
                            </li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {view === 'Upload' && (
              <motion.div 
                key="upload"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="max-w-4xl mx-auto"
              >
                <div 
                  {...getRootProps()} 
                  className={cn(
                    "aspect-[21/9] rounded-3xl border-2 border-dashed transition-all flex flex-col items-center justify-center gap-6 cursor-pointer group",
                    isDragActive ? "border-indigo-500 bg-indigo-500/5" : "border-white/10 hover:border-white/20 bg-zinc-900/50"
                  )}
                >
                  <input {...getInputProps()} />
                  <div className="w-20 h-20 bg-zinc-800 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
                    {isAnalyzing ? (
                      <Loader2 className="w-10 h-10 text-indigo-500 animate-spin" />
                    ) : (
                      <Upload className="w-10 h-10 text-zinc-500 group-hover:text-indigo-500 transition-colors" />
                    )}
                  </div>
                  <div className="text-center">
                    <h3 className="text-xl font-bold text-white mb-2">
                      {isAnalyzing ? 'Analyzing Intelligence...' : 'Drop Media for Analysis'}
                    </h3>
                    <p className="text-zinc-500 text-sm">Support for high-res images and 4K video streams</p>
                  </div>
                  {!isAnalyzing && (
                    <button className="px-8 py-3 bg-white text-black rounded-xl font-bold text-sm hover:bg-zinc-200 transition-all">
                      Browse Files
                    </button>
                  )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-12">
                  <div className="bg-zinc-900/50 p-8 rounded-3xl border border-white/5 space-y-4">
                    <div className="w-12 h-12 bg-indigo-500/10 rounded-xl flex items-center justify-center">
                      <Shield className="w-6 h-6 text-indigo-500" />
                    </div>
                    <h4 className="text-lg font-bold text-white">Deep Scene Analysis</h4>
                    <p className="text-sm text-zinc-500 leading-relaxed">
                      Our neural engine performs multi-pass analysis to identify complex relationships between objects and environmental factors.
                    </p>
                  </div>
                  <div className="bg-zinc-900/50 p-8 rounded-3xl border border-white/5 space-y-4">
                    <div className="w-12 h-12 bg-amber-500/10 rounded-xl flex items-center justify-center">
                      <Zap className="w-6 h-6 text-amber-500" />
                    </div>
                    <h4 className="text-lg font-bold text-white">Temporal Video Tracking</h4>
                    <p className="text-sm text-zinc-500 leading-relaxed">
                      For video uploads, Aegis tracks object trajectories over time to detect anomalies and predict potential security breaches.
                    </p>
                  </div>
                </div>
              </motion.div>
            )}

            {view === 'History' && (
              <motion.div 
                key="history"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-12"
              >
                <div className="flex items-center justify-between">
                  <div className="relative w-96">
                    <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
                    <input 
                      type="text" 
                      placeholder="Search intelligence log..." 
                      className="w-full bg-zinc-900 border border-white/5 rounded-xl py-3 pl-12 pr-4 text-sm focus:outline-none focus:border-indigo-500 transition-all"
                    />
                  </div>
                  <button className="flex items-center gap-2 text-xs font-bold text-zinc-500 hover:text-white transition-colors">
                    <Download className="w-4 h-4" />
                    Export All Data
                  </button>
                </div>

                <div className="space-y-6">
                  <h3 className="text-sm font-bold text-white uppercase tracking-widest flex items-center gap-2">
                    <Play className="w-4 h-4 text-indigo-500" />
                    Recent Recordings
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {[1, 2, 3].map((i) => (
                      <div key={i} className="bg-zinc-900/50 rounded-2xl border border-white/5 overflow-hidden group cursor-pointer hover:border-indigo-500/30 transition-all">
                        <div className="aspect-video relative overflow-hidden">
                          <img 
                            src={`https://picsum.photos/seed/drive${i}/640/360`} 
                            alt="Recording" 
                            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                          />
                          <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                            <Play className="w-10 h-10 text-white fill-current" />
                          </div>
                          <div className="absolute top-3 left-3 bg-black/60 backdrop-blur-md px-2 py-1 rounded text-[8px] font-bold text-white uppercase">
                            04:2{i} MIN
                          </div>
                        </div>
                        <div className="p-4">
                          <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-1">Session ID: #REC-829{i}</div>
                          <div className="text-xs font-bold text-white">Urban Drive Analysis - Route {i}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="space-y-6">
                  <h3 className="text-sm font-bold text-white uppercase tracking-widest flex items-center gap-2">
                    <Monitor className="w-4 h-4 text-indigo-500" />
                    Intelligence Log
                  </h3>
                  <div className="grid grid-cols-1 gap-4">
                    {history.map((item) => (
                    <div key={item.id} className="bg-zinc-900/50 rounded-2xl border border-white/5 p-6 flex gap-8 group hover:bg-zinc-900 transition-all">
                      <div className="w-48 aspect-video rounded-xl overflow-hidden border border-white/10 relative">
                        <img src={item.url} alt={item.name} className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500" />
                        <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                          <Maximize2 className="w-6 h-6 text-white" />
                        </div>
                      </div>
                      
                      <div className="flex-1 space-y-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <h4 className="font-bold text-white">{item.name}</h4>
                            <p className="text-xs text-zinc-500">{formatTimestamp(item.timestamp)} • {item.type.toUpperCase()}</p>
                          </div>
                          <div className="flex gap-2">
                            <button className="p-2 hover:bg-white/5 rounded-lg transition-colors"><Download className="w-4 h-4" /></button>
                            <button className="p-2 hover:bg-rose-500/10 text-rose-500 rounded-lg transition-colors"><Trash2 className="w-4 h-4" /></button>
                          </div>
                        </div>
                        
                        <p className="text-sm text-zinc-400 leading-relaxed italic">"{item.summary}"</p>
                        
                        <div className="flex flex-wrap gap-2">
                          {item.detections.map((det) => (
                            <span key={det.id} className="px-3 py-1 bg-indigo-500/10 text-indigo-400 rounded-full text-[10px] font-bold border border-indigo-500/20">
                              {det.label} ({(det.confidence * 100).toFixed(0)}%)
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

            {view === 'Analytics' && (
              <motion.div 
                key="analytics"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-8"
              >
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div className="bg-zinc-900/50 p-8 rounded-3xl border border-white/5">
                    <h3 className="text-sm font-bold text-white uppercase tracking-widest mb-8">Detection Volume (24h)</h3>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={MOCK_ANALYTICS_DATA}>
                          <defs>
                            <linearGradient id="colorObjects" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
                              <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                          <XAxis dataKey="time" stroke="#ffffff20" fontSize={10} tickLine={false} axisLine={false} />
                          <YAxis stroke="#ffffff20" fontSize={10} tickLine={false} axisLine={false} />
                          <Tooltip 
                            contentStyle={{ backgroundColor: '#18181b', border: '1px solid #ffffff10', borderRadius: '12px' }}
                            itemStyle={{ color: '#fff', fontSize: '12px' }}
                          />
                          <Area type="monotone" dataKey="objects" stroke="#6366f1" fillOpacity={1} fill="url(#colorObjects)" strokeWidth={2} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="bg-zinc-900/50 p-8 rounded-3xl border border-white/5">
                    <h3 className="text-sm font-bold text-white uppercase tracking-widest mb-8">Threat Incidents</h3>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={MOCK_ANALYTICS_DATA}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                          <XAxis dataKey="time" stroke="#ffffff20" fontSize={10} tickLine={false} axisLine={false} />
                          <YAxis stroke="#ffffff20" fontSize={10} tickLine={false} axisLine={false} />
                          <Tooltip 
                            contentStyle={{ backgroundColor: '#18181b', border: '1px solid #ffffff10', borderRadius: '12px' }}
                            itemStyle={{ color: '#fff', fontSize: '12px' }}
                          />
                          <Line type="monotone" dataKey="threats" stroke="#f43f5e" strokeWidth={2} dot={{ fill: '#f43f5e', r: 4 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                  {[
                    { label: 'Total Scans', value: '1,284', trend: '+12%', icon: Search },
                    { label: 'Avg Confidence', value: '94.2%', trend: '+2%', icon: CheckCircle2 },
                    { label: 'False Positives', value: '0.4%', trend: '-1%', icon: AlertCircle },
                    { label: 'Uptime', value: '99.99%', trend: 'Stable', icon: Shield },
                  ].map((stat, i) => (
                    <div key={i} className="bg-zinc-900/50 p-6 rounded-2xl border border-white/5">
                      <div className="flex items-center justify-between mb-4">
                        <stat.icon className="w-5 h-5 text-zinc-500" />
                        <span className={cn(
                          "text-[10px] font-bold px-2 py-0.5 rounded-full",
                          stat.trend.startsWith('+') ? "bg-emerald-500/10 text-emerald-500" : 
                          stat.trend.startsWith('-') ? "bg-rose-500/10 text-rose-500" : "bg-zinc-800 text-zinc-500"
                        )}>
                          {stat.trend}
                        </span>
                      </div>
                      <div className="text-2xl font-bold text-white mb-1">{stat.value}</div>
                      <div className="text-[10px] text-zinc-500 uppercase tracking-widest font-bold">{stat.label}</div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}

            {view === 'Settings' && (
              <motion.div 
                key="settings"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="max-w-2xl space-y-8"
              >
                <section className="space-y-6">
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <Volume2 className="w-5 h-5 text-indigo-500" />
                    Voice Synthesis Alerts
                  </h3>
                  <div className="bg-zinc-900/50 p-8 rounded-3xl border border-white/5 space-y-8">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-bold text-white">Enable Voice Alerts</h4>
                        <p className="text-xs text-zinc-500">Synthesize speech for real-time detections</p>
                      </div>
                      <button 
                        onClick={() => setVoiceSettings(prev => ({ ...prev, enabled: !prev.enabled }))}
                        className={cn(
                          "w-12 h-6 rounded-full transition-all relative",
                          voiceSettings.enabled ? "bg-indigo-600" : "bg-zinc-800"
                        )}
                      >
                        <div className={cn(
                          "absolute top-1 w-4 h-4 rounded-full bg-white transition-all",
                          voiceSettings.enabled ? "left-7" : "left-1"
                        )} />
                      </button>
                    </div>

                    <div className="space-y-4">
                      <div className="flex justify-between text-xs font-bold uppercase tracking-widest text-zinc-500">
                        <span>Pitch</span>
                        <span className="text-white">{voiceSettings.pitch.toFixed(1)}</span>
                      </div>
                      <input 
                        type="range" min="0.5" max="2" step="0.1" 
                        value={voiceSettings.pitch}
                        onChange={(e) => setVoiceSettings(prev => ({ ...prev, pitch: parseFloat(e.target.value) }))}
                        className="w-full h-1.5 bg-zinc-800 rounded-full appearance-none accent-indigo-500"
                      />
                    </div>

                    <div className="space-y-4">
                      <div className="flex justify-between text-xs font-bold uppercase tracking-widest text-zinc-500">
                        <span>Rate</span>
                        <span className="text-white">{voiceSettings.rate.toFixed(1)}</span>
                      </div>
                      <input 
                        type="range" min="0.5" max="2" step="0.1" 
                        value={voiceSettings.rate}
                        onChange={(e) => setVoiceSettings(prev => ({ ...prev, rate: parseFloat(e.target.value) }))}
                        className="w-full h-1.5 bg-zinc-800 rounded-full appearance-none accent-indigo-500"
                      />
                    </div>

                    <button 
                      onClick={() => triggerVoiceAlert("Aegis voice synthesis test successful.")}
                      className="w-full py-3 bg-zinc-800 hover:bg-zinc-700 text-white rounded-xl text-sm font-bold transition-all flex items-center justify-center gap-2"
                    >
                      <Mic className="w-4 h-4" />
                      Test Voice Output
                    </button>
                  </div>
                </section>

                <section className="space-y-6">
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <Cpu className="w-5 h-5 text-indigo-500" />
                    Sensor Calibration
                  </h3>
                  <div className="bg-zinc-900/50 p-8 rounded-3xl border border-white/5 space-y-6">
                    <div className="grid grid-cols-2 gap-8">
                      <div className="space-y-4">
                        <div className="flex justify-between text-xs font-bold uppercase tracking-widest text-zinc-500">
                          <span>Focal Length (px)</span>
                          <span className="text-white">700</span>
                        </div>
                        <input type="range" min="400" max="1200" defaultValue="700" className="w-full h-1.5 bg-zinc-800 rounded-full appearance-none accent-indigo-500" />
                      </div>
                      <div className="space-y-4">
                        <div className="flex justify-between text-xs font-bold uppercase tracking-widest text-zinc-500">
                          <span>Confidence Threshold</span>
                          <span className="text-white">0.65</span>
                        </div>
                        <input type="range" min="0.1" max="0.9" step="0.05" defaultValue="0.65" className="w-full h-1.5 bg-zinc-800 rounded-full appearance-none accent-indigo-500" />
                      </div>
                    </div>
                    <button className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-sm font-bold transition-all">
                      Recalibrate Neural Engine
                    </button>
                  </div>
                </section>

                <section className="space-y-6">
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <Activity className="w-5 h-5 text-amber-500" />
                    System Logs
                  </h3>
                  <div className="bg-zinc-900/50 rounded-3xl border border-white/5 overflow-hidden">
                    <div className="p-4 bg-black/40 font-mono text-[10px] text-zinc-400 h-48 overflow-y-auto custom-scrollbar space-y-1">
                      <p className="text-emerald-500">[09:48:02] Neural Engine initialized successfully</p>
                      <p>[09:48:05] Camera stream connected: 1280x720@30fps</p>
                      <p>[09:48:10] COCO-SSD model weights loaded (24.2MB)</p>
                      <p className="text-amber-500">[09:48:15] Warning: High CPU load detected during inference</p>
                      <p>[09:48:20] Voice synthesis engine ready</p>
                      <p>[09:48:25] Recording session #REC-8294 started</p>
                      <p className="text-indigo-500">[09:48:30] Distance estimation calibrated for 700px focal length</p>
                      <p>[09:48:35] Steering guidance active: CENTER</p>
                    </div>
                  </div>
                </section>

                <section className="space-y-6">
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <Shield className="w-5 h-5 text-emerald-500" />
                    Security & Privacy
                  </h3>
                  <div className="bg-zinc-900/50 p-8 rounded-3xl border border-white/5 space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-zinc-300">End-to-end Encryption</span>
                      <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-widest">Active</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-zinc-300">Local Data Persistence</span>
                      <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-widest">Active</span>
                    </div>
                    <button className="w-full py-3 border border-rose-500/20 text-rose-500 hover:bg-rose-500/5 rounded-xl text-sm font-bold transition-all">
                      Wipe All Intelligence Logs
                    </button>
                  </div>
                </section>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>

      <style dangerouslySetInnerHTML={{ __html: `
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.1);
        }
        input[type="range"]::-webkit-slider-thumb {
          appearance: none;
          width: 16px;
          height: 16px;
          background: #6366f1;
          border-radius: 50%;
          cursor: pointer;
          border: 2px solid #fff;
        }
      `}} />
    </div>
  );
}
