import React, { useState, useEffect, useRef } from 'react';
import { 
  FileText, 
  Upload, 
  ArrowRight, 
  MessageSquare, 
  Settings, 
  Loader2, 
  BookOpen, 
  TrendingUp, 
  ShieldAlert,
  Sparkles,
  Trash2,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  CircleDollarSign,
  ArrowDown,
  Briefcase,
  Play,
  Search
} from 'lucide-react';

// Custom markdown styling helper for Emerald Forest theme
const parseMarkdown = (text) => {
  if (!text) return '';
  let html = text;

  // Escape HTML entities
  html = html
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Blockquotes
  html = html.replace(/^&gt;\s+(.*)$/gm, '<blockquote>$1</blockquote>');
  html = html.replace(/<\/blockquote>\n<blockquote>/g, '<br/>');

  // Tables
  const tableRegex = /\|([^|\n]+)\|([^|\n]+)\|\r?\n\|:?-+:?\|:?-+:?\|\r?\n((?:\|[^|\n]*\|[^|\n]*\|\r?\n?)*)/g;
  html = html.replace(tableRegex, (match, h1, h2, rows) => {
    let rowHtml = rows.split('\n').map(row => {
      if (!row.trim()) return '';
      let cols = row.split('|').slice(1, -1);
      if (cols.length < 2) return '';
      return `<tr>
        <td class="px-4 py-2.5 border-b border-forest-900/40 text-slate-300 font-medium">${cols[0].trim()}</td>
        <td class="px-4 py-2.5 border-b border-forest-900/40 text-emerald-400 font-semibold">${cols[1].trim()}</td>
      </tr>`;
    }).join('');
    
    return `<div class="overflow-x-auto my-4 rounded-xl border border-forest-800/40">
      <table class="min-w-full text-sm text-left bg-obsidian-900/60">
        <thead class="bg-forest-950/80 text-xs uppercase tracking-wider text-emerald-400 border-b border-forest-800/50">
          <tr>
            <th class="px-4 py-2.5">${h1.trim()}</th>
            <th class="px-4 py-2.5">${h2.trim()}</th>
          </tr>
        </thead>
        <tbody>
          ${rowHtml}
        </tbody>
      </table>
    </div>`;
  });

  // Headers
  html = html.replace(/^### (.*)$/gm, '<h3 class="text-base font-semibold text-emerald-300 mt-4 mb-2">$1</h3>');
  html = html.replace(/^## (.*)$/gm, '<h2 class="text-lg font-bold text-emerald-400 mt-5 mb-3 border-b border-forest-900/30 pb-1.5">$1</h2>');
  html = html.replace(/^# (.*)$/gm, '<h1 class="text-xl font-extrabold text-emerald-400 mt-6 mb-4">$1</h1>');

  // Bold
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong class="font-bold text-slate-100">$1</strong>');
  
  // Lists
  html = html.replace(/^\-\s+(.*)$/gm, '<li class="list-disc ml-5 my-1 text-slate-300">$1</li>');
  html = html.replace(/((?:<li class="list-disc ml-5 my-1 text-slate-300">.*<\/li>\n?)+)/g, '<ul class="my-2">$1</ul>');

  // Horizontal rules
  html = html.replace(/^---$/gm, '<hr class="my-4 border-forest-900/30" />');

  // Line breaks
  html = html.replace(/\n/g, '<br />');

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code class="bg-forest-950/50 px-1.5 py-0.5 rounded text-xs font-mono text-emerald-300 border border-forest-900/20">$1</code>');

  return html;
};
const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export default function App() {
  const [query, setQuery] = useState('');
  const [modelChoice, setModelChoice] = useState('groq');
  const [sessionId, setSessionId] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState(null);
  const [hasUploadedFile, setHasUploadedFile] = useState(false);
  const [uploadedFilename, setUploadedFilename] = useState('');
  const [messages, setMessages] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState({});
  const [floatingParticles, setFloatingParticles] = useState([]);
  const [isArchOpen, setIsArchOpen] = useState(false);

  const chatEndRef = useRef(null);
  const workspaceRef = useRef(null);

  // Generate floating stock metrics / background particles on mount
  useEffect(() => {
    // Ensure page starts at the top
    window.scrollTo(0, 0);

    // Generate session ID
    const id = 'sess_' + Math.random().toString(36).substring(2, 15);
    setSessionId(id);

    // Generate floating stock changes
    const items = [
      '+4.85%', '-2.15%', '+12.4%', '-0.92%', '+240 bps', '-150 bps', 
      '+1.8%', '-3.45%', '+$4.20', '-$1.50', '+8.2%', '-5.12%', 
      '+3.14%', '-0.75%', '+$12.80', '-$0.45', '+15.2%', '-1.18%'
    ];
    
    const particles = Array.from({ length: 22 }).map((_, i) => {
      const val = items[Math.floor(Math.random() * items.length)];
      const isPositive = val.startsWith('+');
      return {
        id: i,
        text: val,
        isPositive,
        left: `${Math.random() * 92}%`,
        delay: `${Math.random() * 6}s`,
        duration: `${14 + Math.random() * 10}s`,
        fontSize: `${11 + Math.random() * 5}px`
      };
    });
    setFloatingParticles(particles);
  }, []);

  useEffect(() => {
    if (messages.length > 0) {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isAnalyzing]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setUploadMessage(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setUploadMessage(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    if (sessionId) {
      formData.append('session_id', sessionId);
    }

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await response.json();
      setUploadMessage({ type: 'success', text: data.message });
      setHasUploadedFile(true);
      setUploadedFilename(selectedFile.name);
    } catch (error) {
      setUploadMessage({ type: 'error', text: error.message });
    } finally {
      setIsUploading(false);
    }
  };

  const clearUpload = () => {
    setSelectedFile(null);
    setUploadMessage(null);
    setHasUploadedFile(false);
    setUploadedFilename('');
  };

  const handleSearch = async (overrideQuery = null) => {
    const searchQuery = overrideQuery || query;
    if (!searchQuery.trim() || isAnalyzing) return;

    // Add user message
    const userMsg = { role: 'user', text: searchQuery };
    setMessages(prev => [...prev, userMsg]);
    if (!overrideQuery) setQuery('');
    setIsAnalyzing(true);

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          model_choice: modelChoice,
          session_id: hasUploadedFile ? sessionId : null
        })
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Analysis request failed');
      }

      const data = await response.json();

      const assistantMsg = {
        role: 'assistant',
        mode: data.mode,
        report: data.report,
        retrieved_chunks: data.retrieved_chunks || [],
        error: data.error
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        error: `Error performing analysis: ${error.message}` 
      }]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const scrollToWorkspace = () => {
    workspaceRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const toggleSources = (idx) => {
    setSourcesOpen(prev => ({
      ...prev,
      [idx]: !prev[idx]
    }));
  };

  const exampleQueries = [
    "What was management's tone on gross margins?",
    "Did the company beat earnings expectations?",
    "What risks did management highlight?",
    "What is guidance for next quarter?",
    "what is earnings call"
  ];

  return (
    <div className="min-h-screen bg-obsidian-950 text-slate-100 flex flex-col relative selection:bg-emerald-500/30 selection:text-emerald-200">
      
      {/* ── BACKGROUND FLOATING NUMBERS ANIMATION ────────────────────────────── */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none z-0">
        {floatingParticles.map(p => (
          <div
            key={p.id}
            className="floating-number"
            style={{
              left: p.left,
              animationDelay: p.delay,
              animationDuration: p.duration,
              fontSize: p.fontSize,
              color: p.isPositive ? '#10b981' : '#ef4444' // Emerald vs Red
            }}
          >
            {p.text}
          </div>
        ))}
        {/* Subtle grid background */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#111a2e_1px,transparent_1px),linear-gradient(to_bottom,#111a2e_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] opacity-[0.07]"></div>
      </div>

      {/* ── LANDING HERO PAGE (Top View) ────────────────────────────────────── */}
      <section className="min-h-screen flex flex-col justify-between items-center relative z-10 px-6 py-8">
        
        {/* Landing Navbar */}
        <header className="w-full max-w-6xl flex justify-between items-center flex-shrink-0">
          <div className="flex items-center gap-2.5">
            <div className="w-10 h-10 rounded-xl bg-forest-900 border border-forest-800 flex items-center justify-center text-emerald-400 shadow-md">
              <CircleDollarSign size={24} />
            </div>
            <div>
              <span className="font-extrabold text-base tracking-tight text-slate-100 block">
                EARNINGS CALL ANALYST
              </span>
              <span className="text-[10px] uppercase font-bold tracking-widest text-emerald-400 block -mt-1">
                decoupled agent v1.1
              </span>
            </div>
          </div>
        </header>

        {/* Landing Hero Central Card */}
        <div className="max-w-4xl text-center my-auto space-y-8 py-10 relative">
          
          {/* Badge */}
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold animate-pulse">
            <Sparkles size={12} />
            Emerald Forest Intelligent Workspace
          </div>

          {/* Heading */}
          <h2 className="text-4xl sm:text-5xl font-black tracking-tight leading-[1.1] text-slate-100 max-w-3xl mx-auto">
            Decipher Earnings Calls. <br />
            <span className="bg-gradient-to-r from-emerald-400 to-emerald-200 bg-clip-text text-transparent">
              Uncover Real Financial Signals.
            </span>
          </h2>

          {/* Subtitle / Description */}
          <p className="text-sm sm:text-base text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Ingest financial transcripts instantly. The intelligent LangGraph agent routes queries, performs hybrid vector retrieval via Qdrant, and extracts executive tone, key claims, and risks using custom finetuned models.
          </p>

          {/* THE NEAT LINE */}
          <div className="max-w-md mx-auto relative py-2">
            <div className="absolute inset-0 flex items-center" aria-hidden="true">
              <div className="w-full border-t border-forest-800/40"></div>
            </div>
            <div className="relative flex justify-center">
              <span className="bg-obsidian-950 px-3 text-xs text-forest-700 font-bold uppercase tracking-widest">
                RAG + Local LLM Pipelines
              </span>
            </div>
          </div>

          {/* Landing CTAs */}
          <div className="flex justify-center gap-4 pt-2">
            <button
              onClick={scrollToWorkspace}
              className="px-6 py-3.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 text-white font-bold text-sm tracking-wide transition-all duration-300 flex items-center gap-2 shadow-xl shadow-emerald-950/20 group"
            >
              <Play size={14} className="fill-white group-hover:scale-110 transition-transform" />
              Launch Workspace
            </button>
            <button
              onClick={() => setIsArchOpen(true)}
              className="px-6 py-3.5 rounded-xl bg-forest-900/30 hover:bg-forest-900/50 text-slate-300 font-bold text-sm transition-all duration-300 border border-forest-800/40 flex items-center justify-center cursor-pointer"
            >
              Explore Architecture
            </button>
          </div>

        </div>

        {/* Scroll CTA Indicator */}
        <button
          onClick={scrollToWorkspace}
          className="flex flex-col items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors animate-bounce mt-4 cursor-pointer"
        >
          <span>Scroll Down to Workspace</span>
          <ArrowDown size={14} className="text-emerald-400" />
        </button>

      </section>

      {/* ── ACTIVE PLATFORM WORKSPACE (Bottom View) ─────────────────────────── */}
      <section 
        ref={workspaceRef} 
        id="workspace" 
        className="min-h-screen border-t border-forest-900 bg-obsidian-900/60 backdrop-blur-lg relative z-10 flex flex-col"
      >
        
        {/* Workspace Sub Header */}
        <div className="h-16 border-b border-forest-900 flex items-center justify-between px-8 bg-forest-950/30 flex-shrink-0">
          <div className="flex items-center gap-2">
            <MessageSquare size={16} className="text-emerald-400" />
            <span className="text-sm font-semibold tracking-tight text-slate-200">Analysis Feed</span>
          </div>
          {hasUploadedFile && (
            <div className="bg-emerald-500/10 text-emerald-400 px-3 py-1 rounded-full text-xs font-semibold border border-emerald-500/20 flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping"></span>
              Restricted to: {uploadedFilename}
            </div>
          )}
        </div>

        {/* Unified Layout Dashboard */}
        <div className="flex-1 flex overflow-hidden h-[calc(100vh-4rem)]">
          
          {/* Left Panel: Control center */}
          <div className="w-80 border-r border-forest-900 bg-obsidian-950/50 flex flex-col p-6 space-y-6 overflow-y-auto flex-shrink-0">
            
            {/* Section: Model Configuration */}
            <div className="space-y-3">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                <Settings size={14} className="text-forest-700" />
                Model Selector
              </h3>
              <div className="bg-obsidian-900 rounded-xl p-1 border border-forest-900/50 flex">
                <button
                  onClick={() => setModelChoice('groq')}
                  className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all duration-300 ${
                    modelChoice === 'groq'
                      ? 'bg-forest-900 text-emerald-400 shadow border border-forest-800'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Groq Llama 3.1
                </button>
                <button
                  onClick={() => setModelChoice('mistral')}
                  className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all duration-300 ${
                    modelChoice === 'mistral'
                      ? 'bg-forest-900 text-emerald-400 shadow border border-forest-800'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Finetuned Mistral
                </button>
              </div>
              {modelChoice === 'mistral' && (
                <p className="text-[10px] text-emerald-400 bg-forest-950/40 border border-forest-900/30 p-2.5 rounded-lg leading-relaxed italic">
                  ⚡ Mistral is hosted on HF (supports specialized terminology). Cold start loading may take ~30s. Automatically falls back to Groq Llama if offline.
                </p>
              )}
            </div>

            {/* Section: Dropzone File Ingestion */}
            <div className="space-y-3">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                <Upload size={14} className="text-forest-700" />
                Upload Transcript
              </h3>
              
              <div className="bg-obsidian-900/60 rounded-xl border border-dashed border-forest-900 p-4 hover:border-emerald-500/30 transition-all duration-300">
                {!hasUploadedFile ? (
                  <div className="flex flex-col items-center justify-center text-center space-y-2.5">
                    <div className="w-9 h-9 rounded-full bg-forest-950 flex items-center justify-center text-emerald-400 border border-forest-900/30">
                      <FileText size={18} />
                    </div>
                    <div>
                      <label className="cursor-pointer text-xs font-semibold text-emerald-400 hover:text-emerald-300 block mb-1">
                        Select PDF or TXT File
                        <input 
                          type="file" 
                          accept=".txt,.pdf" 
                          onChange={handleFileChange} 
                          className="hidden" 
                        />
                      </label>
                      <p className="text-[10px] text-slate-500">Maximum size 10MB</p>
                    </div>

                    {selectedFile && (
                      <div className="w-full pt-2.5 border-t border-forest-900/30 mt-1">
                        <p className="text-[11px] font-medium text-slate-300 truncate max-w-[200px] mx-auto mb-2">
                          📄 {selectedFile.name}
                        </p>
                        <button
                          onClick={handleUpload}
                          disabled={isUploading}
                          className="w-full py-1.5 text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 text-white rounded-lg transition-colors flex items-center justify-center gap-1.5 disabled:opacity-50 shadow-md shadow-emerald-950/20"
                        >
                          {isUploading ? (
                            <>
                              <Loader2 size={13} className="animate-spin" />
                              Uploading...
                            </>
                          ) : (
                            <>
                              <Sparkles size={13} />
                              Ingest Transcript
                            </>
                          )}
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex items-start gap-3 bg-forest-950/30 border border-forest-900/40 p-3 rounded-lg">
                    <CheckCircle2 size={18} className="text-emerald-400 mt-0.5 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-semibold text-emerald-400 truncate">
                        Loaded Successfully
                      </p>
                      <p className="text-[10px] text-slate-400 truncate mt-0.5">
                        {uploadedFilename}
                      </p>
                      <button
                        onClick={clearUpload}
                        className="mt-2 text-[10px] font-semibold text-red-400 hover:text-red-300 flex items-center gap-1 transition-colors"
                      >
                        <Trash2 size={10} />
                        Remove from Session
                      </button>
                    </div>
                  </div>
                )}

                {uploadMessage && (
                  <div className={`mt-3 text-[11px] p-2 rounded-lg font-medium border ${
                    uploadMessage.type === 'success'
                      ? 'bg-emerald-950/20 text-emerald-400 border-emerald-900/30'
                      : 'bg-red-950/20 text-red-400 border-red-900/30'
                  }`}>
                    {uploadMessage.text}
                  </div>
                )}
              </div>

            </div>

            {/* Section: Example Queries */}
            <div className="space-y-2.5">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                <BookOpen size={14} className="text-forest-700" />
                Prompt Library
              </h3>
              <div className="flex flex-col gap-1.5">
                {exampleQueries.map((ex, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSearch(ex)}
                    className="w-full text-left text-[11px] font-medium text-slate-450 hover:text-emerald-300 bg-obsidian-900/50 hover:bg-forest-950/30 border border-forest-900/20 px-3 py-2 rounded-lg transition-all duration-300"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>

          </div>

          {/* Right Panel: Interactive chat interface */}
          <div className="flex-1 flex flex-col bg-obsidian-905 h-full relative overflow-hidden">
            
            {/* Messages Feed */}
            <div className="flex-1 overflow-y-auto px-8 py-6 space-y-6">
              {messages.length === 0 ? (
                
                // Active Workspace Landing View
                <div className="max-w-2xl mx-auto my-16 text-center space-y-4">
                  <div className="w-12 h-12 rounded-2xl bg-forest-950/50 border border-forest-900/40 text-emerald-400 flex items-center justify-center mx-auto shadow-md">
                    <MessageSquare size={24} />
                  </div>
                  <h3 className="text-lg font-bold text-slate-200 uppercase tracking-wider">
                     Active Analysis Workspace
                  </h3>
                  <p className="text-xs text-slate-400 max-w-md mx-auto leading-relaxed">
                    Select a model on the left, upload a document if desired, and submit your financial queries below to generate multi-dimensional analyst reports.
                  </p>
                </div>

              ) : (
                
                // Messages List
                <div className="max-w-3xl mx-auto space-y-6">
                  {messages.map((msg, idx) => (
                    <div key={idx} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                      
                      <div className={`max-w-2xl rounded-2xl p-5 border shadow-xl ${
                        msg.role === 'user'
                          ? 'bg-forest-950/20 border-forest-900/40 text-slate-200'
                          : 'bg-obsidian-900/70 border-forest-950 text-slate-350'
                      }`}>
                        
                        {msg.role === 'user' ? (
                          <div className="flex items-start gap-2.5">
                            <MessageSquare size={16} className="text-emerald-400 mt-1 flex-shrink-0" />
                            <p className="text-sm font-medium leading-relaxed">{msg.text}</p>
                          </div>
                        ) : (
                          <div className="space-y-4">
                            
                            {/* Mode header tag */}
                            {msg.mode && (
                              <div className="flex items-center justify-between pb-2 border-b border-forest-950">
                                <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-400 bg-forest-950/30 px-2 py-0.5 rounded border border-forest-900/40">
                                  Mode: {msg.mode.replace('_', ' ')}
                                </span>
                              </div>
                            )}

                            {/* Error panel */}
                            {msg.error && (
                              <div className="flex items-start gap-2 text-xs text-red-400 bg-red-950/15 border border-red-900/20 p-3 rounded-lg">
                                <ShieldAlert size={15} className="mt-0.5 flex-shrink-0" />
                                <p>{msg.error}</p>
                              </div>
                            )}

                            {/* Report output */}
                            {msg.report && (
                              <div 
                                className="text-sm text-slate-300 leading-relaxed space-y-3 prose prose-invert prose-emerald"
                                dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.report) }}
                              />
                            )}

                            {/* Referenced Sources list */}
                            {msg.retrieved_chunks && msg.retrieved_chunks.length > 0 && (
                              <div className="pt-3 border-t border-forest-950">
                                <button
                                  onClick={() => toggleSources(idx)}
                                  className="flex items-center gap-1 text-xs font-semibold text-emerald-400 hover:text-emerald-300 transition-colors"
                                >
                                  {sourcesOpen[idx] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                  {sourcesOpen[idx] ? 'Hide' : 'View'} {msg.retrieved_chunks.length} Sources Referenced
                                </button>

                                {sourcesOpen[idx] && (
                                  <div className="mt-3 space-y-3 pl-2 border-l-2 border-forest-900">
                                    {msg.retrieved_chunks.map((chunk, cIdx) => (
                                      <div key={cIdx} className="bg-obsidian-950/55 p-3 rounded-lg border border-forest-900/20 text-xs">
                                        <div className="flex items-center justify-between mb-1.5">
                                          <span className="font-semibold text-emerald-400 bg-forest-950/20 px-2 py-0.5 rounded border border-forest-900/30 truncate max-w-[200px]">
                                            {chunk.filename || chunk.ticker || 'UPLOAD'}
                                          </span>
                                          <span className="text-[10px] text-slate-500 font-semibold">
                                            Relevance: {(chunk.score * 100).toFixed(0)}%
                                          </span>
                                        </div>
                                        <p className="text-slate-400 italic leading-relaxed">
                                          "{chunk.text}"
                                        </p>
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            )}

                          </div>
                        )}

                      </div>
                    </div>
                  ))}

                  {isAnalyzing && (
                    <div className="flex items-center gap-2 bg-obsidian-900/70 border border-forest-950 p-4 rounded-xl max-w-sm mr-auto text-xs text-slate-450 animate-pulse">
                      <Loader2 size={14} className="animate-spin text-emerald-400" />
                      <span>Executing LangGraph analysis agent...</span>
                    </div>
                  )}
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Input Form */}
            <div className="p-6 border-t border-forest-900 bg-obsidian-950/30 flex-shrink-0">
              <div className="max-w-3xl mx-auto relative flex items-center">
                <Search size={18} className="absolute left-4 text-emerald-500/60 pointer-events-none" />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="Ask a question about financial quarters or uploaded earnings calls..."
                  disabled={isAnalyzing}
                  className="w-full bg-obsidian-900 border border-forest-900/50 focus:border-emerald-500/60 focus:ring-1 focus:ring-emerald-500/25 focus:shadow-[0_0_15px_rgba(16,185,129,0.15)] rounded-xl py-3.5 pl-12 pr-14 text-sm text-slate-200 placeholder-slate-500 outline-none transition-all duration-300 disabled:opacity-60"
                />
                <button
                  onClick={() => handleSearch()}
                  disabled={!query.trim() || isAnalyzing}
                  className="absolute right-2.5 p-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-obsidian-850 disabled:text-slate-600 text-white rounded-lg transition-all duration-300 flex items-center justify-center cursor-pointer font-bold shadow-md"
                >
                  <ArrowRight size={18} />
                </button>
              </div>
            </div>

          </div>

        </div>

      </section>

      {/* ── ARCHITECTURE MODAL ──────────────────────────────────────────────── */}
      {isArchOpen && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4 transition-all duration-300">
          <div className="bg-obsidian-900 border border-forest-900 rounded-2xl max-w-2xl w-full p-6 shadow-2xl space-y-5 animate-in fade-in zoom-in-95 duration-200">
            
            <div className="flex justify-between items-center border-b border-forest-950 pb-3">
              <div className="flex items-center gap-2">
                <Sparkles className="text-emerald-400" size={20} />
                <h3 className="text-lg font-bold text-slate-100">LangGraph Agent Architecture</h3>
              </div>
              <button 
                onClick={() => setIsArchOpen(false)}
                className="text-slate-400 hover:text-slate-200 text-xs font-semibold px-2.5 py-1 rounded bg-forest-950/40 border border-forest-900 hover:bg-forest-900 transition-colors cursor-pointer"
              >
                Close
              </button>
            </div>

            <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-2">
              <p className="text-xs text-slate-400 leading-relaxed">
                The platform is powered by an autonomous financial agent compiled using <strong>LangGraph</strong>. It dynamically classifies your query and decides on the most efficient computational execution path:
              </p>

              <div className="space-y-3">
                {/* Router */}
                <div className="bg-forest-950/20 border border-forest-900/30 p-3.5 rounded-xl space-y-1">
                  <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-widest bg-forest-900/40 px-2 py-0.5 rounded border border-forest-800/40">
                    Step 1: Classification Router (Llama 3.1)
                  </span>
                  <p className="text-xs text-slate-300 leading-relaxed pt-1">
                    Analyzes your question and decides the intent: <code>single_quarter</code> RAG analysis, <code>multi_quarter</code> comparative analysis, or <code>general</code> knowledge queries.
                  </p>
                </div>

                {/* General Chat */}
                <div className="bg-forest-950/20 border border-forest-900/30 p-3.5 rounded-xl space-y-1">
                  <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-widest bg-forest-900/40 px-2 py-0.5 rounded border border-forest-800/40">
                    Step 2a: General Q&amp;A (Bypass)
                  </span>
                  <p className="text-xs text-slate-350 leading-relaxed pt-1">
                    If the query is general (e.g. <em>"what is EPS?"</em>), the router routes it directly to the general chat handler. This completely bypasses document retrieval and analysis, generating an answer instantly.
                  </p>
                </div>

                {/* Hybrid Retrieval */}
                <div className="bg-forest-950/20 border border-forest-900/30 p-3.5 rounded-xl space-y-1">
                  <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-widest bg-forest-900/40 px-2 py-0.5 rounded border border-forest-800/40">
                    Step 2b: Semantic Chunk Retriever (Qdrant)
                  </span>
                  <p className="text-xs text-slate-350 leading-relaxed pt-1">
                    For quarter analysis queries, the retriever embeds your question with a local <code>SentenceTransformer</code> model and fetches the top semantic excerpts from Qdrant. If a file is uploaded in this session, the query is restricted exclusively to its chunks.
                  </p>
                </div>

                {/* LLM Analyzer */}
                <div className="bg-forest-950/20 border border-forest-900/30 p-3.5 rounded-xl space-y-1">
                  <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-widest bg-forest-900/40 px-2 py-0.5 rounded border border-forest-800/40">
                    Step 3: Financial Analyzer (Mistral 7B / Groq)
                  </span>
                  <p className="text-xs text-slate-350 leading-relaxed pt-1">
                    Processes the retrieved context against your query. If <em>Finetuned Mistral</em> is chosen, it runs the HuggingFace Inference API (optimized for corporate finance language) with automatic retry fallback to Groq.
                  </p>
                </div>

                {/* Reporter */}
                <div className="bg-forest-950/20 border border-forest-900/30 p-3.5 rounded-xl space-y-1">
                  <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-widest bg-forest-900/40 px-2 py-0.5 rounded border border-forest-800/40">
                    Step 4: Report Generator (Markdown)
                  </span>
                  <p className="text-xs text-slate-350 leading-relaxed pt-1">
                    Compiles the parsed structured dictionary into a final readable executive markdown report including Management Tone, Sentiment metrics, and citations mapping to original filenames.
                  </p>
                </div>
              </div>

            </div>

          </div>
        </div>
      )}

    </div>
  );
}
