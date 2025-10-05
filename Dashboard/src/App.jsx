import React, { useState, useEffect, useCallback, useRef } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { LayoutDashboard, CandlestickChart, History, BrainCircuit, Settings as SettingsIcon, FileText, Play, Square, Loader, Server, AlertTriangle } from 'lucide-react';

// --- API Configuration ---
const API_BASE_URL = 'http://127.0.0.1:5001';

// --- UI Components ---
const Card = ({ children, className = '' }) => (
    <motion.div
        className={`bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-6 shadow-lg ${className}`}
        initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
    >
        {children}
    </motion.div>
);

const ToastNotification = ({ message, type, isVisible }) => (
    <AnimatePresence>
        {isVisible && (
            <motion.div
                className={`fixed bottom-5 right-5 p-4 rounded-lg shadow-2xl text-white z-50 border ${type === 'success' ? 'bg-emerald-500/80 border-emerald-400' : type === 'error' ? 'bg-red-500/80 border-red-400' : 'bg-yellow-500/80 border-yellow-400'}`}
                initial={{ opacity: 0, y: 50 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 50 }}
                transition={{ duration: 0.5 }}
            >
                {message}
            </motion.div>
        )}
    </AnimatePresence>
);

// --- Pages / Sections ---
const DashboardPage = () => (
    <div className="p-6">
        <h1 className="text-4xl font-bold text-white">Dashboard</h1>
        <p className="text-slate-400 mt-2">Live portfolio and bot performance metrics will be displayed here.</p>
        {/* Placeholder content */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 mt-6">
            <Card><h3 className="text-slate-400 text-sm font-medium">Total P/L (Simulated)</h3><p className="text-3xl font-bold mt-2 text-emerald-400">+$23,780.50</p></Card>
            <Card><h3 className="text-slate-400 text-sm font-medium">Win Rate (Simulated)</h3><p className="text-3xl font-bold mt-2 text-white">78%</p></Card>
        </div>
    </div>
);

const SettingsPage = ({ botStatus, logs, onStart, onStop, onSetup, showToast }) => {
    const [isLoadingSetup, setIsLoadingSetup] = useState(false);
    const [setupOutput, setSetupOutput] = useState('');
    const logsEndRef = useRef(null);

    useEffect(() => {
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs, setupOutput]);

    const handleRunSetup = async () => {
        setIsLoadingSetup(true);
        setSetupOutput('Running setup... This may take several minutes. Please be patient.\n\n');
        try {
            const res = await fetch(`${API_BASE_URL}/api/run-setup`, { method: 'POST' });
            const data = await res.json();
            setSetupOutput(prev => prev + data.output);
            if (res.ok) {
                showToast('Setup completed successfully!', 'success');
            } else {
                showToast('Setup failed. Check the output for errors.', 'error');
            }
        } catch (error) {
            const errorMsg = 'Error: Could not connect to the backend server. Is it running?';
            setSetupOutput(prev => prev + errorMsg);
            showToast(errorMsg, 'error');
        }
        setIsLoadingSetup(false);
    };

    const isBotActive = botStatus === 'ACTIVE';
    const isApiOffline = botStatus === 'OFFLINE';

    return (
        <div className="p-6 space-y-6">
            <h1 className="text-4xl font-bold text-white">Controls & Settings</h1>
            
            {isApiOffline && (
                <Card className="border-red-500/50 bg-red-500/10">
                    <div className="flex items-center">
                        <AlertTriangle className="w-8 h-8 text-red-400 mr-4"/>
                        <div>
                            <h3 className="text-red-400 text-lg font-semibold">Backend Offline</h3>
                            <p className="text-red-400/80">Could not connect to the Python API server. Please ensure `api_server.py` is running in a separate terminal.</p>
                        </div>
                    </div>
                </Card>
            )}

            <Card>
                <h3 className="text-white text-lg font-semibold">Bot Controls</h3>
                <div className="flex items-center justify-between mt-4">
                    <div className="flex items-center space-x-3">
                         <Server className={`w-6 h-6 ${isApiOffline ? 'text-slate-500' : 'text-slate-300'}`} />
                         <span className="text-slate-300">AI Trading Bot Status</span>
                    </div>
                    <div className="flex items-center space-x-4">
                        <span className={`font-bold px-3 py-1 rounded-full text-sm ${isBotActive ? 'bg-emerald-500/20 text-emerald-400' : isApiOffline ? 'bg-slate-700 text-slate-400' : 'bg-red-500/20 text-red-400'}`}>{botStatus}</span>
                        <button onClick={onStart} disabled={isBotActive || isLoadingSetup || isApiOffline} className="p-2 bg-emerald-600 rounded-full disabled:bg-slate-700 disabled:text-slate-500 hover:bg-emerald-500 transition-colors focus:outline-none focus:ring-2 focus:ring-emerald-400"><Play className="w-5 h-5"/></button>
                        <button onClick={onStop} disabled={!isBotActive || isLoadingSetup || isApiOffline} className="p-2 bg-red-600 rounded-full disabled:bg-slate-700 disabled:text-slate-500 hover:bg-red-500 transition-colors focus:outline-none focus:ring-2 focus:ring-red-400"><Square className="w-5 h-5"/></button>
                    </div>
                </div>
            </Card>
            <Card>
                <h3 className="text-white text-lg font-semibold">Run Full Setup</h3>
                <p className="text-slate-400 text-sm mt-1">Run data collection, feature engineering, and train the ultimate AI model. This must be done before starting the bot for the first time.</p>
                <button onClick={handleRunSetup} disabled={isLoadingSetup || isBotActive} className="mt-4 bg-blue-600 font-semibold px-4 py-2 rounded-lg hover:bg-blue-500 transition-colors disabled:bg-slate-700 disabled:cursor-not-allowed flex items-center">
                    {isLoadingSetup && <Loader className="animate-spin mr-2"/>}
                    {isLoadingSetup ? 'Running Setup...' : 'Run Full Setup'}
                </button>
                <AnimatePresence>
                {setupOutput && (
                    <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}>
                        <h4 className="text-white font-semibold mt-6 mb-2">Setup Output:</h4>
                        <div className="mt-2 p-4 bg-slate-900 rounded-lg h-64 overflow-y-auto font-mono text-sm text-slate-400 whitespace-pre-wrap">
                            {setupOutput}
                            <div ref={logsEndRef} />
                        </div>
                    </motion.div>
                )}
                </AnimatePresence>
            </Card>
             <Card>
                 <h3 className="text-white text-lg font-semibold">Live System Logs</h3>
                 <div className="mt-4 p-4 bg-slate-900 rounded-lg h-64 overflow-y-auto font-mono text-sm text-slate-400 whitespace-pre-wrap">
                    {logs || "No logs yet. Start the bot to see live output."}
                    <div ref={logsEndRef} />
                 </div>
            </Card>
        </div>
    );
};

const NavItem = ({ icon: Icon, text, active, onClick }) => ( <li className={`flex items-center p-3 my-1 rounded-lg cursor-pointer transition-colors ${active ? 'bg-emerald-500/20 text-emerald-400' : 'text-slate-400 hover:bg-white/10 hover:text-white'}`} onClick={onClick}> <Icon className="w-6 h-6 mr-4" /> <span className="font-semibold">{text}</span> </li>);

// --- MAIN APP COMPONENT --- //
const App = () => {
    const [activePage, setActivePage] = useState('Settings');
    const [toast, setToast] = useState({ isVisible: false, message: '', type: '' });
    const [botStatus, setBotStatus] = useState('OFFLINE');
    const [logs, setLogs] = useState('');

    const showToast = (message, type = 'success') => {
        setToast({ isVisible: true, message, type });
        setTimeout(() => setToast(prev => ({ ...prev, isVisible: false })), 4000);
    };
    
    const fetchStatusAndLogs = useCallback(async () => {
        try {
            const statusRes = await fetch(`${API_BASE_URL}/api/status`);
            if (!statusRes.ok) throw new Error('Status fetch failed');
            const statusData = await statusRes.json();
            setBotStatus(statusData.status);

            if (statusData.status === 'ACTIVE') {
                const logsRes = await fetch(`${API_BASE_URL}/api/logs`);
                if (!logsRes.ok) throw new Error('Logs fetch failed');
                const logsData = await logsRes.json();
                setLogs(logsData.logs);
            }
        } catch (error) {
            console.error("API connection failed:", error);
            setBotStatus('OFFLINE');
        }
    }, []);

    useEffect(() => {
        fetchStatusAndLogs();
        const interval = setInterval(fetchStatusAndLogs, 3000);
        return () => clearInterval(interval);
    }, [fetchStatusAndLogs]);

    const handleStartBot = async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/api/start`, { method: 'POST' });
            const data = await res.json();
            showToast(data.message, res.ok ? 'success' : 'error');
            fetchStatusAndLogs();
        } catch (error) { showToast('Error: Could not connect to the server.', 'error'); }
    };

    const handleStopBot = async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/api/stop`, { method: 'POST' });
            const data = await res.json();
            showToast(data.message, res.ok ? 'success' : 'error');
            fetchStatusAndLogs();
        } catch (error) { showToast('Error: Could not connect to the server.', 'error'); }
    };

    const handleRunSetup = async (setOutput) => {
        try {
            const res = await fetch(`${API_BASE_URL}/api/run-setup`, { method: 'POST' });
            const data = await res.json();
            setOutput(data.output);
            showToast(res.ok ? 'Setup completed successfully!' : 'Setup failed!', res.ok ? 'success' : 'error');
        } catch (error) {
            setOutput('Failed to connect to the server to run setup.');
            showToast('Error: Could not connect to the server.', 'error');
        }
    };

    const renderPage = () => {
        switch (activePage) {
            case 'Dashboard': return <DashboardPage />;
            case 'Settings': return <SettingsPage botStatus={botStatus} logs={logs} onStart={handleStartBot} onStop={handleStopBot} onSetup={handleRunSetup} showToast={showToast} />;
            default: return <SettingsPage botStatus={botStatus} logs={logs} onStart={handleStartBot} onStop={handleStopBot} onSetup={handleRunSetup} showToast={showToast} />;
        }
    };
    
    const navItems = [ { name: 'Dashboard', icon: LayoutDashboard }, { name: 'Settings', icon: SettingsIcon }];

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 to-black text-white font-sans">
            <div className="flex">
                 <aside className="w-64 min-h-screen p-4 bg-slate-900/50 border-r border-white/10 hidden md:block">
                     <div className="flex items-center mb-10"><BrainCircuit className="w-10 h-10 text-emerald-400" /><h1 className="text-2xl font-bold ml-2">AITradePro</h1></div>
                     <nav><ul>{navItems.map(item => (<NavItem key={item.name} icon={item.icon} text={item.name} active={activePage === item.name} onClick={() => setActivePage(item.name)} />))}</ul></nav>
                 </aside>
                <main className="flex-1">
                     <header className="flex items-center justify-between p-4 border-b border-white/10"><div></div><div className="flex items-center space-x-2"><img src="https://placehold.co/40x40/10b981/FFFFFF?text=A" alt="Avatar" className="rounded-full" /><span>Admin</span></div></header>
                    <div className="max-h-[calc(100vh-65px)] overflow-y-auto">{renderPage()}</div>
                </main>
            </div>
            <ToastNotification {...toast} />
        </div>
    );
};

export default App;

