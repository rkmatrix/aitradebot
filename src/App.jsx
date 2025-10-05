import React, { useState, useEffect } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadialBarChart, RadialBar, Legend, ComposedChart, ReferenceDot } from 'recharts';
import { motion } from 'framer-motion';
import { LayoutDashboard, CandlestickChart, History, BrainCircuit, Settings as SettingsIcon, FileText, Sun, Moon, Bell, ArrowUp, ArrowDown, ChevronDown } from 'lucide-react';

// --- MOCK DATA & SIMULATION --- //
// In a real application, this data would come from your backend APIs.

// For Portfolio Growth Chart
const portfolioData = [
  { name: 'Jan', value: 100000 }, { name: 'Feb', value: 105000 },
  { name: 'Mar', value: 102000 }, { name: 'Apr', value: 115000 },
  { name: 'May', value: 122000 }, { name: 'Jun', value: 135000 },
  { name: 'Jul', value: 130000 }, { name: 'Aug', value: 142000 },
  { name: 'Sep', value: 155000 }, { name: 'Oct', value: 150000 },
];

// For Live Candlestick/Line Chart
const initialMarketData = [
  { time: '10:00', price: 170.12, signals: null }, { time: '10:05', price: 170.55, signals: null },
  { time: '10:10', price: 171.20, signals: { type: 'buy', price: 171.20, pl: 0 } }, { time: '10:15', price: 171.80, signals: null },
  { time: '10:20', price: 172.50, signals: null }, { time: '10:25', price: 171.90, signals: null },
  { time: '10:30', price: 170.80, signals: { type: 'sell', price: 170.80, pl: -0.40 } }, { time: '10:35', price: 171.15, signals: null },
  { time: '10:40', price: 172.00, signals: null }, { time: '10:45', price: 172.85, signals: { type: 'buy', price: 172.85, pl: 0 } },
  { time: '10:50', price: 173.50, signals: null }, { time: '10:55', price: 174.10, signals: null },
];

// For Trade History Table
const initialTradeHistory = [
    { id: 'T8462', symbol: 'NVDA', buyPrice: 125.50, sellPrice: 128.75, quantity: 10, profitLoss: 32.50, duration: '2h 15m', confidence: 0.92 },
    { id: 'T8461', symbol: 'AAPL', buyPrice: 171.20, sellPrice: 170.80, quantity: 5, profitLoss: -2.00, duration: '20m', confidence: 0.85 },
    { id: 'T8460', symbol: 'TSLA', buyPrice: 255.80, sellPrice: 262.10, quantity: 3, profitLoss: 18.90, duration: '1d 4h', confidence: 0.88 },
    { id: 'T8459', symbol: 'AMZN', buyPrice: 130.10, sellPrice: 129.90, quantity: 8, profitLoss: -1.60, duration: '45m', confidence: 0.76 },
    { id: 'T8458', symbol: 'GOOG', buyPrice: 135.40, sellPrice: 138.20, quantity: 5, profitLoss: 14.00, duration: '6h 30m', confidence: 0.95 },
];

// --- HELPER & UI COMPONENTS --- //

const Card = ({ children, className = '' }) => (
    <motion.div
        className={`bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-6 shadow-lg ${className}`}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        whileHover={{ y: -5, transition: { duration: 0.2 } }}
    >
        {children}
    </motion.div>
);

const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-slate-800/80 backdrop-blur-sm text-white p-3 rounded-lg border border-slate-700 shadow-xl">
                <p className="font-bold">{`Time: ${label}`}</p>
                <p className="text-emerald-400">{`Price: $${payload[0].value.toFixed(2)}`}</p>
                {payload[0].payload.signals && (
                    <div className="mt-2 pt-2 border-t border-slate-600">
                        <p className={`font-semibold ${payload[0].payload.signals.type === 'buy' ? 'text-green-400' : 'text-red-400'}`}>
                            {payload[0].payload.signals.type.toUpperCase()} Signal
                        </p>
                        <p>{`Trade Price: $${payload[0].payload.signals.price.toFixed(2)}`}</p>
                        <p>{`P/L: $${payload[0].payload.signals.pl.toFixed(2)}`}</p>
                    </div>
                )}
            </div>
        );
    }
    return null;
};

const ToastNotification = ({ message, type, isVisible }) => (
    <motion.div
        className={`fixed bottom-5 right-5 p-4 rounded-lg shadow-2xl text-white z-50 border ${type === 'success' ? 'bg-emerald-500/80 border-emerald-400' : 'bg-red-500/80 border-red-400'}`}
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: isVisible ? 1 : 0, y: isVisible ? 0 : 50 }}
        transition={{ duration: 0.5 }}
    >
        {message}
    </motion.div>
);


// --- MAIN SECTIONS / COMPONENTS --- //

const Dashboard = () => {
    const [stats, setStats] = useState({
        totalPL: 23780.50,
        winRate: 78,
        activeTrades: 4,
        avgReturn: 12.50,
    });

    useEffect(() => {
        const interval = setInterval(() => {
            setStats(prev => ({
                ...prev,
                totalPL: prev.totalPL + (Math.random() * 100 - 40),
            }));
            portfolioData.push({ name: 'Now', value: portfolioData[portfolioData.length - 1].value + (Math.random() * 2000 - 800) });
            if (portfolioData.length > 15) portfolioData.shift();
        }, 3000);
        return () => clearInterval(interval);
    }, []);
    
    const isProfit = stats.totalPL >= 0;

    return (
        <div className="p-6 space-y-6">
            <h1 className="text-4xl font-bold text-white">Dashboard</h1>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
                <Card>
                    <h3 className="text-slate-400 text-sm font-medium">Total Profit/Loss</h3>
                    <p className={`text-3xl font-bold mt-2 ${isProfit ? 'text-emerald-400' : 'text-red-400'}`}>
                        {isProfit ? '+' : ''}${stats.totalPL.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </p>
                </Card>
                <Card>
                    <h3 className="text-slate-400 text-sm font-medium">Win Rate</h3>
                    <p className="text-3xl font-bold mt-2 text-white">{stats.winRate}%</p>
                </Card>
                <Card>
                    <h3 className="text-slate-400 text-sm font-medium">Active Trades</h3>
                    <p className="text-3xl font-bold mt-2 text-white">{stats.activeTrades}</p>
                </Card>
                <Card>
                    <h3 className="text-slate-400 text-sm font-medium">Avg Return / Trade</h3>
                    <p className="text-3xl font-bold mt-2 text-emerald-400">${stats.avgReturn.toFixed(2)}</p>
                </Card>
            </div>
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                <Card className="xl:col-span-2">
                    <h3 className="text-white text-lg font-semibold">Portfolio Growth</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={portfolioData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                            <defs>
                                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                            <XAxis dataKey="name" stroke="#9ca3af" />
                            <YAxis stroke="#9ca3af" tickFormatter={(value) => `$${(value/1000)}k`} />
                            <Tooltip content={<CustomTooltip />} />
                            <Area type="monotone" dataKey="value" stroke="#10b981" fillOpacity={1} fill="url(#colorValue)" />
                        </AreaChart>
                    </ResponsiveContainer>
                </Card>
                <Card>
                    <h3 className="text-white text-lg font-semibold">AI Decision Confidence</h3>
                    <ResponsiveContainer width="100%" height={300}>
                         <RadialBarChart innerRadius="20%" outerRadius="100%" barSize={20} data={[{ name: 'Confidence', value: 88, fill: '#34d399' }]} startAngle={90} endAngle={-270}>
                            <RadialBar background clockWise dataKey="value" cornerRadius={10} />
                             <Legend iconSize={10} layout="vertical" verticalAlign="middle" wrapperStyle={{ top: '45%', left: '35%', lineHeight: '24px', transform: 'translate(0, -50%)' }} 
                                formatter={(value, entry) => <span className="text-white text-4xl font-bold">{entry.payload.value}%</span>}
                             />
                             <Tooltip />
                        </RadialBarChart>
                    </ResponsiveContainer>
                </Card>
            </div>
        </div>
    );
};

const LiveTrades = () => {
    const [marketData, setMarketData] = useState(initialMarketData);

    useEffect(() => {
        const interval = setInterval(() => {
            setMarketData(prevData => {
                const lastPoint = prevData[prevData.length - 1];
                const newPrice = lastPoint.price + (Math.random() - 0.5) * 1.5;
                const newTime = new Date(Date.now()).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                const newData = [...prevData, { time: newTime, price: newPrice, signals: null }];
                if (newData.length > 20) newData.shift();
                return newData;
            });
        }, 2000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="p-6">
            <h1 className="text-4xl font-bold text-white mb-6">Live Trades: AAPL/USD</h1>
            <Card>
                <ResponsiveContainer width="100%" height={500}>
                    <ComposedChart data={marketData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="time" stroke="#9ca3af" />
                        <YAxis stroke="#9ca3af" domain={['dataMin - 2', 'dataMax + 2']} tickFormatter={(value) => `$${value.toFixed(2)}`} />
                        <Tooltip content={<CustomTooltip />} />
                        <defs>
                             <linearGradient id="lineGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#34d399" stopOpacity={0.4}/>
                                <stop offset="95%" stopColor="#059669" stopOpacity={0.1}/>
                            </linearGradient>
                        </defs>
                        <Area type="monotone" dataKey="price" stroke="#34d399" strokeWidth={2} fill="url(#lineGradient)" />

                        {marketData.map((entry, index) => {
                            if (entry.signals?.type === 'buy') {
                                return <ReferenceDot key={index} x={entry.time} y={entry.price} r={8} fill="#22c55e" stroke="white" strokeWidth={2} />;
                            }
                            if (entry.signals?.type === 'sell') {
                                return <ReferenceDot key={index} x={entry.time} y={entry.price} r={8} fill="#ef4444" stroke="white" strokeWidth={2} />;
                            }
                            return null;
                        })}
                    </ComposedChart>
                </ResponsiveContainer>
            </Card>
        </div>
    );
};

const TradeHistory = () => {
    const [trades, setTrades] = useState(initialTradeHistory);
    // Add state for sorting, pagination etc. here
    return (
        <div className="p-6">
             <h1 className="text-4xl font-bold text-white mb-6">Trade History</h1>
             <Card>
                 <div className="overflow-x-auto">
                     <table className="w-full text-left text-slate-300">
                         <thead className="text-xs text-slate-400 uppercase bg-white/5">
                             <tr>
                                 <th scope="col" className="px-6 py-3">Trade ID</th>
                                 <th scope="col" className="px-6 py-3">Symbol</th>
                                 <th scope="col" className="px-6 py-3">Buy Price</th>
                                 <th scope="col" className="px-6 py-3">Sell Price</th>
                                 <th scope="col" className="px-6 py-3">P/L</th>
                                 <th scope="col" className="px-6 py-3">Confidence</th>
                             </tr>
                         </thead>
                         <tbody>
                             {trades.map((trade) => (
                                 <tr key={trade.id} className="border-b border-slate-800 hover:bg-white/5">
                                     <td className="px-6 py-4 font-mono">{trade.id}</td>
                                     <td className="px-6 py-4 font-bold text-white">{trade.symbol}</td>
                                     <td className="px-6 py-4">${trade.buyPrice.toFixed(2)}</td>
                                     <td className="px-6 py-4">${trade.sellPrice.toFixed(2)}</td>
                                     <td className={`px-6 py-4 font-semibold ${trade.profitLoss >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                         ${trade.profitLoss.toFixed(2)}
                                     </td>
                                     <td className="px-6 py-4">{(trade.confidence * 100).toFixed(0)}%</td>
                                 </tr>
                             ))}
                         </tbody>
                     </table>
                 </div>
             </Card>
        </div>
    );
}

const AIInsights = () => (
    <div className="p-6 space-y-6">
        <h1 className="text-4xl font-bold text-white">AI Insights</h1>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
                <h3 className="text-white text-lg font-semibold">Current AI Sentiment</h3>
                <p className="text-5xl font-bold text-emerald-400 mt-4">BULLISH</p>
                <p className="text-slate-400 mt-2">The model detects strong upward momentum based on technicals and recent news.</p>
            </Card>
            <Card>
                <h3 className="text-white text-lg font-semibold">24h Market Prediction</h3>
                 <div className="flex items-center mt-4">
                    <ArrowUp className="w-16 h-16 text-emerald-400" />
                    <div>
                        <p className="text-3xl font-bold text-white">SPY: +1.25%</p>
                        <p className="text-slate-400">High confidence of broad market gains.</p>
                    </div>
                 </div>
            </Card>
        </div>
        <Card>
             <h3 className="text-white text-lg font-semibold mb-4">Key Influencing Factors</h3>
             <ul className="space-y-3 text-slate-300">
                <li className="flex items-start"><ArrowRight className="w-5 h-5 text-emerald-400 mr-3 mt-1 flex-shrink-0" /> <span>Positive earnings report from a major tech competitor boosts sector confidence.</span></li>
                <li className="flex items-start"><ArrowRight className="w-5 h-5 text-emerald-400 mr-3 mt-1 flex-shrink-0" /> <span>MA(50) just crossed above MA(200), a classic golden cross signal.</span></li>
                <li className="flex items-start"><ArrowRight className="w-5 h-5 text-red-400 mr-3 mt-1 flex-shrink-0" /> <span>Consumer spending index came in slightly lower than expected, a minor bearish indicator.</span></li>
             </ul>
        </Card>
    </div>
);

const Settings = () => {
    const [botStatus, setBotStatus] = useState(true);
    const [tradingMode, setTradingMode] = useState('moderate');
    const [riskLevel, setRiskLevel] = useState(2);

    return (
        <div className="p-6 space-y-6">
            <h1 className="text-4xl font-bold text-white">Controls & Settings</h1>
            <Card>
                <h3 className="text-white text-lg font-semibold">Bot Controls</h3>
                <div className="flex items-center justify-between mt-4">
                    <span className="text-slate-300">AI Trading Bot Status</span>
                    <div className="flex items-center space-x-4">
                         <span className={`font-bold ${botStatus ? 'text-emerald-400' : 'text-red-400'}`}>{botStatus ? 'ACTIVE' : 'STOPPED'}</span>
                        <button onClick={() => setBotStatus(!botStatus)} className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${botStatus ? 'bg-emerald-500' : 'bg-slate-600'}`}>
                            <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${botStatus ? 'translate-x-6' : 'translate-x-1'}`} />
                        </button>
                    </div>
                </div>
            </Card>
            <Card>
                <h3 className="text-white text-lg font-semibold">Trading Mode</h3>
                <div className="mt-4 grid grid-cols-3 gap-4">
                    {['Aggressive', 'Moderate', 'Conservative'].map(mode => (
                         <button key={mode} onClick={() => setTradingMode(mode.toLowerCase())} className={`p-3 rounded-lg text-center font-semibold transition-all ${tradingMode === mode.toLowerCase() ? 'bg-emerald-500 text-white' : 'bg-slate-800 hover:bg-slate-700'}`}>
                             {mode}
                         </button>
                    ))}
                </div>
            </Card>
             <Card>
                <h3 className="text-white text-lg font-semibold">Risk Level ({riskLevel}%)</h3>
                <input type="range" min="0.5" max="5" step="0.5" value={riskLevel} onChange={(e) => setRiskLevel(e.target.value)} className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer mt-4" />
            </Card>
             <Card>
                 <h3 className="text-white text-lg font-semibold">System Logs</h3>
                 <div className="mt-4 p-4 bg-slate-900 rounded-lg h-48 overflow-y-auto font-mono text-sm text-slate-400">
                     <p>[11:56:02] AI model predicted UP for NVDA with 92% confidence.</p>
                     <p>[11:56:03] Searching for suitable CALL option for NVDA...</p>
                     <p className="text-emerald-400">[11:56:05] Placed BUY order for 1 contract of NVDA251220C00130000.</p>
                     <p>[11:55:10] AI signal for AAPL flipped to DOWN. Closing position.</p>
                     <p className="text-yellow-400">[11:55:11] API connection latency at 120ms.</p>
                 </div>
            </Card>
        </div>
    );
};

const Analytics = () => {
     const exportToCSV = () => {
        const headers = "Trade ID,Symbol,Buy Price,Sell Price,P/L\n";
        const rows = initialTradeHistory.map(t => `${t.id},${t.symbol},${t.buyPrice},${t.sellPrice},${t.profitLoss}`).join('\n');
        const csvContent = "data:text/csv;charset=utf-8," + headers + rows;
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "trade_history.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };
    
    return (
        <div className="p-6 space-y-6">
            <h1 className="text-4xl font-bold text-white">Analytics & Reports</h1>
            <div className="flex justify-end">
                <button onClick={exportToCSV} className="bg-emerald-500 text-white font-bold py-2 px-4 rounded-lg hover:bg-emerald-600 transition-colors">Export to CSV</button>
            </div>
            {/* Add more analytics components here */}
        </div>
    )
}

const NavItem = ({ icon: Icon, text, active, onClick }) => (
    <li className={`flex items-center p-3 my-1 rounded-lg cursor-pointer transition-colors ${active ? 'bg-emerald-500/20 text-emerald-400' : 'text-slate-400 hover:bg-white/10 hover:text-white'}`} onClick={onClick}>
        <Icon className="w-6 h-6 mr-4" />
        <span className="font-semibold">{text}</span>
    </li>
);

// --- MAIN APP COMPONENT --- //
const App = () => {
    const [activePage, setActivePage] = useState('Dashboard');
    const [isDarkMode, setIsDarkMode] = useState(true);
    const [toast, setToast] = useState({ isVisible: false, message: '', type: '' });

    useEffect(() => {
        const html = document.querySelector('html');
        if (isDarkMode) html.classList.add('dark');
        else html.classList.remove('dark');
    }, [isDarkMode]);

    useEffect(() => {
         const interval = setInterval(() => {
            setToast({ isVisible: true, message: 'New AI Trade: BUY NVDA Call', type: 'success' });
            setTimeout(() => setToast(prev => ({ ...prev, isVisible: false })), 4000);
        }, 30000); // New trade notification every 30 seconds
        return () => clearInterval(interval);
    }, []);

    const renderPage = () => {
        switch (activePage) {
            case 'Dashboard': return <Dashboard />;
            case 'Live Trades': return <LiveTrades />;
            case 'Trade History': return <TradeHistory />;
            case 'AI Insights': return <AIInsights />;
            case 'Settings': return <Settings />;
            case 'Analytics': return <Analytics />;
            default: return <Dashboard />;
        }
    };
    
    const navItems = [
        { name: 'Dashboard', icon: LayoutDashboard },
        { name: 'Live Trades', icon: CandlestickChart },
        { name: 'Trade History', icon: History },
        { name: 'AI Insights', icon: BrainCircuit },
        { name: 'Analytics', icon: FileText },
        { name: 'Settings', icon: SettingsIcon },
    ];

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 to-black text-white font-sans">
            <div className="flex">
                {/* Sidebar */}
                <aside className="w-64 min-h-screen p-4 bg-slate-900/50 border-r border-white/10 hidden md:block">
                    <div className="flex items-center mb-10">
                         <BrainCircuit className="w-10 h-10 text-emerald-400" />
                         <h1 className="text-2xl font-bold ml-2">AITradePro</h1>
                    </div>
                    <nav>
                        <ul>
                            {navItems.map(item => (
                                <NavItem key={item.name} icon={item.icon} text={item.name} active={activePage === item.name} onClick={() => setActivePage(item.name)} />
                            ))}
                        </ul>
                    </nav>
                </aside>
                
                {/* Main Content */}
                <main className="flex-1">
                    {/* Top Bar */}
                    <header className="flex items-center justify-between p-4 border-b border-white/10">
                        <div className="flex items-center">
                            {/* Search bar can go here */}
                        </div>
                        <div className="flex items-center space-x-6">
                            <button onClick={() => setIsDarkMode(!isDarkMode)}>
                                {isDarkMode ? <Sun className="text-slate-400 hover:text-white"/> : <Moon className="text-slate-400 hover:text-white"/>}
                            </button>
                            <Bell className="text-slate-400 hover:text-white cursor-pointer"/>
                            <div className="flex items-center space-x-2">
                                <img src="https://placehold.co/40x40/10b981/FFFFFF?text=A" alt="Avatar" className="rounded-full" />
                                <span className="font-semibold hidden sm:inline">Admin</span>
                                <ChevronDown className="w-4 h-4 text-slate-400" />
                            </div>
                        </div>
                    </header>
                    
                    {/* Page Content */}
                    <div className="max-h-[calc(100vh-65px)] overflow-y-auto">
                        {renderPage()}
                    </div>
                </main>
            </div>
            <ToastNotification {...toast} />
        </div>
    );
};

export default App;
