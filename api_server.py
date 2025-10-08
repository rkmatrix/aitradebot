from flask import Flask, jsonify
from flask_cors import CORS
import subprocess
import os
import signal # Required for process termination
import sys 

app = Flask(__name__)
CORS(app)

# Note: bot_process now stores a Popen object (the running subprocess)
bot_process = None
LOG_FILE = 'bot_log.txt'

@app.route('/', methods=['GET'])
def health_check():
    """Endpoint for Render's health check."""
    return jsonify({'status': 'ok', 'message': 'AITradePro API is healthy.'})

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Starts the trading bot as a persistent subprocess for ultimate stability."""
    global bot_process
    # Check if the process exists AND is still running (poll() returns None if running)
    if bot_process and bot_process.poll() is None:
        return jsonify({'status': 'error', 'message': 'Bot is already running.'}), 400

    print("API: Received request to start the trading bot...")
    
    try:
        log_file = open(LOG_FILE, 'w')
        
        # Determine OS-specific flags
        if os.name == 'nt': # Windows
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
            preexec_fn = None
        else: # Linux/Render
            creation_flags = 0
            preexec_fn = os.setsid 
            
        # FINAL DEFINITIVE FIX: Use the 'sys.executable' to explicitly launch Python,
        # ensuring the correct Python environment is used and bypassing shell issues.
        process = subprocess.Popen(
            [sys.executable, 'advanced_options_trader.py'],
            stdout=log_file, 
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
            preexec_fn=preexec_fn
        )
        bot_process = process
        print("API: Bot process started successfully.")
        return jsonify({'status': 'success', 'message': 'Bot started successfully.'})
    except Exception as e:
        print(f"FATAL: Failed to start bot subprocess: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to start bot: {e}'}), 500

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Stops the trading bot subprocess gracefully."""
    global bot_process
    if not bot_process or bot_process.poll() is not None:
        return jsonify({'status': 'error', 'message': 'Bot is not running.'}), 400

    print("API: Received request to stop the trading bot...")
    try:
        # Use SIGTERM (15) to signal the process group to stop gracefully
        if os.name == 'nt': # Windows
            os.kill(bot_process.pid, signal.SIGTERM)
        else: # Linux/Render
            os.killpg(os.getpgid(bot_process.pid), signal.SIGTERM)
            
        bot_process.wait(timeout=10)
    except Exception as e:
        print(f"Error stopping bot, forcing kill: {e}")
        bot_process.kill()
    
    bot_process = None
    print("API: Bot stopped successfully.")
    return jsonify({'status': 'success', 'message': 'Bot stopped successfully.'})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Gets the current status of the bot process."""
    # Check poll status: None means running, 0 or positive means terminated
    status = 'ACTIVE' if bot_process and bot_process.poll() is None else 'STOPPED'
    return jsonify({'status': status})

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Gets the last 100 lines from the bot's log file."""
    try:
        # File is created in the same directory as the script
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()[-100:]
        return jsonify({'logs': "".join(lines)})
    except Exception:
        return jsonify({'logs': 'Log file not yet available. Click Start.'})

@app.route('/api/run-setup', methods=['POST'])
def run_setup_endpoint():
    """Runs the monolithic setup script."""
    print("API: Received request to run monolithic setup...")
    try:
        result = subprocess.run(
            ['python', 'setup.py'],
            capture_output=True, text=True, check=True, timeout=900
        )
        return jsonify({'status': 'success', 'output': result.stdout})
    except subprocess.CalledProcessError as e:
        return jsonify({'status': 'error', 'output': e.stderr or e.stdout}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'output': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
