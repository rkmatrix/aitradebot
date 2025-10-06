from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import os
import signal
import platform # New import to detect the operating system
import atexit

# --- Configuration ---
app = Flask(__name__)
CORS(app)

bot_process = {'process': None}
LOG_FILE = 'bot_log.txt'

# --- API Endpoints ---

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Starts the trading bot as a background process, handling OS differences."""
    if bot_process.get('process') and bot_process['process'].poll() is None:
        return jsonify({'status': 'error', 'message': 'Bot is already running.'}), 400

    print("API: Received request to start the trading bot...")
    try:
        log_file = open(LOG_FILE, 'w')
        
        # --- Cross-Platform Process Creation ---
        # This block handles the difference between Windows and Linux/MacOS
        if platform.system() == "Windows":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
            preexec_fn = None
        else: # For Linux (like Render) and MacOS
            creation_flags = 0
            preexec_fn = os.setsid

        process = subprocess.Popen(
            ['python', '-u', 'main.py', '--action', 'trade_options'],
            stdout=log_file, stderr=subprocess.STDOUT,
            creationflags=creation_flags,
            preexec_fn=preexec_fn
        )
        bot_process['process'] = process
        print(f"API: Bot started successfully with PID: {process.pid}")
        return jsonify({'status': 'success', 'message': 'Bot started successfully.'})
    except Exception as e:
        print(f"API: Error starting bot - {e}")
        return jsonify({'status': 'error', 'message': f'Failed to start bot: {e}'}), 500

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Stops the trading bot process, handling OS differences."""
    process = bot_process.get('process')
    if not process or process.poll() is not None:
        # If the process is not tracked or already stopped, just confirm stopped status
        bot_process['process'] = None
        return jsonify({'status': 'success', 'message': 'Bot is already stopped.'})

    print(f"API: Received request to stop the trading bot (PID: {process.pid})...")
    try:
        # --- Cross-Platform Process Termination ---
        if platform.system() == "Windows":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else: # For Linux (like Render) and MacOS
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        process.wait(timeout=5)
        print("API: Bot stopped successfully.")
    except Exception as e:
        print(f"API: Error during graceful stop - {e}, killing process.")
        process.kill()
    finally:
        bot_process['process'] = None
        
    return jsonify({'status': 'success', 'message': 'Bot stopped successfully.'})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Gets the current status of the bot."""
    process = bot_process.get('process')
    status = 'ACTIVE' if process and process.poll() is None else 'STOPPED'
    return jsonify({'status': status})

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Gets the last 50 lines from the bot's log file."""
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()[-50:]
        return jsonify({'logs': "".join(lines)})
    except Exception:
        return jsonify({'logs': 'Log file not yet available.'})

@app.route('/api/run-setup', methods=['POST'])
def run_setup():
    """Runs the setup script."""
    print("API: Received request to run setup...")
    try:
        result = subprocess.run(
            ['python', 'main.py', '--action', 'setup'],
            capture_output=True, text=True, check=True, timeout=600
        )
        return jsonify({'status': 'success', 'output': result.stdout})
    except subprocess.CalledProcessError as e:
        return jsonify({'status': 'error', 'output': e.stderr or e.stdout}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'output': str(e)}), 500

def cleanup_on_exit():
    """Ensures the bot process is terminated when the server shuts down."""
    print("API Server shutting down. Cleaning up bot process...")
    stop_bot()

if __name__ == '__main__':
    atexit.register(cleanup_on_exit)
    # The port is set by Render's environment variable, default to 10000 for local testing if needed
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)

