from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import os
import atexit

# --- Configuration ---
app = Flask(__name__)
# This enables your React UI to communicate with this server
CORS(app)

# Use a dictionary to store the bot's process information
bot_process = {'process': None}
LOG_FILE = 'bot_log.txt'

# --- API Endpoints ---

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Starts the trading bot as a background process."""
    if bot_process.get('process') and bot_process['process'].poll() is None:
        return jsonify({'status': 'error', 'message': 'Bot is already running.'}), 400

    print("API: Received request to start the trading bot...")
    try:
        # We redirect the bot's output to a log file
        log_file = open(LOG_FILE, 'w')
        # Using '-u' for unbuffered python output is crucial for live logs
        process = subprocess.Popen(
            ['python', '-u', 'main.py', '--action', 'trade_options'],
            stdout=log_file, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        bot_process['process'] = process
        print(f"API: Bot started successfully with PID: {process.pid}")
        return jsonify({'status': 'success', 'message': 'Bot started successfully.'})
    except Exception as e:
        print(f"API: Error starting bot - {e}")
        return jsonify({'status': 'error', 'message': f'Failed to start bot: {e}'}), 500

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Stops the trading bot process."""
    process = bot_process.get('process')
    if not process or process.poll() is not None:
        return jsonify({'status': 'error', 'message': 'Bot is not running.'}), 400

    print(f"API: Received request to stop the trading bot (PID: {process.pid})...")
    try:
        # Terminate the process
        process.terminate()
        process.wait(timeout=5) # Wait for the process to terminate
        bot_process['process'] = None
        print("API: Bot stopped successfully.")
        return jsonify({'status': 'success', 'message': 'Bot stopped successfully.'})
    except subprocess.TimeoutExpired:
        print("API: Process did not terminate gracefully, killing.")
        process.kill()
        bot_process['process'] = None
        return jsonify({'status': 'warning', 'message': 'Bot did not stop gracefully, it was killed.'})
    except Exception as e:
        print(f"API: Error stopping bot - {e}")
        return jsonify({'status': 'error', 'message': f'Failed to stop bot: {e}'}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Gets the current status of the bot."""
    process = bot_process.get('process')
    if process and process.poll() is None:
        status = 'ACTIVE'
    else:
        status = 'STOPPED'
    return jsonify({'status': status})

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Gets the last 50 lines from the bot's log file."""
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            last_50_lines = lines[-50:]
        return jsonify({'logs': "".join(last_50_lines)})
    except FileNotFoundError:
        return jsonify({'logs': 'Log file not found. Start the bot to create it.'})
    except Exception as e:
        return jsonify({'logs': f'Error reading logs: {e}'})

@app.route('/api/run-setup', methods=['POST'])
def run_setup():
    """Runs the setup script and returns its output."""
    print("API: Received request to run setup...")
    try:
        # Use Popen to stream output if needed, but run is simpler for a complete process
        result = subprocess.run(
            ['python', 'main.py', '--action', 'setup'],
            capture_output=True, text=True, check=True, timeout=600 # 10 minute timeout
        )
        print("API: Setup completed successfully.")
        return jsonify({'status': 'success', 'output': result.stdout})
    except subprocess.CalledProcessError as e:
        print(f"API: Setup failed with an error - {e.stderr}")
        return jsonify({'status': 'error', 'output': e.stderr}), 500
    except Exception as e:
        print(f"API: An unexpected error occurred during setup - {e}")
        return jsonify({'status': 'error', 'output': str(e)}), 500

def cleanup_bot_process():
    """Ensure the bot process is terminated when the server shuts down."""
    process = bot_process.get('process')
    if process and process.poll() is None:
        print("API Server shutting down. Terminating bot process...")
        process.terminate()
        process.wait()

if __name__ == '__main__':
    # Register the cleanup function to be called on server exit
    atexit.register(cleanup_bot_process)
    # Use debug=False for production, but True is fine for local dev
    app.run(host='127.0.0.1', port=5001, debug=True)

