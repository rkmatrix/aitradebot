from flask import Flask, jsonify
from flask_cors import CORS
import subprocess
import os
import threading
import time
import config

# --- File Imports ---
# We import the functions directly to run them in a thread
import advanced_options_trader
import main as main_runner

# --- Configuration ---
app = Flask(__name__)
CORS(app)

# Use threading events to control the bot's lifecycle
bot_thread = None
stop_bot_event = threading.Event()
LOG_FILE = 'bot_log.txt'

# --- API Endpoints ---

@app.route('/', methods=['GET'])
def health_check():
    """Endpoint for Render's health check."""
    return jsonify({'status': 'ok', 'message': 'AITradePro API is healthy.'})

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Starts the trading bot in a separate, managed thread."""
    global bot_thread
    if bot_thread and bot_thread.is_alive():
        return jsonify({'status': 'error', 'message': 'Bot is already running.'}), 400

    print("API: Received request to start the trading bot...")
    stop_bot_event.clear() # Clear the stop signal
    
    # The target function for our thread
    def run_bot_in_thread():
        # Redirect output to log file
        with open(LOG_FILE, 'w') as log_f:
            # Temporarily redirect stdout and stderr
            original_stdout, original_stderr = os.sys.stdout, os.sys.stderr
            os.sys.stdout, os.sys.stderr = log_f, log_f
            try:
                # Pass the stop event to the trader function
                advanced_options_trader.run_trader(stop_event=stop_bot_event)
            finally:
                # Restore stdout and stderr
                os.sys.stdout, os.sys.stderr = original_stdout, original_stderr
        print("Bot thread has finished.")

    bot_thread = threading.Thread(target=run_bot_in_thread)
    bot_thread.start()
    print("API: Bot thread started.")
    return jsonify({'status': 'success', 'message': 'Bot started successfully.'})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Signals the trading bot thread to stop gracefully."""
    global bot_thread
    if not bot_thread or not bot_thread.is_alive():
        return jsonify({'status': 'error', 'message': 'Bot is not running.'}), 400

    print("API: Received request to stop the trading bot...")
    stop_bot_event.set() # Send the stop signal
    bot_thread.join(timeout=10) # Wait for the thread to finish
    
    if bot_thread.is_alive():
        print("API: Bot thread did not stop gracefully. It may take a moment to terminate.")
        return jsonify({'status': 'warning', 'message': 'Bot stop signal sent. May take a moment to terminate.'})
    
    bot_thread = None
    print("API: Bot stopped successfully.")
    return jsonify({'status': 'success', 'message': 'Bot stopped successfully.'})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Gets the current status of the bot thread."""
    status = 'ACTIVE' if bot_thread and bot_thread.is_alive() else 'STOPPED'
    return jsonify({'status': status})

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Gets the last 100 lines from the bot's log file."""
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()[-100:]
        return jsonify({'logs': "".join(lines)})
    except Exception:
        return jsonify({'logs': 'Log file not yet available.'})

@app.route('/api/run-setup', methods=['POST'])
def run_setup():
    """Runs the setup script in a subprocess as it's a one-off task."""
    print("API: Received request to run setup...")
    try:
        # Using subprocess here is fine as it's a finite task
        result = subprocess.run(
            ['python', 'main.py', '--action', 'setup'],
            capture_output=True, text=True, check=True, timeout=900 # 15 min timeout
        )
        return jsonify({'status': 'success', 'output': result.stdout})
    except subprocess.CalledProcessError as e:
        return jsonify({'status': 'error', 'output': e.stderr or e.stdout}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'output': str(e)}), 500

if __name__ == '__main__':
    # For production, Gunicorn runs the 'app' object.
    # The port is set by Render's environment variable.
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

