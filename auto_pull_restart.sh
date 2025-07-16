#!/bin/bash

# Script to automatically pull code from master branch and restart the Streamlit app if updates exist.
# Logs all steps, including errors, to auto_pull_restart.log.
# On every restart (successful or failed), appends recent Streamlit errors/logs for debugging.
# Run manually or via cron (e.g., crontab -e: */5 * * * * /home/ubuntu/rag_app/searchjudgements/auto_pull_restart.sh)
# Assumes repo dir, venv, etc., as before.

# Set variables
REPO_DIR="/home/ubuntu/rag_app/searchjudgements"
BRANCH="master"
VENV_PATH="$REPO_DIR/venv/bin/activate"
STREAMLIT_PORT=8501
LOG_FILE="$REPO_DIR/auto_pull_restart.log"
STREAMLIT_LOG="$REPO_DIR/streamlit.log"

# Function to log messages
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# Step 1: Navigate to repo
cd "$REPO_DIR" || { log "Error: Directory $REPO_DIR not found."; exit 1; }

# Step 2: Check for updates
log "Checking for updates on branch $BRANCH..."
git fetch origin
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse origin/$BRANCH 2>/dev/null)

if [ "$LOCAL" = "$REMOTE" ]; then
    log "No updates available. App not restarted."
    exit 0
fi

# Step 3: Pull changes
log "Updates found. Pulling changes..."
git checkout "$BRANCH" || { log "Error: Failed to checkout branch $BRANCH."; exit 1; }
git pull origin "$BRANCH" || { log "Error: Git pull failed."; exit 1; }

# Step 4: Kill existing processes
log "Killing existing Streamlit processes..."
pkill -f "streamlit run main.py" || true
sleep 2

# Step 5: Setup venv and dependencies
log "Setting up virtual environment and dependencies..."
if [ ! -d "$REPO_DIR/venv" ]; then
    python3 -m venv venv
fi
source "$VENV_PATH"
pip install --upgrade pip
pip install -r requirements.txt || { log "Error: Failed to install from requirements.txt."; exit 1; }
pip install streamlit || { log "Error: Failed to install Streamlit."; exit 1; }

# Step 6: Restart app
log "Restarting Streamlit app..."
nohup streamlit run main.py > "$STREAMLIT_LOG" 2>&1 &

# Step 7: Check status and log errors (always append recent Streamlit log for every restart)
sleep 5
if pgrep -f "streamlit run main.py" > /dev/null; then
    log "App restarted successfully."
else
    log "Error: App failed to start."
fi

# Always log recent Streamlit output/errors after restart attempt
log "Appending recent Streamlit logs for this restart:"
tail -n 20 "$STREAMLIT_LOG" >> "$LOG_FILE" 2>/dev/null || log "Warning: No Streamlit log file found."
log "Restart process completed."
