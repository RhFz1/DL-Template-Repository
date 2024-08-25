import logging
import logging.handlers
import os
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

# Create a rotating file handler
handler = logging.handlers.RotatingFileHandler(
    LOG_FILE_PATH, maxBytes=5*1024*1024, backupCount=3
)

# Define the log format with more details
log_format = ('[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] '
              '- %(message)s')

# Configure the logging
logging.basicConfig(
    level=logging.INFO,            # Set the logging level
    format=log_format,             # Use the detailed format defined above
    handlers=[handler]             # Use the rotating file handler
)