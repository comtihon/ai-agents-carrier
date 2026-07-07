import logging
import sys

# stdout, not the stderr default: GKE/Cloud Logging classifies every stderr
# line as severity ERROR, which drowned real errors in INFO noise.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

from app.api.app import create_app

app = create_app()
