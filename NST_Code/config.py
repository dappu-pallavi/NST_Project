import os
import secrets

# All paths are resolved relative to this file, so the app runs from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pretrained model weights (shipped with the repository)
VGG_PATH = os.path.join(BASE_DIR, 'vgg_normalised.pth')
DECODER_PATH = os.path.join(BASE_DIR, 'experiment', 'final_exp', 'decoder_final.pth')

# Bundled sample images offered as one-click choices (read-only, tracked in Git)
PRESET_DIRS = {
    'content': os.path.join(BASE_DIR, 'content_data'),
    'style': os.path.join(BASE_DIR, 'style_data'),
}

# User uploads and generated results are kept apart from the bundled samples.
# This folder is created at startup and ignored by Git; tests point it elsewhere.
RUNTIME_DIR = os.environ.get('NST_RUNTIME_DIR') or os.path.join(BASE_DIR, 'runtime')
UPLOAD_DIR = os.path.join(RUNTIME_DIR, 'uploads')
RESULT_DIR = os.path.join(RUNTIME_DIR, 'results')
THUMB_DIR = os.path.join(RUNTIME_DIR, 'thumbs')
THUMB_SIZE = 240

# Files older than this are deleted automatically (seconds)
FILE_MAX_AGE = 60 * 60

# Input validation limits
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_UPLOAD_MB = 10
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024   # whole request (both images)
MAX_IMAGE_PIXELS = 40_000_000                      # ~40 megapixels; guards against decompression bombs
# The model resizes the *shorter* side to 512 px, so very long strips would become huge tensors
MAX_ASPECT_RATIO = 4
PROCESSING_SIZE = 512                              # matches transforms.Resize(512) in utils/inference.py
ALPHA_MIN, ALPHA_MAX = 0.0, 1.0

# Read the secret key from the environment; fall back to a random key per run
# (never hard-code secrets in source code)
SECRET_KEY = os.environ.get('NST_SECRET_KEY') or secrets.token_hex(32)

# Debug mode is off unless explicitly enabled, e.g. `set FLASK_DEBUG=1`
DEBUG = os.environ.get('FLASK_DEBUG', '0') == '1'
