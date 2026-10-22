import os
import threading
import time
import uuid

import torch
from flask import Flask, abort, jsonify, render_template, request, send_from_directory, url_for
from flask_wtf.csrf import CSRFError, CSRFProtect
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

import config
from utils.inference import load_models, load_image, style_transfer, save_image


app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
# Every POST must carry the CSRF token rendered into the page
csrf = CSRFProtect(app)

# Images larger than this make Pillow raise DecompressionBombError instead of exhausting memory
Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS

for folder in (config.UPLOAD_DIR, config.RESULT_DIR, config.THUMB_DIR):
    os.makedirs(folder, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder, decoder = load_models(config.VGG_PATH, config.DECODER_PATH, device)

# One style transfer at a time: parallel requests would only compete for the same CPU/GPU
inference_lock = threading.Lock()


class InputError(Exception):
    """A problem with the user's input; the message is shown to the user as-is."""


def get_extension(filename):
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''


def list_presets(kind):
    folder = config.PRESET_DIRS[kind]
    return sorted(name for name in os.listdir(folder) if get_extension(name) in config.ALLOWED_EXTENSIONS)


def preset_label(name):
    return name.rsplit('.', 1)[0].replace('_', ' ').title()


def cleanup_old_files():
    """Delete uploads and results older than FILE_MAX_AGE."""
    cutoff = time.time() - config.FILE_MAX_AGE
    for folder in (config.UPLOAD_DIR, config.RESULT_DIR):
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            try:
                if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                    os.remove(path)
            except OSError:
                pass  # file may be in use (e.g. being downloaded); retry on the next cleanup


def remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


def validate_image_file(path, label):
    """Make sure the file is a real, readable image within the size limits."""
    try:
        with Image.open(path) as image:
            image.verify()  # checks file structure
        with Image.open(path) as image:
            image.load()    # fully decodes pixels; catches truncated JPEGs that verify() misses
            width, height = ImageOps.exif_transpose(image).size
    except Image.DecompressionBombError:
        raise InputError(f'{label} image is too large in resolution. '
                         f'Maximum is {config.MAX_IMAGE_PIXELS // 1_000_000} megapixels.')
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise InputError(f'{label} image is not a valid image file or is corrupted.')

    if width * height > config.MAX_IMAGE_PIXELS:
        raise InputError(f'{label} image is too large ({width}x{height} pixels). '
                         f'Maximum is {config.MAX_IMAGE_PIXELS // 1_000_000} megapixels.')
    if max(width, height) > config.MAX_ASPECT_RATIO * min(width, height):
        raise InputError(f'{label} image is too long and thin ({width}x{height} pixels). '
                         f'Please use an image whose sides differ by at most {config.MAX_ASPECT_RATIO}x.')


def resolve_input(kind, label, temp_files):
    """Return the path of the image to use for `kind` ('content' or 'style').

    The user either uploads a file (saved under a unique temporary name) or picks
    one of the bundled sample images by name.
    """
    upload = request.files.get(kind)
    if upload and upload.filename:
        extension = get_extension(upload.filename)
        if extension not in config.ALLOWED_EXTENSIONS:
            shown = f'".{extension}"' if extension else 'without an extension'
            raise InputError(f'{label} image: unsupported file type {shown}. Please upload a PNG or JPG image.')

        # Unique name, so different uploads never overwrite each other
        path = os.path.join(config.UPLOAD_DIR, f'{uuid.uuid4().hex}.{extension}')
        upload.save(path)
        temp_files.append(path)
        validate_image_file(path, label)
        return path

    preset = request.form.get(f'{kind}_preset', '')
    if preset:
        # Only names from the bundled folder are accepted (no paths, no ../ tricks)
        if preset not in list_presets(kind):
            raise InputError(f'Unknown sample {label.lower()} image. Please choose another one.')
        return os.path.join(config.PRESET_DIRS[kind], preset)

    raise InputError(f'Please upload a {label.lower()} image or choose a sample.')


def parse_alpha():
    message = f'Style strength must be a number between {config.ALPHA_MIN} and {config.ALPHA_MAX}.'
    try:
        alpha = float(request.form.get('alpha', '1.0'))
    except ValueError:
        raise InputError(message)
    # `not (a <= x <= b)` also rejects NaN
    if not (config.ALPHA_MIN <= alpha <= config.ALPHA_MAX):
        raise InputError(message)
    return alpha


def error_response(message, status):
    return jsonify(error=message), status


@app.route('/')
def index():
    presets = {kind: [{'name': name, 'label': preset_label(name)} for name in list_presets(kind)]
               for kind in config.PRESET_DIRS}
    return render_template('index.html', presets=presets, max_upload_mb=config.MAX_UPLOAD_MB,
                           processing_size=config.PROCESSING_SIZE, device=device.type.upper())


@app.route('/api/stylize', methods=['POST'])
def stylize():
    cleanup_old_files()
    temp_files = []
    try:
        errors = []
        paths = {}
        for kind, label in (('content', 'Content'), ('style', 'Style')):
            try:
                paths[kind] = resolve_input(kind, label, temp_files)
            except InputError as e:
                errors.append(str(e))
        try:
            alpha = parse_alpha()
        except InputError as e:
            errors.append(str(e))
        if errors:
            return error_response(' '.join(errors), 400)

        content_image = load_image(paths['content'])
        style_image = load_image(paths['style'])

        # Keep PNG for PNG inputs, JPEG otherwise
        extension = 'png' if get_extension(paths['content']) == 'png' else 'jpg'
        result_name = f'stylized_{uuid.uuid4().hex}.{extension}'

        with inference_lock:
            started = time.perf_counter()
            stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)
            elapsed = time.perf_counter() - started
        save_image(stylized_image, os.path.join(config.RESULT_DIR, result_name))

        height, width = stylized_image.shape[-2:]
        return jsonify(
            result_url=url_for('send_result', filename=result_name),
            download_url=url_for('download_result', filename=result_name),
            width=width, height=height, alpha=alpha,
            seconds=round(elapsed, 2), device=device.type.upper(),
        )
    except HTTPException:
        raise  # e.g. 413 Request Entity Too Large; handled by its own error handler below
    except torch.cuda.OutOfMemoryError:
        app.logger.exception('Out of GPU memory during style transfer')
        return error_response('The GPU ran out of memory. Please try smaller images.', 500)
    except Exception:
        app.logger.exception('Style transfer failed')
        return error_response('Style transfer failed while processing these images. '
                              'Please try different images.', 500)
    finally:
        # Uploaded inputs are only needed during the request
        for path in temp_files:
            remove_quietly(path)


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(e):
    return error_response(f'The uploaded files are too large. The total limit is {config.MAX_UPLOAD_MB} MB.', 413)


@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    return error_response('Your session has expired. Please reload the page and try again.', 400)


@app.route('/results/<filename>')
def send_result(filename):
    return send_from_directory(config.RESULT_DIR, filename)


@app.route('/download/<filename>')
def download_result(filename):
    return send_from_directory(config.RESULT_DIR, filename, as_attachment=True,
                               download_name=f'stylized_image.{get_extension(filename)}')


@app.route('/presets/<kind>/<name>')
def send_preset(kind, name):
    if kind not in config.PRESET_DIRS or name not in list_presets(kind):
        abort(404)
    return send_from_directory(config.PRESET_DIRS[kind], name)


@app.route('/presets/<kind>/<name>/thumb')
def send_preset_thumb(kind, name):
    """Small JPEG previews for the sample gallery, generated once and cached."""
    if kind not in config.PRESET_DIRS or name not in list_presets(kind):
        abort(404)
    thumb_name = f'{kind}_{name}.jpg'
    thumb_path = os.path.join(config.THUMB_DIR, thumb_name)
    if not os.path.isfile(thumb_path):
        image = load_image(os.path.join(config.PRESET_DIRS[kind], name))
        image.thumbnail((config.THUMB_SIZE, config.THUMB_SIZE))
        # Write to a unique temp file first so concurrent requests never see a half-written thumbnail
        temp_path = f'{thumb_path}.{uuid.uuid4().hex}.tmp'
        image.save(temp_path, format='JPEG', quality=85)
        os.replace(temp_path, thumb_path)
    return send_from_directory(config.THUMB_DIR, thumb_name)


cleanup_old_files()


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=config.DEBUG, use_debugger=config.DEBUG, threaded=True)
