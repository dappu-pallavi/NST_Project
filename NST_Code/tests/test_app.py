"""Automated tests for the Neural Style Transfer Studio.

Run from the project root (uses the real pretrained model, CPU is fine):
    .venv\\Scripts\\python.exe -m unittest discover -s NST_Code\\tests -v
"""
import io
import os
import shutil
import sys
import tempfile
import time
import unittest
import warnings
from unittest import mock

import numpy as np
import torch
from PIL import Image

NST_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NST_CODE_DIR)

# Send uploads/results to a throwaway folder; must be set before `config` is imported
RUNTIME_DIR = tempfile.mkdtemp(prefix='nst-test-')
os.environ['NST_RUNTIME_DIR'] = RUNTIME_DIR

import config  # noqa: E402
import app as nst_app  # noqa: E402
from utils.inference import style_transfer  # noqa: E402
from utils.utils import adaptive_instance_normalization, calc_mean_std  # noqa: E402

CONTENT_SAMPLE = os.path.join(NST_CODE_DIR, 'content_data', 'brad_pitt.jpg')
STYLE_SAMPLE = os.path.join(NST_CODE_DIR, 'style_data', 'sketch.png')


def tearDownModule():
    shutil.rmtree(RUNTIME_DIR, ignore_errors=True)


def image_bytes(size=(320, 240), color=(120, 90, 200), fmt='PNG'):
    buffer = io.BytesIO()
    Image.new('RGB', size, color).save(buffer, format=fmt)
    return buffer.getvalue()


def sample_bytes(path):
    with open(path, 'rb') as f:
        return f.read()


class AdaINTests(unittest.TestCase):
    """The core AdaIN maths from utils/utils.py."""

    def test_calc_mean_std_matches_torch(self):
        feat = torch.randn(2, 8, 16, 16)
        mean, std = calc_mean_std(feat)
        self.assertEqual(mean.shape, (2, 8, 1, 1))
        torch.testing.assert_close(mean.flatten(), feat.mean(dim=(2, 3)).flatten())
        torch.testing.assert_close(std.flatten(), (feat.var(dim=(2, 3), unbiased=False) + 1e-5).sqrt().flatten())

    def test_adain_gives_content_the_style_statistics(self):
        content = torch.randn(1, 4, 32, 32) * 3 + 1
        style = torch.randn(1, 4, 32, 32) * 0.5 - 2
        out = adaptive_instance_normalization(content, style)
        out_mean, out_std = calc_mean_std(out)
        style_mean, style_std = calc_mean_std(style)
        torch.testing.assert_close(out_mean, style_mean, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(out_std, style_std, atol=1e-3, rtol=1e-3)

    def test_adain_with_same_image_is_identity(self):
        feat = torch.randn(1, 4, 16, 16)
        torch.testing.assert_close(adaptive_instance_normalization(feat, feat), feat, atol=1e-4, rtol=1e-4)


class InferenceTests(unittest.TestCase):
    """The full encoder -> AdaIN -> decoder pipeline with the real weights."""

    @classmethod
    def setUpClass(cls):
        cls.content = Image.open(CONTENT_SAMPLE).convert('RGB')
        cls.style = Image.open(STYLE_SAMPLE).convert('RGB')

    def run_model(self, alpha):
        return style_transfer(self.content, self.style, nst_app.encoder, nst_app.decoder, alpha, nst_app.device)

    def test_output_is_a_valid_image_tensor(self):
        out = self.run_model(1.0)
        self.assertEqual(out.shape[:2], (1, 3))
        self.assertGreaterEqual(min(out.shape[-2:]), config.PROCESSING_SIZE)
        self.assertTrue(torch.isfinite(out).all())

    def test_inference_is_deterministic(self):
        torch.testing.assert_close(self.run_model(1.0), self.run_model(1.0))

    def test_style_strength_changes_the_output(self):
        full, none, half = self.run_model(1.0), self.run_model(0.0), self.run_model(0.5)
        self.assertGreater((full - none).abs().mean().item(), 0.01)
        # alpha=0.5 lies between the two extremes
        self.assertLess((half - none).abs().mean().item(), (full - none).abs().mean().item())


class ApiTests(unittest.TestCase):
    """The Flask routes, validation and file handling."""

    def setUp(self):
        nst_app.app.config['TESTING'] = True
        nst_app.app.config['WTF_CSRF_ENABLED'] = False
        self.client = nst_app.app.test_client()

    def post(self, content=None, style=None, alpha='1.0', **extra):
        data = {'alpha': alpha, **extra}
        if content is not None:
            data['content'] = (io.BytesIO(content[1]), content[0])
        if style is not None:
            data['style'] = (io.BytesIO(style[1]), style[0])
        return self.client.post('/api/stylize', data=data, content_type='multipart/form-data')

    def assert_error(self, response, status, text):
        self.assertEqual(response.status_code, status, response.get_data(as_text=True))
        self.assertIn(text, response.get_json()['error'])

    # --- successful requests -------------------------------------------------

    def test_index_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('Neural Style Transfer Studio', html)
        self.assertIn('name="csrf-token"', html)
        self.assertIn('data-name="sketch.png"', html)

    def test_stylize_uploaded_images_end_to_end(self):
        response = self.post(('photo.jpg', sample_bytes(CONTENT_SAMPLE)), ('art.png', sample_bytes(STYLE_SAMPLE)))
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()

        # The result file is a real image with the reported size
        result_path = os.path.join(config.RESULT_DIR, os.path.basename(payload['result_url']))
        with Image.open(result_path) as result:
            self.assertEqual(result.size, (payload['width'], payload['height']))
        self.assertEqual(payload['alpha'], 1.0)

        with self.client.get(payload['result_url']) as view:
            self.assertEqual(view.status_code, 200)
            self.assertEqual(view.mimetype, 'image/jpeg')

        with self.client.get(payload['download_url']) as download:
            self.assertEqual(download.status_code, 200)
            self.assertIn('attachment', download.headers['Content-Disposition'])
            self.assertIn('stylized_image.jpg', download.headers['Content-Disposition'])

    def test_result_matches_direct_model_output(self):
        """The web app returns the model's own output, not a stored sample."""
        content_png = image_bytes((256, 256), (200, 120, 40))
        response = self.post(('c.png', content_png), style_preset='la_muse.jpg')
        self.assertEqual(response.status_code, 200)
        result_path = os.path.join(config.RESULT_DIR, os.path.basename(response.get_json()['result_url']))

        expected = style_transfer(Image.open(io.BytesIO(content_png)).convert('RGB'),
                                  Image.open(os.path.join(NST_CODE_DIR, 'style_data', 'la_muse.jpg')).convert('RGB'),
                                  nst_app.encoder, nst_app.decoder, 1.0, nst_app.device)
        expected = (expected.squeeze(0).clamp(0, 1) * 255).round().byte().permute(1, 2, 0)
        with Image.open(result_path) as result:
            actual = torch.from_numpy(np.array(result))
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertLessEqual((actual.int() - expected.int()).abs().max().item(), 1)  # PNG is lossless

    def test_stylize_with_sample_presets(self):
        response = self.post(content_preset='chicago.jpg', style_preset='la_muse.jpg', alpha='0.5')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['alpha'], 0.5)

    def test_png_content_gives_png_result(self):
        response = self.post(('c.png', image_bytes()), style_preset='sketch.png')
        self.assertTrue(response.get_json()['result_url'].endswith('.png'))

    def test_grayscale_and_transparent_images_are_accepted(self):
        gray, rgba = io.BytesIO(), io.BytesIO()
        Image.new('L', (300, 300), 128).save(gray, 'PNG')
        Image.new('RGBA', (300, 300), (255, 0, 0, 128)).save(rgba, 'PNG')
        response = self.post(('g.png', gray.getvalue()), ('t.png', rgba.getvalue()))
        self.assertEqual(response.status_code, 200)

    def test_uploads_are_deleted_after_processing(self):
        self.post(('c.png', image_bytes()), ('s.png', image_bytes()))
        self.post(('bad.jpg', b'not an image'), ('s.png', image_bytes()))
        self.assertEqual(os.listdir(config.UPLOAD_DIR), [])

    def test_unique_result_names(self):
        first = self.post(content_preset='chicago.jpg', style_preset='sketch.png').get_json()
        second = self.post(content_preset='chicago.jpg', style_preset='sketch.png').get_json()
        self.assertNotEqual(first['result_url'], second['result_url'])

    # --- invalid input -------------------------------------------------------

    def test_missing_both_images(self):
        response = self.post()
        self.assert_error(response, 400, 'Please upload a content image')
        self.assertIn('Please upload a style image', response.get_json()['error'])

    def test_missing_style_image(self):
        self.assert_error(self.post(('c.png', image_bytes())), 400, 'Please upload a style image')

    def test_unsupported_file_type(self):
        self.assert_error(self.post(('notes.txt', b'hello'), ('s.png', image_bytes())), 400, 'unsupported file type ".txt"')

    def test_file_that_is_not_really_an_image(self):
        self.assert_error(self.post(('fake.jpg', b'not an image'), ('s.png', image_bytes())), 400, 'not a valid image')

    def test_truncated_jpeg(self):
        data = sample_bytes(CONTENT_SAMPLE)
        self.assert_error(self.post(('cut.jpg', data[:len(data) // 3]), ('s.png', image_bytes())), 400, 'not a valid image')

    def test_invalid_style_strength(self):
        for alpha in ('1.5', '-0.1', 'abc', '', 'nan'):
            with self.subTest(alpha=alpha):
                response = self.post(content_preset='chicago.jpg', style_preset='sketch.png', alpha=alpha)
                self.assert_error(response, 400, 'Style strength must be a number between 0.0 and 1.0')

    def test_too_many_pixels(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', Image.DecompressionBombWarning)  # expected: that is what we test
            response = self.post(('huge.png', image_bytes((8000, 6000))), ('s.png', image_bytes()))
        self.assert_error(response, 400, 'too large (8000x6000')

    def test_extreme_aspect_ratio(self):
        self.assert_error(self.post(('strip.png', image_bytes((3000, 100))), ('s.png', image_bytes())), 400, 'too long and thin')

    def test_upload_larger_than_limit(self):
        too_big = os.urandom(config.MAX_CONTENT_LENGTH + 1024)
        self.assert_error(self.post(('big.jpg', too_big), ('s.png', image_bytes())), 413, f'limit is {config.MAX_UPLOAD_MB} MB')

    def test_unknown_or_malicious_preset_names(self):
        for name in ('does_not_exist.jpg', '../app.py', '..\\config.py', '../vgg_normalised.pth'):
            with self.subTest(name=name):
                self.assert_error(self.post(content_preset=name, style_preset='sketch.png'), 400, 'Unknown sample content image')

    def test_csrf_token_is_required(self):
        nst_app.app.config['WTF_CSRF_ENABLED'] = True
        try:
            response = self.post(content_preset='chicago.jpg', style_preset='sketch.png')
            self.assert_error(response, 400, 'session has expired')
        finally:
            nst_app.app.config['WTF_CSRF_ENABLED'] = False

    def test_processing_failure_returns_friendly_error(self):
        with mock.patch.object(nst_app, 'style_transfer', side_effect=RuntimeError('boom')), \
                self.assertLogs(nst_app.app.logger, level='ERROR'):
            response = self.post(content_preset='chicago.jpg', style_preset='sketch.png')
        self.assert_error(response, 500, 'Style transfer failed')
        self.assertNotIn('boom', response.get_data(as_text=True))  # internals are not leaked

    # --- file serving --------------------------------------------------------

    def test_path_traversal_is_blocked(self):
        for url in ('/results/../app.py', '/download/..%2Fapp.py', '/presets/style/..%2F..%2Fapp.py', '/presets/other/x.jpg'):
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 404)

    def test_preset_thumbnail(self):
        with self.client.get('/presets/style/la_muse.jpg/thumb') as response:
            self.assertEqual(response.status_code, 200)
            with Image.open(io.BytesIO(response.data)) as thumb:
                self.assertLessEqual(max(thumb.size), config.THUMB_SIZE)

    def test_cleanup_removes_only_old_files(self):
        old_file = os.path.join(config.RESULT_DIR, 'stylized_old.jpg')
        new_file = os.path.join(config.RESULT_DIR, 'stylized_new.jpg')
        for path in (old_file, new_file):
            with open(path, 'wb') as f:
                f.write(b'x')
        two_hours_ago = time.time() - 2 * 3600
        os.utime(old_file, (two_hours_ago, two_hours_ago))

        nst_app.cleanup_old_files()
        self.assertFalse(os.path.exists(old_file))
        self.assertTrue(os.path.exists(new_file))
        os.remove(new_file)


if __name__ == '__main__':
    unittest.main()
