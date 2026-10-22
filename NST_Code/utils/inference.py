import torch
from PIL import Image, ImageOps
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


def load_models(vgg_path, decoder_path, device):
    """Load the pretrained VGG encoder and the trained AdaIN decoder in eval mode."""
    encoder = VGGEncoder(vgg_path).to(device)
    decoder = Decoder().to(device)
    # decoder_final.pth was saved from a GPU; map_location lets it load on CPU-only machines
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    encoder.eval()
    decoder.eval()
    return encoder, decoder


def load_image(path):
    """Open an image as RGB, applying the EXIF rotation that phone cameras store."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)
