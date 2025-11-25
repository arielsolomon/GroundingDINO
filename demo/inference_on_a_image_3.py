import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    for box, label in zip(boxes, labels):
        box = box * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]

        color = tuple(np.random.randint(0, 255, size=3).tolist())
        x0, y0, x1, y1 = map(int, box)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, x0 + w, y0 + h)

        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None,
                         with_logits=True, cpu_only=False, token_spans=None):

    assert text_threshold is not None or token_spans is not None
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]

    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold

        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        pred_phrases = []

        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

    else:
        positive_maps = create_positive_map_from_span(
            model.tokenizer(caption),
            token_span=token_spans
        ).to(image.device)

        logits_for_phrases = positive_maps @ logits.T
        all_boxes = []
        all_phrases = []

        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            filt_mask = logit_phr > box_threshold

            all_boxes.append(boxes[filt_mask])
            if with_logits:
                logit_vals = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(v.item())[:4]})" for v in logit_vals])
            else:
                all_phrases.extend([phrase] * filt_mask.sum())

        boxes_filt = torch.cat(all_boxes).cpu()
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c",
                        default="groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--checkpoint_path", "-p",
                        default="weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--input_path", "-i",
                        default="inputs")   # <<<< CHANGED: directory of many images
    parser.add_argument("--text_prompt", "-t", default="Car")
    parser.add_argument("--output_dir", "-o", default="outputs")

    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--token_spans", type=str, default=None)
    parser.add_argument("--cpu-only", action="store_true")

    args = parser.parse_args()

    config_file = args.config_file
    checkpoint_path = args.checkpoint_path
    input_path = args.input_path
    text_prompt = args.text_prompt
    output_dir = args.output_dir

    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans

    os.makedirs(output_dir, exist_ok=True)

    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    # ------------------------------------------------------------
    # NEW: Process all images in folder
    # ------------------------------------------------------------
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    image_files = [f for f in os.listdir(input_path) if f.lower().endswith(exts)]

    print(f"Found {len(image_files)} images in folder: {input_path}")

    for file_name in image_files:

        img_path = os.path.join(input_path, file_name)
        print(f"Processing {img_path}...")

        image_pil, image = load_image(img_path)

        if token_spans is not None:
            text_threshold = None
            print("Using token_spans. Set the text_threshold to None.")

        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt,
            box_threshold, text_threshold,
            cpu_only=args.cpu_only,
            token_spans=eval(f"{token_spans}")
        )

        size = image_pil.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],
            "labels": pred_phrases,
            "scores": torch.ones(len(boxes_filt))
        }

        # output names indexed by original file name
        base_name = os.path.splitext(file_name)[0]

        pred_img_path = os.path.join(output_dir, f"{base_name}_pred.jpg")
        pred_txt_path = os.path.join(output_dir, f"{base_name}_pred.txt")

        image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
        image_with_box.save(pred_img_path)

        # ------------------------------------------------------------
        # YOLO export (unchanged except path uses base_name)
        # ------------------------------------------------------------
        with open(pred_txt_path, "w") as f:
            for box in boxes_filt:
                cx, cy, bw, bh = box.tolist()

                cx = float(max(0.0, min(1.0, cx)))
                cy = float(max(0.0, min(1.0, cy)))
                bw = float(max(1e-6, min(1.0, abs(bw))))
                bh = float(max(1e-6, min(1.0, abs(bh))))

                class_id = 0
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print("DONE.")
