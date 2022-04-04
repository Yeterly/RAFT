import sys

sys.path.append("core")

import argparse
import os
import cv2
import re
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from utils.frame_utils import writeFlow

DEVICE = "cuda"

natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split("(\d+)", s)]
# ['a1', 'a2', 'a3', 'a10', 'a11', 'a22']


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load("models/raft-things.pth"))
    ps_path = "F:\Batuhan\Yazılım\Freelance\Yeterly\KP_MLKit\main_data\\frame_outputs"
    base_output_path = "flow_outputs"
    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        for participant in os.listdir(ps_path):
            # TODO: count not store in them in one array because ram is not enough to hold every item
            for video in os.listdir(ps_path + "/" + participant):
                folder_path = ps_path + "/" + participant + "/" + video
                output_path = base_output_path + "/" + participant + "/"
                print(folder_path)
                images = glob.glob(os.path.join(folder_path, "*.png")) + glob.glob(
                    os.path.join(folder_path, "*.jpg")
                )

                images = sorted(images, key=natsort)
                frame_count = 0
                for imfile1, imfile2 in zip(images[:-1], images[1:]):
                    frame_count += 1
                    image1 = load_image(imfile1)
                    image2 = load_image(imfile2)

                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)

                    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                    # vstack np_array with flow_up
                    flow_up = torch.squeeze(flow_up).cpu().numpy()

                    final_output_path = (
                        output_path + video + "/frame_" + str(frame_count) + "_flow.npy"
                    )

                    if not os.path.exists(output_path + video + "/"):
                        os.makedirs(output_path + video + "/")

                    np.save(final_output_path, flow_up)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
)
parser.add_argument("--path", default="batu-frames")
parser.add_argument("--small", action="store_true")
parser.add_argument("--mixed_precision", action="store_true")
parser.add_argument(
    "--alternate_corr",
    action="store_true",
)
args = parser.parse_args()

demo(args)
