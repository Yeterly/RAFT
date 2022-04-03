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
    base_output_path = "flow_outputs/"
    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        for participant in os.listdir(ps_path):
            # Create empyt numpy array for vstacking
            np_array = []
            for video in os.listdir(ps_path + "/" + participant):
                folder_path = ps_path + "/" + participant + "/" + video
                output_path = base_output_path + "/" + participant + "/"
                print(folder_path)
                images = glob.glob(os.path.join(folder_path, "*.png")) + glob.glob(
                    os.path.join(folder_path, "*.jpg")
                )

                images = sorted(images, key=natsort)
                for imfile1, imfile2 in zip(images[:-1], images[1:]):
                    image1 = load_image(imfile1)
                    image2 = load_image(imfile2)

                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)

                    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                    # vstack np_array with flow_up
                    flow_up = torch.squeeze(flow_up).cpu().numpy()
                    np_array.append(flow_up)

                    print(imfile1, imfile2, "is done")
                    # output_file = os.path.join("test", "frame.flo")
                    # print(flow.shape)
                    # writePFM("frame.pfm", flow)
                empty_array = np.empty(len(np_array), object)
                empty_array[:] = np_array
                # if output path not exist
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                # save flow_up to output path
                np.save(output_path + video + "_flow.npy", empty_array)


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
