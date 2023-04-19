import os

import megengine as mge
import megengine.functional as F
import argparse
import numpy as np
import cv2

from src.lightning.lightning_ltaw import PL_LTAW


def load_model(model_path):
    print("Loading model:", os.path.abspath(model_path))
    pretrained_dict = mge.load(model_path)
    model = PL_LTAW(max_disp=256, mixed_precision=False, test_mode=True)

    model.load_state_dict(pretrained_dict["state_dict"], strict=True)

    model.eval()
    return model


def inference(left, right, model, n_iter=20):
    print("Model Forwarding...")
    imgL = left.transpose(2, 0, 1)
    imgR = right.transpose(2, 0, 1)
    imgL = np.ascontiguousarray(imgL[None, :, :, :])
    imgR = np.ascontiguousarray(imgR[None, :, :, :])

    imgL = mge.tensor(imgL).astype("float32")
    imgR = mge.tensor(imgR).astype("float32")

    imgL_dw2 = F.nn.interpolate(
        imgL,
        size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    imgR_dw2 = F.nn.interpolate(
        imgR,
        size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)

    pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
    pred_disp = F.squeeze(pred_flow[:, 0, :, :]).numpy()

    return pred_disp


if __name__ == "__main__":
    img_path1 = "img/test/L/left_3.jpg"  # img/test/L/left_0.jpg
    img_path2 = "img/test/R/right_3.jpg"


    parser = argparse.ArgumentParser(description="A demo to run LTAW.")
    parser.add_argument(
        "--model_path",
        default="ltaw_eth3d.mge",
        help="The path of pre-trained MegEngine model.",
    )
    parser.add_argument(
        "--left", default=img_path1, help="The path of left image."      # "img/test/left.png"
    )
    parser.add_argument(
        "--right", default=img_path2, help="The path of right image."
    )
    parser.add_argument(
        "--size",
        default="1024x1536",
        help="The image size for inference. Te default setting is 1024x1536. \
                        To evaluate on ETH3D Benchmark, use 768x1024 instead.",
    )
    parser.add_argument(
        "--output", default='./pred_results/'+img_path1[-10:-4]+'_l.png', help="The path of output disparity."
    )
    args = parser.parse_args() # './pred_results/'+args.left[-10:-4]+'_l.png'  default="disparity.png"

    assert os.path.exists(args.model_path), "The model path do not exist."
    assert os.path.exists(args.left), "The left image path do not exist."
    assert os.path.exists(args.right), "The right image path do not exist."

    model_func = load_model(args.model_path)
    left = cv2.imread(args.left)
    right = cv2.imread(args.right)

    assert left.shape == right.shape, "The input images have inconsistent shapes."

    in_h, in_w = left.shape[:2]

    print("Images resized:", args.size)
    eval_h, eval_w = [int(e) for e in args.size.split("x")]
    left_img = cv2.resize(left, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    right_img = cv2.resize(right, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

    pred = inference(left_img, right_img, model_func, n_iter=20)

    t = float(in_w) / float(eval_w)
    disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t

    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    parent_path = os.path.abspath(os.path.join(args.output, os.pardir))
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    cv2.imwrite(args.output, disp_vis)
    print("Done! Result path:", os.path.abspath(args.output))
