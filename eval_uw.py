"""
eval_uw.py — The generalization exam: score OUR trained models (trained
only on our two objects, our camera) on UW RGB-D Object Dataset frames
(different balls/boxes, different Kinect-class camera).

No retraining. Reports accuracy per resolution and variant, plus the same
single-height-threshold baseline for context.

Expectation stated in advance (honesty): our models leaned on absolute
height (~68 mm ball vs ~36 mm box); UW relief heights are different, so
transfer may be poor — and WHICH features break is the finding.

Usage: python3 eval_uw.py
"""
import os

import numpy as np

import config
from train_model import features

H_LIST = [10, 20, 30, 40]


def predict(model, F):
    Z = (F - model["mu"]) / model["sd"]
    return 1 / (1 + np.exp(-(Z @ model["w"] + float(model["b"])))) > 0.5


def main():
    print(f"{'h':>4} {'variant':>8} {'UW acc':>7} {'ball':>6} {'box':>6}")
    for h in H_LIST:
        d = np.load(os.path.join(config.RESULTS_DIR,
                                 f"uw_dataset_h{h}.npz"),
                    allow_pickle=True)
        y = d["y"]
        for variant, with_size in [("pattern", False), ("full", True)]:
            model = np.load(os.path.join(
                config.RESULTS_DIR, f"model_h{h}_{variant}.npz"),
                allow_pickle=True)
            F, _ = features(d["X"], d["n"], with_size)
            pred = predict(model, F)
            acc = float((pred == y).mean())
            a_ball = float((pred[y == 0] == 0).mean())
            a_box = float((pred[y == 1] == 1).mean())
            print(f"{h:>4} {variant:>8} {acc:>7.3f} "
                  f"{a_ball:>6.3f} {a_box:>6.3f}")
    print("\n(0.5 = coin flip; per-class columns reveal one-sided "
          "failures, e.g. 'calls everything box')")


if __name__ == "__main__":
    main()