import argparse
import numpy as np


def binary_bce_with_logits(logits, targets):
    """Stable element-wise BCEWithLogits without pos_weight."""
    return np.logaddexp(0.0, logits) - targets * logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", required=True)
    args = parser.parse_args()

    data = np.load(args.input_npz, allow_pickle=True)

    labels = [
        x.decode() if isinstance(x, bytes) else str(x)
        for x in data["labels"].tolist()
    ]

    y = data["y"].astype(np.float64)
    mask = data["mask"].astype(bool)
    img = data["img_full"].astype(np.float64)
    fus = data["fus_full"].astype(np.float64)

    residual = fus - img

    for k, label in enumerate(labels):
        valid = mask[:, k]

        yy = y[valid, k]
        zi = img[valid, k]
        zf = fus[valid, k]
        rr = residual[valid, k]

        confidence = np.abs(zi)

        # 같은 수의 환자가 들어가도록 confidence quartile 사용
        edges = np.quantile(
            confidence,
            [0.0, 0.25, 0.50, 0.75, 1.0],
        )
        edges[0] = -np.inf
        edges[-1] = np.inf

        loss_img = binary_bce_with_logits(zi, yy)
        loss_fus = binary_bce_with_logits(zf, yy)

        # 양수이면 fusion이 image보다 loss를 줄임
        delta_loss = loss_img - loss_fus

        # residual이 정답 방향으로 움직였는가?
        # y=1이면 residual>0, y=0이면 residual<0이 올바른 방향
        signed_direction = (2.0 * yy - 1.0) * rr
        helpful_direction = signed_direction > 0

        print(f"\nLabel: {label}")
        print(
            f"{'confidence':<18s} "
            f"{'n':>6s} {'pos':>7s} "
            f"{'mean|r|':>10s} "
            f"{'correct_r':>10s} "
            f"{'helped':>9s} "
            f"{'mean_dBCE':>11s}"
        )

        for q in range(4):
            if q < 3:
                selected = (
                    (confidence >= edges[q])
                    & (confidence < edges[q + 1])
                )
            else:
                selected = (
                    (confidence >= edges[q])
                    & (confidence <= edges[q + 1])
                )

            n = int(selected.sum())

            print(
                f"Q{q + 1} "
                f"{'(uncertain)' if q == 0 else '(confident)' if q == 3 else '':<13s} "
                f"{n:>6d} "
                f"{yy[selected].mean():>7.4f} "
                f"{np.abs(rr[selected]).mean():>10.5f} "
                f"{helpful_direction[selected].mean():>10.4f} "
                f"{(delta_loss[selected] > 0).mean():>9.4f} "
                f"{delta_loss[selected].mean():>+11.6f}"
            )

        print(
            f"Overall: helped={(delta_loss > 0).mean():.4f}, "
            f"correct_direction={helpful_direction.mean():.4f}, "
            f"mean_delta_BCE={delta_loss.mean():+.6f}"
        )


if __name__ == "__main__":
    main()