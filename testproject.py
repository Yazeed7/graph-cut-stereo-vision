import os

import matplotlib.pyplot as plt
import cv2
import numpy as np

from graph_cut_stereo_correspondence import GraphCutStereoCorrespondance
from naive_stereo_correspondence import NaiveStereoCorrespondence

def process(img_name):
    """
    Given path of directory containing 2005 Middlebury images,
    It computes stereo correspondence using various techniques
    and returns the following figures
        - Naive stereo correspondence .
        - Naive stereo correspondence with disparity awareness.
        - Graph cut minimization with a single alpha-expansion.
        - Ground truth disparity
    """
    img1 = cv2.imread(os.path.join('images', img_name, 'view1.png'))
    img2 = cv2.imread(os.path.join('images', img_name, 'view5.png'))
    gt = cv2.imread(os.path.join('images', img_name, 'disp1.png'), cv2.IMREAD_GRAYSCALE)

    s = 2 #scale
    m,n = img1.shape[:2]
    img1 = cv2.resize(img1, (n//s, m//s))
    img2 = cv2.resize(img2, (n//s, m//s))
    gt = cv2.resize(gt, (n//s, m//s))

    gt = np.around(gt/6).astype(np.uint8)

    disp = gt.max()+1
    step = 1

    # Regular naive SC
    print(f'Calculating naive SC for {img_name} images ... ', end='')
    naive_sc = NaiveStereoCorrespondence(img1, img2, color=True, k_size=3, ndisp=disp, disp_aware=False)
    naive_pred = naive_sc.calculate()
    print('Done!')

    # Disparity aware naive SC
    print(f'Calculating disparity aware naive SC for {img_name} images ... ', end='')
    disp_aware_naive_sc = NaiveStereoCorrespondence(img1, img2, color=True, k_size=3, ndisp=disp, disp_aware=True)
    disp_aware_naive_pred = disp_aware_naive_sc.calculate()
    print('Done!')

    print(f'Calculating graph-cut energy minimization for {img_name} images:')
    graph_cut_sc = GraphCutStereoCorrespondance(img1, img2, disp, color=True, k_size=3, label_step=step)
    graph_cut_pred = graph_cut_sc.calculate_a_expansion()
    # graph_cut_pred = disp_aware_naive_pred
    print('Done!')

    print(f'Saving outputs and statistics for {img_name} in the results folder ... ', end='')

    f = open(os.path.join('results', 'statistics.txt'), 'a')
    preds = [naive_pred, disp_aware_naive_pred, graph_cut_pred]
    names = ['naive', 'disp_aware_naive', 'graph_cut']

    for pred, name in zip(preds, names):
        pred = np.clip(pred, 0, 255).astype(np.uint8)
        pred_m = cv2.medianBlur(pred, 3)
        for i in range(3):
            f.write(f'{img_name}_{name} {i}bad (normal, filtered 3x3 median) = {100 * (np.abs(pred-gt)<=i).sum()/gt.size:.2f}, {100 * (np.abs(pred_m-gt)<=i).sum()/gt.size:.2f}\n')
        f.write('\n')
    f.write('-'*50+'\n\n')
    f.close()

    print('Done!')

    return naive_pred, disp_aware_naive_pred, graph_cut_pred, gt


def main():
    # Clear statistics file
    open(os.path.join('results', 'statistics.txt'), 'w').close()

    imgs = []
    for i, img in enumerate(['Art', 'Books', 'Dolls', 'Laundry', 'Moebius', 'Reindeer']):
        imgs.extend(list(process(img)))

    print('Saving figure ...')
    fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(8,10))
    for ax, im in zip(axes.flat, imgs):
        img = ax.imshow(im, 'gray', vmin=0, vmax=40)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1, 0.015, 0.03, 0.96])
    fig.colorbar(img, cax=cbar_ax)

    cols = ['Naive', 'DA naive', 'Graph-cut', 'Ground-truth']
    rows = ['Art', 'Books', 'Dolls', 'Laundry', 'Moebius', 'Reindeer']

    for i, col in enumerate(cols):
        axes[0][i].set_title(col)
    for i, row in enumerate(rows):
        axes[i][0].set_ylabel(row, rotation=90, size='large')

    plt.ylabel('Pixel Disparities', rotation=90, size='large')

    fig.tight_layout()

    plt.savefig(os.path.join('results', 'cv_proj.png'), dpi=150, bbox_inches='tight')
    print('Done!')


if __name__ == '__main__':
    main()
