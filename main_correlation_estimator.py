"""SeFa."""

import os
import argparse
from tqdm import tqdm
import numpy as np
import cv2

import torch

from models import parse_gan_type
from utils import to_tensor
from utils import postprocess
from utils import load_generator
from utils import factorize_weight
from utils import HtmlPageVisualizer
from classifier.data_celebA_jh import get_input_transform2, CelebADataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Discover semantics from the pre-trained weight.')
    parser.add_argument('model_name', type=str,
                        help='Name to the pre-trained model.')

    parser.add_argument('--classifier_model_path', type=str, required=True,
                        help='Name to the pre-trained classifier model.')

    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save the visualization pages. '
                             '(default: %(default)s)')
    parser.add_argument('-L', '--layer_idx', type=str, default='all',
                        help='Indices of layers to interpret. '
                             '(default: %(default)s)')
    parser.add_argument('-N', '--num_samples', type=int, default=5,
                        help='Number of samples used for visualization. '
                             '(default: %(default)s)')
    parser.add_argument('-K', '--num_semantics', type=int, default=5,
                        help='Number of semantic boundaries corresponding to '
                             'the top-k eigen values. (default: %(default)s)')
    parser.add_argument('--thres_cc', type=float, default=0.75,
                        help='Threshold of correlation-coefficient.'
                             '(default: %(default)s)')
    parser.add_argument('--start_distance', type=float, default=-5.0,
                        help='Start point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--end_distance', type=float, default=5.0,
                        help='Ending point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--step', type=int, default=11,
                        help='Manipulation step on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--viz_size', type=int, default=256,
                        help='Size of images to visualize on the HTML page. '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_psi', type=float, default=0.7,
                        help='Psi factor used for truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_layers', type=int, default=8,
                        help='Number of layers to perform truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling. (default: %(default)s)')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU(s) to use. (default: %(default)s)')
    return parser.parse_args()


def main():
    """Main function."""

    for eigvec_id in tqdm(range(num_eigvecs), desc='Semantic ', leave=False):  # eigenvector idx sorted by eigenvalue
        boundary = boundaries[eigvec_id:eigvec_id + 1]
        for sam_id in tqdm(range(num_sam), desc='Sample ', leave=False):
            code = codes[sam_id:sam_id + 1]

            # For every distance, generate images
            for dist_id, d in enumerate(distances):
                temp_code = code.copy()
                if gan_type == 'pggan':
                    temp_code += boundary * d
                    image = generator(to_tensor(temp_code))['image']
                elif gan_type in ['stylegan', 'stylegan2']:
                    temp_code[:, layers, :] += boundary * d
                    image = generator.synthesis(to_tensor(temp_code))['image']
                image = postprocess(image)[0]

                # The, use these images to evaluate classifier scores.
                cls_model.evaluate_scores(image, dist_id, device)

            # get correlation coefficient using the above scores for the current sample
            cls_model.calculate_cc(sam_id)
            cls_model.reset_scores()

        # get average correlation coefficient correlation across samples. THis average C.C. will be used.
        cls_model.calculate_avg_cc(eigvec_id)
        cls_model.reset_cc_scores()
    return


def visualize_output_eigvec_first():
    # Find classes of high correlation to eigenvector
    cls_cc_pairs_list = cls_model.find_cls_w_high_cc()

    # print the result
    for eig_id, cls_cc_pairs in enumerate(cls_cc_pairs_list):
        print("----- eigen-vector id ", eig_id)
        for cls_id in cls_cc_pairs.keys():
            print("class : ", CelebADataset.class_names[cls_id], ", CC : ", cls_cc_pairs[cls_id])

    # visualize the result
    vizer_1 = HtmlPageVisualizer(num_rows=len(cls_cc_pairs_list) * (num_sam + 1) + 1,
                                 num_cols=args.step + 1,
                                 viz_size=args.viz_size)
    headers = [''] + [f'Distance {d:.2f}' for d in distances]
    vizer_1.set_headers(headers)
    for eig_id, cls_cc_pairs in enumerate(cls_cc_pairs_list):
        vizer_1.set_cell(eig_id * (num_sam + 1), 0,
                         text=f'Semantic {eig_id:03d}<br>({eig_values[eig_id]:.3f})',
                         highlight=True)
        cc_tmp = []
        for id, cls_id in enumerate(cls_cc_pairs.keys()):
            cc_tmp.append(abs(cls_cc_pairs[cls_id]))
        sorted_idx = np.argsort(cc_tmp)[::-1]
        id_real = 0
        for id, idx in enumerate(sorted_idx):
            if id_real >= args.step:
                break
            cls_id = list(cls_cc_pairs.keys())[idx]
            if CelebADataset.class_names[cls_id] in CelebADataset.class_in_attention:
                vizer_1.set_cell(eig_id * (num_sam + 1), id_real+1,
                                 text=f'{CelebADataset.class_names[cls_id]}<br>({cls_cc_pairs[cls_id]:.4f})',
                                 highlight=True)
                id_real += 1

        boundary = boundaries[eig_id:eig_id + 1]
        for sam_id in tqdm(range(num_sam), desc='Sample ', leave=False):
            code = codes[sam_id:sam_id + 1]
            vizer_1.set_cell(eig_id * (num_sam + 1) + sam_id + 1, 0, text=f'Sample {sam_id:03d}')
            for col_id, d in enumerate(distances, start=1):
                temp_code = code.copy()
                if gan_type == 'pggan':
                    temp_code += boundary * d
                    image = generator(to_tensor(temp_code))['image']
                elif gan_type in ['stylegan', 'stylegan2']:
                    temp_code[:, layers, :] += boundary * d
                    image = generator.synthesis(to_tensor(temp_code))['image']
                image = postprocess(image)[0]
                vizer_1.set_cell(eig_id * (num_sam + 1) + sam_id + 1, col_id, image=image)

    prefix = (f'{args.model_name}_'
              f'JH2_N{num_sam}_K{num_eigvecs}_L{args.layer_idx}_seed{args.seed}')
    vizer_1.save(os.path.join(args.save_dir, f'{prefix}_sample_first.html'))
    return


if __name__ == '__main__':

    """Main function."""
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # os.makedirs(args.save_dir, exist_ok=True)

    # Factorize weights.
    generator = load_generator(args.model_name)
    gan_type = parse_gan_type(generator)
    layers, boundaries, eig_values = factorize_weight(generator, args.layer_idx)  # finding eigenvectors and eigenvalues from incorporated given layers.

    # Set random seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Prepare codes.
    codes = torch.randn(args.num_samples, generator.z_space_dim).to(device)
    if gan_type == 'pggan':
        codes = generator.layer0.pixel_norm(codes)
    elif gan_type in ['stylegan', 'stylegan2']:
        codes = generator.mapping(codes)['w']
        codes = generator.truncation(codes,
                                     trunc_psi=args.trunc_psi,
                                     trunc_layers=args.trunc_layers)
    codes = codes.detach().cpu().numpy()

    # Generate visualization pages.
    distances = np.linspace(args.start_distance, args.end_distance, args.step)
    num_sam = args.num_samples
    num_eigvecs = args.num_semantics

    from classifier.custom_model import Classifier
    cls_model = Classifier(args.classifier_model_path, distances, num_sam, num_eigvecs, thres_cc=args.thres_cc, device=device)

    main()
    visualize_output_eigvec_first()
