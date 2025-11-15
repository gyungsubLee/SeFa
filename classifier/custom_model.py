import cv2
import numpy as np
import torch

from classifier.data_celebA_jh import get_input_transform2

"""
from architecture.resnet import resnext101_32x8d
def build_model(num_classes=40, gpu=None, pretrained=True, fine_tuning_fc=True):
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'resnext50_32x4d', pretrained=True)
    # or
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'resnext101_32x8d', pretrained=True)
    model = resnext101_32x8d(pretrained=pretrained)
    if fine_tuning_fc:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    # model.fc.requires_grad_()
    # model.add_module('binary_sigmoid', torch.nn.Sigmoid())
    if gpu is not None:
        model = model.cuda(gpu)
    return model
"""


class Classifier:
    def __init__(self, classifier_model_path, distances, num_samples, num_eigvecs, thres_cc=0.77, device=None):
        self.model = torch.load(classifier_model_path, map_location=device)  # lambda storage, loc: storage.cuda())
        self.model.eval()
        self.cls_input_transform = get_input_transform2()

        self.distances = distances
        self.num_samples = num_samples
        self.cls_scores = np.zeros((len(distances), self.model.fc.out_features), float)
        self.cls_scores_cc = np.zeros((num_samples, self.model.fc.out_features), float)
        self.cls_scores_all_samples = np.zeros((num_eigvecs, self.model.fc.out_features), float)

        self.thres_cc = thres_cc
        pass

    def evaluate_scores(self, image, dist_id, device):
        """
        get classifier scores
        @param image: numpy array image
        @param dist_id: index of distance
        @rtype int, 0
        """
        input_img = cv2.resize(image, (256, 256))
        input_img = self.cls_input_transform(input_img)
        input_img = input_img.unsqueeze(0).to(device)

        cls_out = self.model(input_img)
        cls_out = torch.nn.Sigmoid()(cls_out)
        self.cls_scores[dist_id] = cls_out.squeeze(0).detach().cpu().numpy()  # numpy array of classification scores
        return 0  # cls_out.squeeze(0).detach().cpu().numpy()

    def reset_scores(self):
        self.cls_scores.fill(0)

    def calculate_cc(self, sample_id):
        cls_scores = self.cls_scores.transpose((1, 0))
        self.cls_scores_cc[sample_id] = np.asarray(
            [np.corrcoef(cls_scores[i], self.distances)[1, 0] for i in range(self.model.fc.out_features)])

    def reset_cc_scores(self):
        self.cls_scores_cc.fill(0)

    def calculate_avg_cc(self, eigvec_id):
        fisher_z = np.asarray([np.arctanh(np.clip(self.cls_scores_cc[i], -1 + 2 ** -10, 1 - 2 ** -10)) for i in
                               range(self.num_samples)]).mean(axis=0)  # take average over samples in fisher's z space
        self.cls_scores_all_samples[eigvec_id] = np.tanh(fisher_z)  # back to the cc value

    def find_eigenvec_w_high_cc(self):
        """

        @return: dictionary where keys are class indexes
        and values are a sorted list of (eigen-vector index, correlation-coefficient) pairs
        that absolute value of corr-coefs are above the threshold.
        eg. {1: [(6, 0.9452), (4, -0.9127)], 2: [], 3: [(1, 0.9375)]}
        """
        cc_by_cls = self.cls_scores_all_samples.transpose(1, 0)
        ret = dict()
        for cls_id in range(cc_by_cls.shape[0]):
            cnt_above_thres = sum(abs(cc_by_cls[cls_id]) > self.thres_cc)
            eigvec_id_above_thr = np.zeros(0, dtype=int) if cnt_above_thres == 0 \
                else abs(cc_by_cls[cls_id]).argsort()[-cnt_above_thres:][::-1]

            ret[cls_id] = [(eig_idx, cc_by_cls[cls_id, eig_idx]) for eig_idx in eigvec_id_above_thr]
        return ret

    def find_cls_w_high_cc(self):
        ret_eigenvec = list()  # eigenvector list that contains classes-cc pairs of high cc.
        for eigvec_id in range(self.cls_scores_all_samples.shape[0]):
            dict_cls = dict()
            for cls_id, cc in enumerate(self.cls_scores_all_samples[eigvec_id]):
                if abs(cc) > self.thres_cc:
                    dict_cls[cls_id] = cc
            ret_eigenvec.append(dict_cls)
        return ret_eigenvec
