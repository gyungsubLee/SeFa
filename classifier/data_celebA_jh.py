import os
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.dataset import T_co
import torchvision.transforms as transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def get_input_transform(input_size):
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.499, 0.399, 0.431], std=[0.273, 0.254, 0.251]),
    ])


def get_input_transform2():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.499, 0.399, 0.431], std=[0.273, 0.254, 0.251]),
    ])


class CelebADataset(data.Dataset):

    class_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
                   'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                   'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                   'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                   'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                   'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
                   'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                   'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    class_in_attention = ['Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses',
                          'Gray_Hair', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
                          'No_Beard', 'Oval_Face', 'Pale_Skin', 'Sideburns',
                          'Smiling', 'Wearing_Hat',
                          'Wearing_Lipstick', 'Young']

    def __init__(self, trainingset_root='/f_data1/TrainingSets', input_size=(256, 256)):
        super(CelebADataset, self).__init__()
        need_attr_clusters = False

        # no_count_attrs = ['Big_Lips', 'Oval_Face', 'Pointy_Nose', 'Wearing_Necklace']

        self.data = list()
        self.input_transform = get_input_transform(input_size)
        self.image_root = os.path.join(trainingset_root, 'CelebA/Img/img_celeba')
        annotation_root = os.path.join(trainingset_root, 'CelebA/Anno/')
        attr_list_file_name = 'list_attr_celeba.txt'

        with open(os.path.join(annotation_root, attr_list_file_name), 'r') as f_attr_list:
            n_file = f_attr_list.readline()  # the first line stores the total number of images (202599)
            n_file = int(n_file.strip())
            attrs = f_attr_list.readline()  # the second line stores attribute names
            self.attrs = attrs.strip().split(' ')

            if need_attr_clusters:
                self.attr_dict = dict()
                for att in attrs:
                    self.attr_dict[att] = list()
                # special for Female
                self.attr_dict['Female'] = list()
                # self.attrs.append('Female')

            # img_list = list()
            read_cnt = 0
            while True:
                img_data = f_attr_list.readline()
                if not img_data: break

                img_data = img_data.strip().split()
                # noinspection PyTypeChecker
                # img_data = img_data[:1] + [int(t) for t in img_data[1:]]
                img_data = img_data[:1] + [int(t) if t == '1' else 0 for t in img_data[1:]]

                self.data.append(img_data)

                if need_attr_clusters:
                    for i, data in enumerate(self.attr_dict.keys()):
                        if img_data[i + 1] == '1':
                            self.attr_dict[data].append(img_data[0])
                        elif data == 'Male' and img_data[i + 1] == '0':
                            self.attr_dict['Female'].append(img_data[0])

                # img_list.append(img_data)
                read_cnt += 1

            assert len(self.data) == n_file, "The size of dataset does not match"

        pass

    def __getitem__(self, index) -> T_co:
        input_img = default_loader(os.path.join(self.image_root, self.data[index][0]))
        if self.input_transform:
            input_img = self.input_transform(input_img)
        # binary_gt = np.asarray(self.data[index][1], np.int8)
        binary_gts = torch.tensor(self.data[index][1:], dtype=torch.float32)  # .unsqueeze(0)
        return input_img, binary_gts

    def __len__(self):
        return len(self.data)

    def get_attribute_list(self):
        return self.attrs

    def get_images_in_attr(self, attr):
        return self.attr_dict[attr]

    def get_mean_n_std(self):
        mean_rgb = list()
        std_rgb = list()
        for data_t in self.data:
            img_t = default_loader(os.path.join(self.image_root, data_t[0]))
            img_np = np.asarray(img_t) / 255
            mean_rgb.append(np.mean(img_np, axis=(0, 1)))
            std_rgb.append(np.std(img_np, axis=(0, 1)))

        mean_r = np.mean([m[0] for m in mean_rgb])
        mean_g = np.mean([m[1] for m in mean_rgb])
        mean_b = np.mean([m[2] for m in mean_rgb])
        std_r = np.mean([s[0] for s in std_rgb])
        std_g = np.mean([s[1] for s in std_rgb])
        std_b = np.mean([s[2] for s in std_rgb])
        return [mean_r, mean_g, mean_b], [std_r, std_g, std_b]


if __name__ == '__main__':
    celeba = CelebADataset()

    for img, label in celeba:
        print('tt')

    print('done')
