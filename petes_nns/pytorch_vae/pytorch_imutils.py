import random
import string

import numpy as np
import torch

from artemis.fileman.local_dir import get_artemis_data_path
from src.peters_stuff.bbox_utils import bbox_to_position
from src.peters_stuff.image_crop_generator import batch_crop
from petes_nns.pytorch_vae.pytorch_helpers import get_default_device



def chanfirst(im):
    return np.rollaxis(im, 3, 1)


def chanlast(im):
    return np.rollaxis(im, 1, 4)


def normalize_image(im):
    return torch.from_numpy((chanfirst(im)/127.)-1.).float()


def denormalize_image(im):
    return ((chanlast(im.detach().cpu().numpy())+1.)*127.)


def generate_random_model_path(code_gen_len=16, suffix='.pth'):
    code = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(code_gen_len))
    model_path = get_artemis_data_path('models/{}{}'.format(code, suffix), make_local_dir=True)
    return model_path


def get_normed_crops_and_position_tensors(img, bboxes, scale = 1.):
    raw_image_crops = batch_crop(img=img, bboxes=bboxes)
    normed_image_crop_tensors = normalize_image(raw_image_crops).to(get_default_device()).float()
    position_tensors = torch.from_numpy(bbox_to_position(bboxes=bboxes, scale=scale, image_size=img.shape[:2])).to(get_default_device()).float()
    return raw_image_crops, normed_image_crop_tensors, position_tensors


if __name__ == '__main__':
    from src.peters_stuff.sample_data import SampleImages
    from src.peters_stuff.image_crop_generator import iter_bbox_batches

    from artemis.plotting.db_plotting import dbplot, hold_dbplots, DBPlotTypes


    img = SampleImages.sistine_512()
    normscale = 0.25

    dbplot(img, 'image')

    for i, bboxes in enumerate(iter_bbox_batches(image_shape=img.shape[:2], crop_size=(64, 64), batch_size=64, position_generator_constructor='normal', n_iter=None, normscale=normscale)):

        raw_image_crops, normed_image_crops, positions = get_normed_crops_and_position_tensors(img=img, bboxes=bboxes, scale=1./normscale)

        with hold_dbplots():
            dbplot(raw_image_crops, 'crops')
            for i, bbox in enumerate(bboxes):
                dbplot(bbox, f'bbox[{i}]', axis='image', plot_type=DBPlotTypes.BBOX_R)
            dbplot((positions[:, 0].numpy(), positions[:, 1].numpy()), plot_type=DBPlotTypes.SCATTER)