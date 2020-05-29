import os
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
# from tqdm import tqdm_notebook as tqdm
from utils import utils, warp, image_utils, constant_var
from models import end_2_end_optimization
from options import fake_options
import cv2

# if want to run on CPU, please make it False
constant_var.USE_CUDA = True
utils.fix_randomness()

# if GPU is RTX 20XX, please disable cudnn
torch.backends.cudnn.enabled = True

# set some options
opt = fake_options.FakeOptions()
opt.batch_size = 1
opt.coord_conv_template = True
opt.error_model = 'loss_surface'
opt.error_target = 'iou_whole'
opt.goal_image_path = './data/world_cup_2018.png'
# opt.goal_image_path = './data/3.jpg'
opt.guess_model = 'init_guess'
opt.homo_param_method = 'deep_homography'
opt.load_weights_error_model = 'pretrained_loss_surface'
opt.load_weights_upstream = 'pretrained_init_guess'
opt.lr_optim = 1e-5
opt.need_single_image_normalization = True
opt.need_spectral_norm_error_model = True
opt.need_spectral_norm_upstream = False
opt.optim_criterion = 'l1loss'
opt.optim_iters = 200
opt.optim_method = 'stn'
opt.optim_type = 'adam'
opt.out_dir = './out'
opt.prevent_neg = 'sigmoid'
opt.template_path = './data/world_cup_template.png'
opt.warp_dim = 8
opt.warp_type = 'homography'

# read original image
goal_image = imageio.imread(opt.goal_image_path, pilmode='RGB')
imageio.imwrite('goal_image.jpg', goal_image)

# plt.imshow(goal_image)
# plt.show()

# resize image to square shape, 256 * 256, and squash to [0, 1]
pil_image = Image.fromarray(np.uint8(goal_image))
pil_image = pil_image.resize([256, 256], resample=Image.NEAREST)
goal_image = np.array(pil_image)

imageio.imwrite('goal_image_resize.jpg', goal_image)
# plt.imshow(goal_image)
# plt.show()

# covert np image to torch image, and do normalization
goal_image = utils.np_img_to_torch_img(goal_image)
if opt.need_single_image_normalization:
    goal_image = image_utils.normalize_single_image(goal_image)
print('mean of goal image: {0}'.format(goal_image.mean()))
print('std of goal image: {0}'.format(goal_image.std()))

# read template image
template_image = imageio.imread(opt.template_path, pilmode='RGB')
template_image = template_image / 255.0
if opt.coord_conv_template:
    template_image = image_utils.rgb_template_to_coord_conv_template(template_image)

imageio.imwrite('template_image.jpg', template_image)
# plt.imshow(template_image)
# plt.show()

# covert np image to torch image, and do normalization
template_image = utils.np_img_to_torch_img(template_image)
if opt.need_single_image_normalization:
    template_image = image_utils.normalize_single_image(template_image)
print('mean of template: {0}'.format(template_image.mean()))
print('std of template: {0}'.format(template_image.std()))

e2e = end_2_end_optimization.End2EndOptimFactory.get_end_2_end_optimization_model(opt)
orig_homography, optim_homography = e2e.optim(goal_image[None], template_image)

# reload image and template for visualization
# overload goal image
goal_image_draw = imageio.imread(opt.goal_image_path, pilmode='RGB')
goal_image_draw = goal_image_draw / 255.0
outshape = goal_image_draw.shape[0:2]

# overload template image
template_image_draw = imageio.imread(opt.template_path, pilmode='RGB')
template_image_draw = template_image_draw / 255.0
template_image_draw = image_utils.rgb_template_to_coord_conv_template(template_image_draw)
template_image_draw = utils.np_img_to_torch_img(template_image_draw)

# warp template image with initial guess
warped_tmp_orig = warp.warp_image(template_image_draw, orig_homography, out_shape=outshape)[0]
warped_tmp_orig = utils.torch_img_to_np_img(warped_tmp_orig)

imageio.imwrite('warped_tmp_orig.jpg', warped_tmp_orig)
# plt.imshow(warped_tmp_orig)
# plt.show()

# warp template image with optimized guess
warped_tmp_optim = warp.warp_image(template_image_draw, optim_homography, out_shape=outshape)[0]
warped_tmp_optim = utils.torch_img_to_np_img(warped_tmp_optim)

imageio.imwrite('warped_tmp_optim.jpg', warped_tmp_optim)
# plt.imshow(warped_tmp_optim)
# plt.show()

# show initial guess overlay
show_image = np.copy(goal_image_draw)
valid_index = warped_tmp_orig[:, :, 0] > 0.0
overlay = (goal_image_draw[valid_index].astype('float32') + warped_tmp_orig[valid_index].astype('float32'))/2
show_image[valid_index] = overlay

imageio.imwrite('overlay.jpg', show_image)
# plt.imshow(show_image)
# plt.show()

# show optimized guess overlay
show_image = np.copy(goal_image_draw)
valid_index = warped_tmp_optim[:, :, 0] > 0.0
overlay = (goal_image_draw[valid_index].astype('float32') + warped_tmp_optim[valid_index].astype('float32'))/2
show_image[valid_index] = overlay

imageio.imwrite('overlay_opt.jpg', show_image)
# plt.imshow(show_image)
# plt.show()

opt.optim_iters = 80

video_cap = cv2.VideoCapture('./data/sample.mp4')
frame_list = []
while True:
    success, image = video_cap.read()
    if not success:
        break
    pil_image = Image.fromarray(np.uint8(image[..., ::-1]))
    pil_image = pil_image.resize([1280, 720], resample=Image.NEAREST)
    image = np.array(pil_image)
    frame_list.append(image)

orig_homography_list = []
for idx, frame in enumerate(frame_list):
    pil_image = Image.fromarray(np.uint8(frame))
    pil_image = pil_image.resize([256, 256], resample=Image.NEAREST)
    frame = np.array(pil_image)
    frame = utils.np_img_to_torch_img(frame)
    if opt.need_single_image_normalization:
        frame = image_utils.normalize_single_image(frame)
    orig_homography = e2e.homography_inference.infer_upstream_homography(frame[None])
    orig_homography_list.append(orig_homography.detach())

first_frame = True
optim_homography_list = []
for idx, frame in enumerate(frame_list):
    print('{0} / {1}'.format(idx+1, len(frame_list)))
    pil_image = Image.fromarray(np.uint8(frame))
    pil_image = pil_image.resize([256, 256], resample=Image.NEAREST)
    frame = np.array(pil_image)
    frame = utils.np_img_to_torch_img(frame)
    if opt.need_single_image_normalization:
        frame = image_utils.normalize_single_image(frame)
    _, optim_homography = e2e.optim(frame[None], template_image, refresh=first_frame)
    optim_homography_list.append(optim_homography.detach())
    first_frame = False

warped_tmp_orig_list = []
warped_tmp_optim_list = []
for orig_h, optim_h in zip(orig_homography_list, optim_homography_list):
    warped_tmp_orig = warp.warp_image(template_image_draw, orig_h, out_shape=(720, 1280))[0]
    warped_tmp_orig = utils.torch_img_to_np_img(warped_tmp_orig)
    warped_tmp_orig_list.append(warped_tmp_orig)
    warped_tmp_optim = warp.warp_image(template_image_draw, optim_h, out_shape=(720, 1280))[0]
    warped_tmp_optim = utils.torch_img_to_np_img(warped_tmp_optim)
    warped_tmp_optim_list.append(warped_tmp_optim)

def save_to_vid(frame_list, template_list, fname):
    video_name = fname
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','J','P','G'), 25, (1280, 720))
    edge_color = [0, 1.0, 0]
    for frame, template in zip(frame_list, template_list):
        video_content = frame[..., ::-1] / 255.0
        template_content = template
        valid_index = template_content[..., 0]>0
        edge_index = template_content[..., 0] >= 254.0/255.0
        overlay = (video_content[valid_index].astype('float32') + template_content[valid_index].astype('float32'))/2
        out_frame = video_content.copy()
        out_frame[valid_index] = overlay
        out_frame[edge_index] = edge_color
        out_frame = out_frame * 255.0
        out_frame = out_frame.astype('uint8')
        video.write(out_frame)
    cv2.destroyAllWindows()
    video.release()
    os.system("/usr/bin/ffmpeg -y -i {0} -vcodec libx264 {1}".format(fname, fname.replace('.mp4', '_h264.mp4')))

save_to_vid(frame_list, warped_tmp_orig_list, './orig_overlay.mp4')
save_to_vid(frame_list, warped_tmp_optim_list, './optim_overlay.mp4')