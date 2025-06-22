import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import TQC
import torch
from torch import nn
import cv2
import gymnasium as gym

###### VAE ######

class Encoder(nn.Module):
    def __init__(self, latent_dims, width, height):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=2, padding=(0, 0)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            # nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 4), stride=2, padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 4), stride=2, padding=(0, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            # nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )

        test = np.ones((32, 1, width, height))
        with torch.no_grad():
            n_features = self.encoder(
                torch.as_tensor(test).float()
            ).shape[1]
        print("encoder features: ", n_features)
        self.fc_mu = nn.Sequential(nn.Linear(in_features=n_features, out_features=latent_dims))
        self.fc_logvar = nn.Sequential(nn.Linear(in_features=n_features, out_features=latent_dims))

    def forward(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(in_features=latent_dims, out_features=128 * 9 * 12)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 4), stride=(2, 2), padding=(0, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.Sigmoid(),
            )

        test = np.ones((1, 128, 9, 12))
        with torch.no_grad():
            n_features = self.decoder(torch.as_tensor(test).float()).shape

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 9, 12)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        return self.decoder(x)


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim, beta, width, heigth):
        super(VariationalAutoencoder, self).__init__()
        self.beta = beta
        self.encoder = Encoder(latent_dim, width, heigth)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(x[0].detach().cpu().numpy().reshape(300, 400))
        #ax[1].imshow(x_recon[0].detach().cpu().numpy().reshape(300, 400))
        #plt.show()
        return x_recon, latent

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


def load_vae(saved_model_path):
    vae = VariationalAutoencoder(32, 0.5, 300, 400)
    vae.load_state_dict(torch.load(saved_model_path))
    vae = vae.to("cuda")
    vae.eval()
    return vae

###### VPP ######

def load_vpp_agent(model_path):
    return TQC.load(model_path)


def translate_to_other_boundries(a, b, c, d, y):
    return (((y - a) * (c - d)) / (b - a)) + d

def get_action_in_world_coord(action):
    boundaries = np.array([[-0.34, 0.34], [0.45, 0.7], [0.97, 0.97 + 0.2], [-15, 15], [0, 20]]) #workspace bounds
    transformed_action = np.zeros_like(action)
    for idx, b in enumerate(boundaries):
        transformed_action[idx] = translate_to_other_boundries(-1, 1, b[1], b[0], action[idx])
    return transformed_action

def crop_maps(m):
    crop_map = np.ones((300, 400)) * 0.5
    crop_map[5:296, 6:393] = m[142:-1, 45:432]
    return crop_map

def normalization(c_max, dist, c_min=0):
    normed_min, normed_max = 0, 1
    if (dist - c_min) == 0:
        return 0.
    if (c_max - c_min) == 0:
        print("(c_max - c_min) == 0")
    x_normed = (dist - c_min) / (c_max - c_min)
    x_normed = x_normed * (normed_max - normed_min) + normed_min
    return round(x_normed, 4)

def preprocess_images_torch(image):
    unify = image.reshape(1, image.shape[0], image.shape[1])
    current_min, current_max = np.amin(unify), np.amax(unify)
    if current_min == current_max:
        return torch.from_numpy((unify * 0).reshape(1, 1, image.shape[0], image.shape[1])).float()
    normed_min, normed_max = 0, 1
    x_normed = (unify - current_min) / (current_max - current_min)
    x_normed = x_normed * (normed_max - normed_min) + normed_min
    return torch.from_numpy(x_normed.reshape(1, 1, image.shape[0], image.shape[1])).float()

def get_latent_space(m, vae):
    # cut probmap to relevant area
    cropped_map = crop_maps(m)
    reconst_img, latent_space = vae(preprocess_images_torch(cropped_map).to("cuda"))
    return list(latent_space.cpu().detach().numpy().reshape(32))

def get_observation(m, vae, last_target_pos, ig, mc, unknown_area_center):
    observation = []
    # get latent space
    latent = get_latent_space(m, vae)

    observation.extend(latent)
    # get last action [x,y,z, yaw]
    observation.extend(last_target_pos.astype(np.float64))
    #print("ltp: ", last_target_pos)
    # get current entropy
    observation.extend([ig])
    #print("ig: ", ig)
    # get motion cost
    observation.extend([mc])
    #print("mc: ", mc)
    # check if collssion or dropp
    observation.extend([int(False)])
    # get enter of biggest unknown area
    observation.extend(unknown_area_center)
    #print("unknown_area_center: ", unknown_area_center)
    return observation

def get_cnts_boxes_and_rectangles(img, size_thresh=10):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )
    center_points = []
    boxes = []
    processed_cnts = []
    chulls = []
    max_area_idx = 0
    max_area = 0
    inner_count = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area >= size_thresh:
            if area > max_area:
                max_area = area
                max_area_idx = inner_count
            contour[:, :, 0] += 54
            contour[:, :, 1] += 141
            chulls.append(cv2.convexHull(contour, False))
            processed_cnts.append(contour)
            rect = cv2.minAreaRect(contour)
            boxes.append(cv2.boxPoints(rect).astype(np.int32))
            #if area > 100:
            #    # find the center point of the contour
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center_points.append((cX, cY))
            inner_count += 1
    return processed_cnts, chulls, boxes, center_points, max_area_idx

def unknown_cp_in_cartesian(cp, mapping):
    transformed_cp = np.zeros((len(cp), 3))
    for idx, p in enumerate(cp):
        transformed_cp[idx] = mapping.hg.map_pixel_to_world_point(p)
    return transformed_cp


def find_center_point(m, mapping):
    # crop to relevant scene
    cm = m[141:423, 54:425]
    # get contours and polygon of unknown cells
    um = np.where(np.isclose(cm, 0.5, atol=0.0001), cm, 0) * 255
    um = cv2.normalize(um, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    c_u, h_u, b_u, cp_u, max_idx_u = get_cnts_boxes_and_rectangles(um, size_thresh=160)
    unknown_area_center_points = unknown_cp_in_cartesian(cp_u, mapping)

    biggest_cp_u = unknown_area_center_points[max_idx_u]
    return biggest_cp_u

def get_known_unknown_cells(m):
    cropped_map = m[141:423, 54:425]
    uk = np.count_nonzero(np.isclose(cropped_map, 0.5, atol=0.0001))
    k = np.count_nonzero(np.logical_not(np.isclose(cropped_map, 0.5, atol=0.0001)))
    print(uk, k, uk + k, cropped_map.size)
    e = round(uk /cropped_map.size, 4)
    if k + uk !=cropped_map.size:
        print("Something went wrong: unknown and known number sum not up correctly")
    return uk, k, e


def get_information_gain(m, last_m):
    print(m.shape, last_m.shape)
    current_unknown_num, current_known_num, current_entropy = get_known_unknown_cells(m)
    last_unknown_num, last_known_num, last_entropy = get_known_unknown_cells(last_m)
    information_gain = normalization(last_unknown_num, (current_known_num - last_known_num))
    return information_gain, last_entropy - current_entropy

###### Push Prediction ######

class PushPredictions(nn.Module):
    def __init__(self, width, height):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32,  kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64,  kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=64,  kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=64,  kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        test = np.ones((32, 1, width, height))
        with torch.no_grad():
            n_features = self.encoder(
                torch.as_tensor(test).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(in_features=n_features, out_features=1))

    def forward(self, x):
        x = self.encoder(x)
        return self.linear(x)

def load_push_agent(model_path):
    model = PushPredictions(400, 400).to("cuda")
    load_checkpoint(torch.load(model_path), model)
    return model

def load_checkpoint(checkpoint,model):
    print('=>loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
