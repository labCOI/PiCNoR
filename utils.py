import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.transform import EuclideanTransform
from skimage.measure import ransac
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

from itertools import combinations
import networkx as nx

import shutil
from DCN import DCN

import seaborn as sns

from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import rbd, numpy_image_to_torch, read_image
import torch




def loadImage(address):
    """
    loads an Image using cv2.

    Parameters:
        address: image file address.

    Returns:
        Loaded Image.
    """
    if not os.path.exists(address):
        raise FileNotFoundError(f"Error: image not found at {address}")
    return cv2.imread(address)

def changeImageColor(img, colorcode):
    """
    Change Image Color to desired colorcode

    Parameters:
        img: image file address.
        colorcode: "Gray" or "RGB"

    Returns:
        Converted Image.
    """
    if colorcode =="Gray" and len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif colorcode =="RGB" and len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Error converting image color") 

def overlay(img1,img2):
    """
    overlay two images

    Parameters:
        img1: red channel.
        img2: green-blue channels.

    Returns:
        Overlayed Image
    
    """
    if img1.shape != img2.shape:
        raise ValueError("The two images are not the same size.")
    height, width = img1.shape[:2]
    composite = np.zeros((height, width, 3), dtype=np.uint8)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    composite[..., 0] = img1_gray  # Red channel
    composite[..., 1] = img2_gray  # Green channel
    composite[..., 2] = img2_gray  # Blue channel
    return composite

def images_overlay(source, target, show, save, saveName):
    """
    show and save the overlay of source and target images

    Parameters:
        source: source image.
        target: target image.
        save: boolean to allow saving.
        saveName: file name for saving.

    Returns:
    None
    """
    ratios = [4 / 3]
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, ax = plt.subplots(
    1, 1, figsize=figsize, dpi=100, gridspec_kw={"width_ratios": ratios}
    )
    plt.imshow(overlay(source, target))
    plt.axis('off')
    if save:
        plt.savefig(f"{saveName}.png")
    if not show:
        plt.clf()
    return




def draw_keypoints(img, keypoints, color=(0, 255, 255)):
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(img, (int(x), int(y)), color=color, radius=5, thickness=-1)
def conv_DCN(M):
    p0 = np.sqrt( M[0,0] + M[1,0]*1j)
    p1 = (M[0,2] + M[1,2]*1j)/(2*p0)
    return DCN(p0,p1)
def ssd(image1, image2) -> float:
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape.")
    image1_normalized = cv2.normalize(image1, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image2_normalized = cv2.normalize(image2, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    squared_differences = np.square(image1_normalized - image2_normalized)
    ssd_error = np.sum(squared_differences)
    return ssd_error
def blend_DCN(Cs,w) -> DCN:
    c_hat_temp = [ci.scalar_mult(wi) for wi,ci in zip(w,Cs)]
    c_hat = sum(c_hat_temp)
    c_hat = c_hat/c_hat.norm()
    return c_hat


def draw_all_matches(img1, img2, matches, keypoints1, keypoints2, save, outFolder):

    all_matches_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, 
                                        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(all_matches_image)
    if save:
        plt.imsave(f"{outFolder}/all_matches.jpg", all_matches_image)

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def create_weighted_graph(points):
    G = nx.Graph()
    for i, point in enumerate(points):
        G.add_node(i, pos=point)
    for (i, point1), (j, point2) in combinations(enumerate(points), 2):
        distance = euclidean_distance(point1, point2)
        G.add_edge(i, j, weight=round(distance,3))
    return G

def grad2(v):
    a = np.array([(v[i+1] + v[i-1] - 2*v[i]) for i in range(1,len(v)-1)])
    return a

def find_direct_neighbor(G, target_node):
    neighbors = []
    for neighbor in G.neighbors(target_node):
        edge_weight = G[target_node][neighbor]['weight']
        neighbors.append(([neighbor, 1 / edge_weight if edge_weight != 0 else float('inf')]))
    return neighbors

def visualize_entire_graph(G, size, show, save, outFolder):
    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    if True:
        plt.figure(figsize=(size[0],size[1]))
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=12, font_weight='bold', edge_color='gray')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        if save:
            plt.savefig(f"{outFolder}/graph.png")

        plt.show(block=False)
    
def draw_inliers(img1, img2, kp1, kp2, mask, save, outFolder):
    source_inlier_keypoints = [kp for kp, inlier in zip(kp1, mask) if inlier]
    target_inlier_keypoints = [kp for kp, inlier in zip(kp2, mask) if inlier]
    # source_image_with_inliers = cv2.drawKeypoints(source_image, source_inlier_keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # target_image_with_inliers = cv2.drawKeypoints(target_image, target_inlier_keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    inliers_image = np.hstack((img1, img2))
    for i, (src_kp, tgt_kp) in enumerate(zip(source_inlier_keypoints, target_inlier_keypoints)):
        src_pt = (int(src_kp.pt[0]), int(src_kp.pt[1]))
        tgt_pt = (int(tgt_kp.pt[0]), int(tgt_kp.pt[1]))
        cv2.circle(img1, src_pt, 5, (255, 0, 0), -1)
        cv2.circle(img2, tgt_pt, 5, (255, 0, 0), -1)
        cv2.line(inliers_image, (src_pt[0], src_pt[1]), (tgt_pt[0] + img1.shape[1], tgt_pt[1]), (255, 0, 0), 1)
    plt.imshow(inliers_image)

    if save:
        plt.imsave(f"{outFolder}/inliers.jpg", inliers_image)
    plt.show(block=False)




def createFinalOutputs(sFolder, dFolder, nFiles):

    os.makedirs(dFolder, exist_ok=True)

    for folder_name in range(nFiles):
        current_folder = os.path.join(sFolder, str(folder_name))
        source_file = os.path.join(current_folder, 'SourceTransformed_fineT.jpg')
        new_file_name = f'{folder_name}.jpg'
        destination_file = os.path.join(dFolder, new_file_name)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, destination_file)

