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

from kneed import KneeLocator
from itertools import combinations
import networkx as nx

import shutil
from DCN import DCN
import sys

import hdbscan
import seaborn as sns

from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import rbd, numpy_image_to_torch, read_image
import torch

from spectralnet import SpectralNet


random_seed = 3
np.random.seed(random_seed)
torch.manual_seed(random_seed)

def overlay(im1,im2, alpha=0.5):
    im_out = np.zeros_like(im1)
    im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    im_out[:,:,0] = im1
    im_out[:,:,1] = im2
    return im_out

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
def loadImage(address, logging):
    logging.info(f"Loading image from {address}")
    if not os.path.exists(address):
        logging.error(f"Error: Unable to load image from {address}")
        raise FileNotFoundError(f"Error: image not found at {address}")
    return cv2.imread(address)
def changeImageColor(img1, img2, colorcode, logging):
    if colorcode =="Gray":
        logging.info(f"Converting to {colorcode}")
        return cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    elif colorcode =="RGB":
        logging.info(f"Converting to {colorcode}")
        return cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    else:
        logging.error(f"Error: Undefined color scale {colorcode}")
        raise ValueError(f"Colorcode must be Gray or RGB, but got {colorcode}") 
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


def regPair(source, target, outFolder, colorScale, detector, threshold, maxkps, matcher, nclusters, save, fix, show, logging):
    logging.info(f"Registering{source} and {target}, saving {save}, output {outFolder}")
    # Loading Images
    source_image = loadImage(source, logging)
    target_image = loadImage(target, logging)
    source_image, target_image = changeImageColor(source_image, target_image, colorScale, logging)
    logging.info(f"images shape: {source_image.shape}, {target_image.shape}")
    if show:
        ratios = [4 / 3]
        figsize = [sum(ratios) * 4.5, 4.5]
        fig, ax = plt.subplots(
        1, 1, figsize=figsize, dpi=100, gridspec_kw={"width_ratios": ratios}
        )
        plt.imshow(overlay(source_image, target_image))
        plt.title(f"Overlay Source , Target Images , ssd={ssd(source_image,target_image)}")
        plt.axis('off')
    # Initializing Detector
    if detector == "SIFT":
        logging.info(f"Initializing  {detector} with {threshold}")
        detector = cv2.SIFT_create(contrastThreshold=threshold)
    elif detector == "AKAZE":
        logging.info(f"Initializing  {detector} with {threshold}")
        detector = cv2.AKAZE_create(threshold=threshold)
    elif detector == "SUPER":
        logging.info(f"Initializing  {detector} with {maxkps}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor = SuperPoint(max_num_keypoints=maxkps).eval().to(device)

    else:
        logging.error(f"Error: Undefined detector {detector}")
        raise ValueError(f"Detector must be SIFT or AKAZE or SUPER, but got {detector}") 
    
    # Feature Extarction
    logging.info(f"Starting  Global registration")
    stime = datetime.now()
    i1 = source_image.copy()
    i2 = target_image.copy()
    logging.info(f"SSD original image pairs: {ssd(i1,i2)}")

    if detector == "SUPER":     
        image1 = numpy_image_to_torch(i1)
        image2 = numpy_image_to_torch(i2)
        feats1 = extractor.extract(image1.to(device))
        feats2 = extractor.extract(image2.to(device))
        f1, f2 = [
            rbd(x) for x in [feats1, feats2]
        ]
        kp_temp = f1["keypoints"]
        keypoints_source = tuple(
            [cv2.KeyPoint(x=int(k[0]),y=int(k[1]),size=1) for k in kp_temp]
        )
        kp_temp = f2["keypoints"]
        keypoints_target = tuple(
            [cv2.KeyPoint(x=int(k[0]),y=int(k[1]),size=1) for k in kp_temp]
        )
    else:
        keypoints_source, descriptors_source = detector.detectAndCompute(source_image,None)
        keypoints_target, descriptors_target = detector.detectAndCompute(target_image,None)
    if show:
        src_pts = np.uint([ [k.pt[0],k.pt[1]] for k in keypoints_source ]).reshape(-1,2) #i1
        dst_pts = np.uint([ [k.pt[0],k.pt[1]] for k in keypoints_target ]).reshape(-1,2) #i2
        viz2d.plot_images([source_image, target_image])
        viz2d.plot_keypoints([src_pts, dst_pts], ps=10)
    if save:
        src_pts = np.uint([ [k.pt[0],k.pt[1]] for k in keypoints_source ]).reshape(-1,2) #i1
        dst_pts = np.uint([ [k.pt[0],k.pt[1]] for k in keypoints_target ]).reshape(-1,2) #i2
        viz2d.plot_images([source_image, target_image])
        viz2d.plot_keypoints([src_pts, dst_pts], ps=10)
        viz2d.save_plot(f"{outFolder}/keypoints.jpg")
    # Matching
    if matcher =="L2":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        logging.info(f"Matcher: {matcher}")
    elif matcher == "Hamming":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        logging.info(f"Matcher: {matcher}")
    elif matcher == "Light":
        lightmatcher = LightGlue(features="superpoint").eval().to(device)
        logging.info(f"Matcher: {matcher}")
    else:
        logging.error(f"Error: Undefined mathcer {matcher}")
        raise ValueError(f"Matcher must be L2 or Hamming, but got {matcher}")
    if matcher == "Light":
        matches01 = lightmatcher({"image0": feats1, "image1": feats2})
        matches01 = [
            rbd(x) for x in [matches01]
        ][0]
        match_temp = matches01["matches"].numpy().copy()
        matches = [
            cv2.DMatch(_queryIdx=match[0],_trainIdx=match[1], _distance=1) for match in match_temp
        ]
    else:
        matches = bf.match(descriptors_source,descriptors_target)
        matches = sorted(matches, key = lambda x:x.distance)
    if show:
        src_pts = np.float32([ keypoints_source[m.queryIdx].pt for m in matches ]).reshape(-1,2) #i1
        dst_pts = np.float32([ keypoints_target[m.trainIdx].pt for m in matches ]).reshape(-1,2) #i2
        viz2d.plot_images([source_image, target_image])
        viz2d.plot_matches(src_pts, dst_pts, color="lime", lw=0.2)
    if save:
        src_pts = np.float32([ keypoints_source[m.queryIdx].pt for m in matches ]).reshape(-1,2) #i1
        dst_pts = np.float32([ keypoints_target[m.trainIdx].pt for m in matches ]).reshape(-1,2) #i2
        viz2d.plot_images([source_image, target_image])
        viz2d.plot_matches(src_pts, dst_pts, color="lime", lw=0.2)
        viz2d.save_plot(f"{outFolder}/all_matches.jpg")
    logging.info(f"Number of Matched Keypoints: {len(matches)}")
    src_pts = np.float32([ keypoints_source[m.queryIdx].pt for m in matches ]).reshape(-1,2) #i1
    dst_pts = np.float32([ keypoints_target[m.trainIdx].pt for m in matches ]).reshape(-1,2) #i2
    # Transform
    logging.info(f"Finding Euclidean Transform")
    model_robust, inlierIndex = ransac((src_pts, dst_pts), EuclideanTransform, min_samples=3,
                                    residual_threshold=9, max_trials=2000)
    M = model_robust.params
    logging.info(f"Euclidean Transform found \n {M}\n inliers:{np.sum(inlierIndex)}")
    keypoints_source_match = [keypoints_source[m.queryIdx] for m in matches]
    keypoints_target_match = [keypoints_target[m.trainIdx] for m in matches]
    if show:
        source_inlier_keypoints = [kp for kp, inlier in zip(keypoints_source_match, inlierIndex) if inlier]
        target_inlier_keypoints = [kp for kp, inlier in zip(keypoints_target_match, inlierIndex) if inlier]
        src_pts = np.uint([ [k.pt[0],k.pt[1]] for k in source_inlier_keypoints ]).reshape(-1,2) #i1
        dst_pts = np.uint([ [k.pt[0],k.pt[1]] for k in target_inlier_keypoints ]).reshape(-1,2) #i2
        viz2d.plot_images([source_image, target_image])
        viz2d.plot_matches(src_pts, dst_pts, color="tomato", lw=0.5)
    if save:
        source_inlier_keypoints = [kp for kp, inlier in zip(keypoints_source_match, inlierIndex) if inlier]
        target_inlier_keypoints = [kp for kp, inlier in zip(keypoints_target_match, inlierIndex) if inlier]
        src_pts = np.uint([ [k.pt[0],k.pt[1]] for k in source_inlier_keypoints ]).reshape(-1,2) #i1
        dst_pts = np.uint([ [k.pt[0],k.pt[1]] for k in target_inlier_keypoints ]).reshape(-1,2) #i2
        viz2d.plot_images([source_image, target_image])
        viz2d.plot_matches(src_pts, dst_pts, color="tomato", lw=0.5)
        viz2d.save_plot(f"{outFolder}/inliers.jpg")
    source_transformed = cv2.warpPerspective(source_image, M, (i2.shape[1],i2.shape[0]),flags=cv2.INTER_LINEAR)
    if show:
        ratios = [4 / 3]
        figsize = [sum(ratios) * 4.5, 4.5]
        fig, ax = plt.subplots(
        1, 1, figsize=figsize, dpi=100, gridspec_kw={"width_ratios": ratios}
        )
        plt.imshow(source_transformed)
        plt.title("Source Transformed")
        plt.axis('off')
        plt.show(block=False)

    #Fixing Transformed image
    if fix:
        _, mask = cv2.threshold(source_transformed, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        kernel = np.ones((3, 3), np.uint8)
        mask_inv_dilated = cv2.dilate(mask_inv, kernel, iterations=1)
        source_transformed[mask_inv_dilated == 255] = [255]
    if show:
            ratios = [4 / 3]
            figsize = [sum(ratios) * 4.5, 4.5]
            fig, ax = plt.subplots(
            1, 1, figsize=figsize, dpi=100, gridspec_kw={"width_ratios": ratios}
            )
            plt.imshow(source_transformed)
            plt.title("Source Transformed Fixed")
            plt.axis('off')
            plt.show(block=False)
    #Save
    plt.imsave(f"{outFolder}/SourceTransformed.jpg", source_transformed)
    etime = datetime.now()
    if show:
        viz2d.plot_images([source_image,target_image])
        # plt.imshow(np.hstack((source_transformed,target_image)))
        plt.title("Source Transformed + Target Image")
        plt.axis('off')
        plt.show(block=False)
    logging.info(f"Global registration Done")
    logging.info(f"SSD Globally registered image pairs: {ssd(source_image,target_image)}")

    logging.info(f"start time: {stime}\n final time: {etime}\n elapsed time: {(etime-stime).total_seconds()}")
    # Fine Registration
    logging.info(f"Starting Fine registration")
    source_image = source_transformed.copy()
    target_image = target_image.copy()
    if show:
        ratios = [4 / 3]
        figsize = [sum(ratios) * 4.5, 4.5]
        fig, ax = plt.subplots(
        1, 1, figsize=figsize, dpi=100, gridspec_kw={"width_ratios": ratios}
        )
        plt.imshow(overlay(source_image, target_image))
        plt.title(f"Overlay Source Transformed , Target Image, ssd={ssd(source_image,target_image)}")
        plt.axis('off')
    i1 = source_image.copy()
    i2 = target_image.copy()
    if detector == "SUPER":    
        image1 = numpy_image_to_torch(i1)
        image2 = numpy_image_to_torch(i2)
        feats1 = extractor.extract(image1.to(device))
        feats2 = extractor.extract(image2.to(device))
        f1, f2 = [
            rbd(x) for x in [feats1, feats2]
        ]
        kp_temp = f1["keypoints"]
        keypoints_source = tuple(
            [cv2.KeyPoint(x=int(k[0]),y=int(k[1]),size=1) for k in kp_temp]
        )
        kp_temp = f2["keypoints"]
        keypoints_target = tuple(
            [cv2.KeyPoint(x=int(k[0]),y=int(k[1]),size=1) for k in kp_temp]
        )
    else:
        keypoints_source, descriptors_source = detector.detectAndCompute(source_image,None)
        keypoints_target, descriptors_target = detector.detectAndCompute(target_image,None)
    if nclusters != 'auto':
        logging.info(f"Number of Clusters: {nclusters}")
        logging.info("Fitting GMM to keypoints")
        nclusters = int(nclusters)
        gmm = GaussianMixture(n_components=nclusters, random_state=0,covariance_type='full', n_init=10, max_iter=500)
        gmm.fit(np.float32([ k.pt for k in keypoints_source ]).reshape(-1,2))
        logging.info("GMM fitting done")
        logging.info(f"GMM means: {gmm.means_}")
    else: 
        logging.info(f"Finding the Optimum number of clusters")
        data = np.float32([ k.pt for k in keypoints_source ]).reshape(-1,2)
        n_components = np.arange(2, 20, 1)
        bics = []
        sils = []
        for n in n_components:
            gmm = GaussianMixture(n_components=n, random_state=0, covariance_type='full', n_init=10, max_iter=500)
            gmm.fit(data)
            bics.append(gmm.bic(data))
            sils.append(silhouette_score(data,gmm.fit_predict(data)))
        
        max_curvature_idx = np.argmax(np.abs(grad2(bics)))
        plt.figure(figsize=(10,6))
        plt.plot(n_components,bics,'bo-',markersize=5)
        plt.xlim([1,20])
        plt.xticks(n_components)
        plt.plot(n_components[max_curvature_idx+1], bics[max_curvature_idx+1], 'ro', markersize=7)
        plt.savefig(f"{outFolder}/components.png")
        plt.show(block=False)
        nclusters = n_components[max_curvature_idx+1]
        # nclusters = n_components[np.argmax(sils)]
        logging.info(f"Optimal Cluster Numbers: {nclusters}")
        logging.info("Fitting GMM to keypoints")
        # gmm = GaussianMixture(n_components=nclusters, means_init=cluster_centers,  weights_init=np.ravel(np.ones([nclusters,1])/nclusters),
        #                       random_state=0, covariance_type='full', n_init=10, max_iter=500)
        gmm = GaussianMixture(n_components=nclusters,  weights_init=np.ravel(np.ones([nclusters,1])/nclusters),
                              random_state=0, covariance_type='full', n_init=10, max_iter=500)
        gmm.fit(data)
        logging.info("GMM fitting done")
        logging.info(f"GMM means: {gmm.means_}")

    valid_centers = np.zeros([nclusters,1])
    height, width = i1.shape[:2]
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    positions = np.column_stack([xs.flatten(), ys.flatten()])
    colors = np.random.randint(50, 230, size=(nclusters, 3), dtype=np.uint8)
    pred = gmm.predict(positions)
    predicted_image = pred.reshape(height, width)
    color_image = colors[predicted_image]

    # class_centers = np.array([np.mean(np.argwhere(predicted_image == i), axis=0) for i in range(nclusters)])
    # class_centers = class_centers.astype(int)
    class_centers = gmm.means_

    for idx, center in enumerate(class_centers):
        cv2.circle(color_image, (int(center[0]), int(center[1])), 5, (0,0,0), -1)
    if show:
        ratios = [4 / 3]
        figsize = [sum(ratios) * 4.5, 4.5]
        fig, ax = plt.subplots(
        1, 1, figsize=figsize, dpi=100, gridspec_kw={"width_ratios": ratios}
        )
        plt.imshow(cv2.addWeighted(i1,0.2,color_image,0.8,0))
        plt.title("GMM Space overlayed")
        plt.axis('off')

        legend_labels = [f'Cluster {i}' for i in range(nclusters)]
        legend_colors = [colors[i] / 255 for i in range(nclusters)]
        patches = [plt.plot([], [], marker="o", ms=10, ls="", mec=None,
                            color=legend_colors[i], label="{:s}".format(legend_labels[i]))[0] for i in range(nclusters)]

        plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.15, 1))

        plt.show(block=False)
    if save:
        plt.imsave(f"{outFolder}/GMM_overlay.jpg", color_image)
    if show:
        ratios = [4 / 3]
        figsize = [sum(ratios) * 4.5, 4.5]
        fig, ax = plt.subplots(
        1, 1, figsize=figsize, dpi=100, gridspec_kw={"width_ratios": ratios}
        )
        plt.imshow(color_image)

        legend_labels = [f'Cluster {i}' for i in range(nclusters)]
        legend_colors = [colors[i] / 255 for i in range(nclusters)]
        patches = [plt.plot([], [], marker="o", ms=10, ls="", mec=None,
                            color=legend_colors[i], label="{:s}".format(legend_labels[i]))[0] for i in range(nclusters)]

        plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.15, 1))

        plt.title("GMM Space overlayed")
        plt.axis('off')
        plt.show(block=False)

    masks = [(predicted_image == i).astype(np.uint8) * 255 for i in range(nclusters)]
    if show:
        fig = plt.figure(figsize=(20,10))
        fig2 = plt.figure(figsize=(20,10))
    Transforms = []
    Transforms_DCN = []
    def add_batch_dimension(data: dict) -> dict:
        """Add batch dimension to elements in data"""
        return {
            k: np.expand_dims(v, axis=0) if isinstance(v, np.ndarray) else
            v.unsqueeze(0) if isinstance(v, torch.Tensor) else
            [v] if isinstance(v, list) else
            v
            for k, v in data.items()
        }
    if detector == "SUPER":    
        image1 = numpy_image_to_torch(i1)
        image2 = numpy_image_to_torch(i2)
        feats1 = extractor.extract(image1.to(device))
        feats2 = extractor.extract(image2.to(device))
        f1, f2 = [
            rbd(x) for x in [feats1, feats2]
        ]
        keypoints_source = f1["keypoints"].numpy()
        keypoints_target = f2["keypoints"].numpy()
        descriptors_source = f1['descriptors'].numpy()
        descriptors_target = f2['descriptors'].numpy()

        keypoints_indices_source = keypoints_source.astype(int)
        keypoints_indices_target = keypoints_target.astype(int)
        keypoints_source = f1["keypoints"]
        keypoints_target = f2["keypoints"]
        for index, mask in enumerate(masks):
            height, width = mask.shape
            
            valid_indices = (keypoints_indices_source[:, 1] < height) & (keypoints_indices_source[:, 0] < width)
            source_indices = valid_indices & (mask[keypoints_indices_source[:, 1], 
                                                                           keypoints_indices_source[:, 0]] == 255)
            source_feats = {}
            source_feats["keypoints"] = f1["keypoints"][source_indices]
            source_feats["keypoint_scores"] = f1["keypoint_scores"][source_indices]
            source_feats["descriptors"] = f1["descriptors"][source_indices]
            source_feats["image_size"] = f1["image_size"]
            source_feats = add_batch_dimension(source_feats)
            kp_temp = source_feats["keypoints"].numpy()[0]
            keypoints_source_mask = tuple(
            [cv2.KeyPoint(x=int(k[0]),y=int(k[1]),size=1) for k in kp_temp]
            )

            valid_indices = (keypoints_indices_target[:, 1] < height) & (keypoints_indices_target[:, 0] < width)
            target_indices = valid_indices & (mask[keypoints_indices_target[:, 1], 
                                                                           keypoints_indices_target[:, 0]] == 255)
            target_feats = {}
            target_feats["keypoints"] = f2["keypoints"][target_indices]
            target_feats["keypoint_scores"] = f2["keypoint_scores"][target_indices]
            target_feats["descriptors"] = f2["descriptors"][target_indices]
            target_feats["image_size"] = f2["image_size"]
            target_feats = add_batch_dimension(target_feats)
            kp_temp = target_feats["keypoints"].numpy()[0]
            keypoints_target_mask = tuple(
            [cv2.KeyPoint(x=int(k[0]),y=int(k[1]),size=1) for k in kp_temp]
            )
            

            matches01 = lightmatcher({"image0": source_feats, "image1": target_feats})
            matches01 = [
            rbd(x) for x in [matches01]
            ][0]
            match_temp = matches01["matches"].numpy().copy()
            matches = [
            cv2.DMatch(_queryIdx=match[0],_trainIdx=match[1], _distance=1) for match in match_temp
            ]
            if show:
                plt.figure(fig.number)
                plt.subplot(nclusters//2 + 1,2, index+1)
                draw_all_matches(source_image.copy(),target_image.copy(),matches,
                                keypoints_source_mask, keypoints_target_mask, False, "")
                plt.title(f"cluster {index} all matches")
                plt.axis('off')
            src_pts_cluster = np.float32([ keypoints_source_mask[m.queryIdx].pt for m in matches ]).reshape(-1,2)
            dst_pts_cluster = np.float32([ keypoints_target_mask[m.trainIdx].pt for m in matches ]).reshape(-1,2)
            print(src_pts_cluster.shape, dst_pts_cluster.shape)
            try:
                model_robust, inlierIndex = ransac((src_pts_cluster, dst_pts_cluster), EuclideanTransform, min_samples=3,
                                        residual_threshold=9, max_trials=2000)
                M = model_robust.params
                valid_centers[index] = 1
                if model_robust:
                    M = model_robust.params
                    valid_centers[index] = 1
                else:
                    M = np.eye(3)
                    inlierIndex = [False * src_pts_cluster.shape[0]]
            except ValueError as ve:
                M = np.eye(3)
                inlierIndex = [False * src_pts_cluster.shape[0]]

            if show:
                plt.figure(fig2.number)
                plt.subplot(nclusters//2 + 1,2, index+1)
                draw_inliers(source_image.copy(), target_image.copy(), 
                            keypoints_source_mask, keypoints_target_mask, inlierIndex, False, outFolder)
                plt.title(f"cluster {index} inliers")
                plt.axis('off')

            Transforms.append(M)
            Transforms_DCN.append(conv_DCN(np.linalg.inv(M)))
            logging.info(f"Cluster {index}:\n \
                        Transform: {M} \n \
                        inliers: {np.sum(inlierIndex)}")
            
        if show:
            plt.figure(fig.number)  
            plt.tight_layout()
            plt.show(block=False)
            plt.figure(fig2.number)
            plt.tight_layout()
            plt.show(block=False)
            
    else:
        for index, mask in enumerate(masks):
            keypoints_source_mask, descriptors_source_mask = detector.detectAndCompute(source_image,mask)
            keypoints_target_mask, descriptors_target_mask = detector.detectAndCompute(target_image,mask)
            matches = bf.match(descriptors_source_mask,descriptors_target_mask)
            matches = sorted(matches, key = lambda x:x.distance)
            if show:
                plt.figure(fig.number)
                plt.subplot(nclusters//2 + 1,2, index+1)
                draw_all_matches(source_image.copy(),target_image.copy(),matches,
                                keypoints_source_mask, keypoints_target_mask, False, "")
                plt.title(f"cluster {index} all matches")
                plt.axis('off')
            src_pts_cluster = np.float32([ keypoints_source_mask[m.queryIdx].pt for m in matches ]).reshape(-1,2)
            dst_pts_cluster = np.float32([ keypoints_target_mask[m.trainIdx].pt for m in matches ]).reshape(-1,2)
            try:
                model_robust, inlierIndex = ransac((src_pts_cluster, dst_pts_cluster), EuclideanTransform, min_samples=3,
                                            residual_threshold=9, max_trials=2000)
                if model_robust:
                    M = model_robust.params
                    WW = 0.2
                    valid_centers[index] = 1
                    if (M[0,2]> source_image.shape[0]*WW):
                        logging.info(f"Tx: {M[0,2]} too large")
                        valid_centers[index] = 0
                    if (M[1,2] > source_image.shape[1]*WW):
                        logging.info(f"Ty: {M[1,2]} too large")
                        valid_centers[index] = 0
                    if (np.abs(np.arccos(M[0,0])) > np.pi/8):
                        logging.info(f"Theta: {np.degrees(np.abs(np.arccos(M[0,0])))} too large")
                        valid_centers[index] = 0
                else:
                    M = np.eye(3)
                    inlierIndex = [False * src_pts_cluster.shape[0]]
            except ValueError as ve:
                M = np.eye(3)
                inlierIndex = [False * src_pts_cluster.shape[0]]
            
            if show:
                plt.figure(fig2.number)
                plt.subplot(nclusters//2 + 1,2, index+1)
                draw_inliers(source_image.copy(), target_image.copy(), 
                            keypoints_source_mask, keypoints_target_mask, inlierIndex, False, outFolder)
                plt.title(f"cluster {index} inliers")
                plt.axis('off')
            
            Transforms.append(M)
            Transforms_DCN.append(conv_DCN(np.linalg.inv(M)))

            logging.info(f"Cluster {index}:\n \
                        Transform: {M} \n \
                        Transform Par: ({M[0,2]},{M[1,2]},{np.degrees(np.abs(np.arccos(M[0,0])))}) \n \
                        inliers: {np.sum(inlierIndex)}")
        

        if show:
            plt.figure(fig.number)  
            plt.tight_layout()
            plt.show(block=False)
            plt.figure(fig2.number)
            plt.tight_layout()
            plt.show(block=False)
    
    # invalid transforms
    # neigh = NearestNeighbors(n_neighbors=2)
    # neigh.fit(class_centers)
    # _, indices = neigh.kneighbors(class_centers)
    # nearest_neighbors_indices = np.ravel(indices[:, 1:2])

    # Test
    # valid_centers[2] = 0

    # for index, center in enumerate(class_centers):
    #     if valid_centers[index] == 0:
    #         logging.info(f"Cluster {index} is replaced by {nearest_neighbors_indices[index]}")
    #         Transforms[index] = Transforms[nearest_neighbors_indices[index]].copy()
    #         Transforms_DCN[index] = Transforms_DCN[nearest_neighbors_indices[index]].copy()

    # invalid Transforms with graph whole nodes
    # fixing Transforms, Transforms_DCN
    fixed_Transforms = Transforms.copy()
    fixed_Transforms_DCN = Transforms_DCN.copy()
    if 0 in valid_centers:    
        logging.info("Fixing Invalid Transforms")
        points = np.array(class_centers)/np.array([source_image.shape[1],source_image.shape[0]])
        
        G = create_weighted_graph(points)
        visualize_entire_graph(G, (10,(10*source_image.shape[0]/source_image.shape[1])), show, save,outFolder)

        for index, center in enumerate(class_centers):
            if valid_centers[index] == 0:
                replacements = find_direct_neighbor(G, index)
                correct_replacements = []
                for i, r in enumerate(replacements):
                    if valid_centers[r[0]] != 0:
                        correct_replacements.append(replacements[i])
                logging.info(f"Cluster {index} is replaced by valid neighbors: {[i for i,r in correct_replacements]}")
                selected_Transforms = [Transforms_DCN[i] for i,r in correct_replacements]
                weights = [r for i, r in correct_replacements]
                weights = weights/sum(weights)
                fixed_Transforms_DCN[index] = blend_DCN(selected_Transforms, weights)
                fixed_Transforms[index] = fixed_Transforms_DCN[index].toTransform()
        
        for i in range(nclusters):
            logging.info(f"cluster {i} \n{fixed_Transforms[i]}")


    # Prob
    stime = datetime.now()
    logging.info(f"Starting to blend transforms")
    probs = gmm.predict_proba(positions)
    probs = probs.reshape(height, width, -1)
    all_transforms = np.empty((height, width, 3, 3))
    def blend_and_transform(probabilities):
        blended = blend_DCN(fixed_Transforms_DCN, probabilities)
        return blended.toTransform()
    vectorized_blend_transform = np.vectorize(blend_and_transform, signature='(n)->(m,m)')
    all_transforms = vectorized_blend_transform(probs)
    etime = datetime.now()
    logging.info(f"Blending Ended, time elapsed: {(etime-stime).total_seconds()}, transforms shape: {all_transforms.shape}")

    logging.info(f"Applying Transforms")
    stime = datetime.now()

    homogeneous_coords = np.stack([xs, ys, np.ones_like(xs)], axis=-1)
    transformed_coords = np.einsum('...ij,...j->...i', all_transforms, homogeneous_coords)
    mapx = transformed_coords[..., 0]
    mapy = transformed_coords[..., 1]

    # Perform the remap operation
    source_image_fineT = cv2.remap(source_image, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_CUBIC).astype(np.uint8)

    etime = datetime.now()
    logging.info(f"Transforms Applied, time elapsed: {(etime-stime).total_seconds()}")
    if show:
        ratios = [4 / 3]
        figsize = [sum(ratios) * 4.5, 4.5]
        fig, ax = plt.subplots(
        1, 1, figsize=figsize, dpi=100, gridspec_kw={"width_ratios": ratios}
        )
        plt.imshow(source_image_fineT)
        plt.title("Source Transformed (fine)")
        plt.axis('off')
        plt.show(block=False)

    #Fixing Transformed image
    if fix:
        _, mask = cv2.threshold(source_image_fineT, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        kernel = np.ones((3, 3), np.uint8)
        mask_inv_dilated = cv2.dilate(mask_inv, kernel, iterations=1)
        source_image_fineT[mask_inv_dilated == 255] = [255]
    if show:
        ratios = [4 / 3]
        figsize = [sum(ratios) * 4.5, 4.5]
        fig, ax = plt.subplots(
        1, 1, figsize=figsize, dpi=100, gridspec_kw={"width_ratios": ratios}
        )
        plt.imshow(source_image_fineT)
        plt.title("Source Transformed (fine) Fixed")
        plt.axis('off')
        plt.show(block=False)
    #Save
    plt.imsave(f"{outFolder}/SourceTransformed_fineT.jpg", source_image_fineT)
    
    if show:
        viz2d.plot_images([source_image_fineT,target_image])
        # plt.imshow(np.hstack((source_image_fineT,target_image)))
        plt.title("Source Transformed (fine) + Target Image")
        plt.axis('off')
        plt.show(block=False)
    if show:
        ratios = [4 / 3]
        figsize = [sum(ratios) * 4.5, 4.5]
        fig, ax = plt.subplots(
        1, 1, figsize=figsize, dpi=100, gridspec_kw={"width_ratios": ratios}
        )
        plt.imshow(overlay(source_image_fineT, target_image))
        plt.title(f"Overlay Source Transformed (fine) , Target Image, ssd={ssd(source_image_fineT,target_image)}")
        plt.axis('off')
        plt.show(block=True)
    else:
        plt.close("all")
    if save:
        plt.imsave(f"{outFolder}/source_target_overlay.jpg", overlay(source_image_fineT, target_image))
    
    logging.info(f"SSD Fine registered image pairs: {ssd(source_image_fineT,target_image)}")

    logging.info(f"Fine registration Done")
    return nclusters

def createFinalOutputs(sFolder, dFolder, nFiles):

    os.makedirs(dFolder, exist_ok=True)

    for folder_name in range(nFiles):
        current_folder = os.path.join(sFolder, str(folder_name))
        source_file = os.path.join(current_folder, 'SourceTransformed_fineT.jpg')
        new_file_name = f'{folder_name}.jpg'
        destination_file = os.path.join(dFolder, new_file_name)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, destination_file)

