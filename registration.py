from utils import *
import numpy as np
import cv2
import sys
from PlotViewer import PlotViewer

from skimage.transform import EuclideanTransform
from skimage.measure import ransac
from datetime import datetime
from sklearn.mixture import GaussianMixture

from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import rbd, numpy_image_to_torch, read_image
import torch

def regPair(source, target,colorScale, outFolder, detector, threshold, maxkps, globalreg, numtry, finereg,  
            valid_weight, matcher, nclusters, max_clusers, save, fix, show, logger):

    """
    Load, Register and save a pair of images with Matched Features extracted from images

    Parameters:
        source: source image,
        target: target image,
        colorScale: images colorcodes,
        outFolder: saving location,
        detector: feature detection algorithm,
        threshold: feature extraction threshold,
        maxkps: deep feature maximum number of features,
        globalreg: global registration,
        numtry: number of tries for global transformation,
        finereg: fine registration,
        matcher: feature matching algorithm,
        nclusters: number of clusters for gaussian mixture,
        max_clusters: maximum number of clusters,
        save: allowing saving files,
        fix: allowing fixing background,
        show: showing outputs,
        logger: Custom logging,
        saveName: file name for saving,

    Returns:
    Number of Used Clusters.
    """

    viewer = None
    if show:
        viewer = PlotViewer()
    random_seed = 3
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    full_logger = logger[0]
    short_logger = logger[1]
    full_logger.message(f"Image Directories:\n"
                        f"Source Image: {source}:\n"
                        f"Target Image: {target}\n"
                        f"Saving in {outFolder}"
    )
    short_logger.message(f"Source Image: {source}:\n"
                         f"Target Image: {target}\n"
                         f"Saving in {outFolder}"
    )

    # Loading Images
    try:
        source_image = changeImageColor(loadImage(source), colorScale)
        target_image = changeImageColor(loadImage(target), colorScale)
    except OSError as e:
        full_logger.log_error(f"Error loading Files:\n"
                         f"{e}")
        sys.exit(1)
    except ValueError as e:
        full_logger.log_error(f"Error Converting Images:\n"
                         f"{e}")
        sys.exit(1)
    else:
        full_logger.message(f"Images Successfuly Loaded and Converted to {colorScale}\n"
                       f"Source Shape: {source_image.shape}\n"
                       f"Target Shape: {target_image.shape}")
        short_logger.message(f"Source Shape: {source_image.shape}\n"
                       f"Target Shape: {target_image.shape}")
   
    # Initial Overlay
    
    fig, ax = plot_overlay(source=source_image,
                    target=target_image,
                    save=save,
                    saveAddress=outFolder,
                    fileName="InitialOverlay")
    if show:    
        viewer.add_figure(fig)

    # Initializing Detector
    if detector == "SIFT":
        full_logger.message(f"Initializing  {detector} with {threshold}")
        short_logger.message(f"Initializing  {detector} with {threshold}")
        detector = cv2.SIFT_create(contrastThreshold=threshold)
    elif detector == "AKAZE":
        full_logger.message(f"Initializing  {detector} with {threshold}")
        short_logger.message(f"Initializing  {detector} with {threshold}")
        detector = cv2.AKAZE_create(threshold=threshold)
    elif detector == "SUPER":
        full_logger.message(f"Initializing  {detector} with {maxkps}")
        short_logger.message(f"Initializing  {detector} with {maxkps}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor = SuperPoint(max_num_keypoints=maxkps).eval().to(device)
    else:
        full_logger.log_error(f"Error: Undefined detector {detector}")
        raise ValueError(f"Undefined detector {detector}")
    
    # Global Registration
    if globalreg:

        # Feature Extarction
        full_logger.message(f"Starting  Global registration")
        stime = datetime.now()
        i1 = source_image.copy()
        i2 = target_image.copy()
        full_logger.message(f"SSD original image pairs: {ssd(i1,i2)}")
        short_logger.message(f"SSD original image pairs: {ssd(i1,i2)}")
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
        full_logger.message(f"Number of Keypoints\n"
                            f"Source Keypoints:{len(keypoints_source)}\n"
                            f"Target Keypoints:{len(keypoints_target)}")
        short_logger.message(f"Source Keypoints:{len(keypoints_source)}\n"
                            f"Target Keypoints:{len(keypoints_target)}")
        src_pts = np.uint([ [k.pt[0],k.pt[1]] for k in keypoints_source ]).reshape(-1,2) #i1
        dst_pts = np.uint([ [k.pt[0],k.pt[1]] for k in keypoints_target ]).reshape(-1,2) #i2
        fig, ax = plot_keypoints(images=(i1,i2), 
                                    keypoints=(src_pts, dst_pts), 
                                    save=save, 
                                    saveAddress=outFolder,
                                    fileName="GlobalKeypoints")
        if show:    
            viewer.add_figure(fig)

        # Matching
        if matcher in ["L2", "Hamming"]:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            full_logger.message(f"Matcher: {matcher}")
            short_logger.message(f"Matcher: {matcher}")
            matches = bf.match(descriptors_source,descriptors_target)
        elif matcher == "Light":
            lightmatcher = LightGlue(features="superpoint").eval().to(device)
            full_logger.message(f"Matcher: {matcher}")
            short_logger.message(f"Matcher: {matcher}")
            matches01 = lightmatcher({"image0": feats1, "image1": feats2})
            matches01 = [
                rbd(x) for x in [matches01]
            ][0]
            match_temp = matches01["matches"].numpy().copy()
            matches = [
                cv2.DMatch(_queryIdx=match[0],_trainIdx=match[1], _distance=1) for match in match_temp
            ]
        else:
            full_logger.log_error(f"Error: Undefined mathcer {matcher}")
            raise ValueError(f"Undefined mathcer {matcher}")   
        full_logger.message(f"Matched Keypoints: {len(matches)}")
        short_logger.message(f"Matched Keypoints: {len(matches)}")
        src_pts = np.float32([ keypoints_source[m.queryIdx].pt for m in matches ]).reshape(-1,2) #i1 
        dst_pts = np.float32([ keypoints_target[m.trainIdx].pt for m in matches ]).reshape(-1,2) #i2
        fig, ax = plot_matches(images=(i1,i2),
                                keypoints=(src_pts, dst_pts), 
                                save=save, 
                                saveAddress=outFolder,
                                color='lime',
                                fileName="GlobalMatches")
        if show:    
            viewer.add_figure(fig)

        # Global Transform
        full_logger.message(f"Finding Global Transform")
        glob_av = False
        for trynumber in range(numtry):
            model_robust, inlierIndex = ransac((src_pts, dst_pts), EuclideanTransform, min_samples=3,
                                            residual_threshold=9, max_trials=2000)
            full_logger.message(f"Try Number {trynumber}, Inliers: {np.sum(inlierIndex)}")
            if np.sum(inlierIndex) > 50:
                glob_av = True
                break
        else:
            full_logger.log_error("Global Registration Failed, Moving to Fine")
            short_logger.log_error("Global Registration Failed, Moving to fine")
            source_image = source_image.copy()
            target_image = target_image.copy()
            # raise ValueError("Can not find Global Registration Transform")
        
        if glob_av:
            M = model_robust.params
            full_logger.log_transform_info(-1, M, inlierIndex)
            short_logger.log_transform_info(-1, M, inlierIndex)
            src_pts = np.float32([kp for kp, inlier in zip(src_pts, inlierIndex) if inlier]).reshape(-1,2) #i1 
            dst_pts = np.float32([kp for kp, inlier in zip(dst_pts, inlierIndex) if inlier]).reshape(-1,2) #i1
            fig, ax = plot_matches(images=(i1,i2),
                                    keypoints=(src_pts, dst_pts), 
                                    save=save, 
                                    saveAddress=outFolder,
                                    color='tomato',
                                    fileName="GlobalInliers")
            if show:    
                viewer.add_figure(fig)

            # Applying Global Transform
            source_transformed = cv2.warpPerspective(source_image, M, (i2.shape[1],i2.shape[0]),flags=cv2.INTER_LINEAR)
            if fix:
                _, mask = cv2.threshold(source_transformed, 1, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                kernel = np.ones((3, 3), np.uint8)
                mask_inv_dilated = cv2.dilate(mask_inv, kernel, iterations=1)
                source_transformed[mask_inv_dilated == 255] = [255]
            fig, ax = plot_image(img=source_transformed, 
                                    save=save,
                                    saveAddress=outFolder,
                                    fileName="SourceTransformed")
            if show:    
                viewer.add_figure(fig)
            fig, ax = plot_3image(images=(source_image,source_transformed, target_image),
                                    save=False,
                                    saveAddress=outFolder,
                                    fileName="SourceTarget")
            if show:    
                viewer.add_figure(fig)
            fig, ax = plot_overlay(source=source_transformed,
                                    target=target_image,
                                    save=False,
                                    saveAddress=outFolder,
                                    fileName="GlobalOverlay")
            if show:    
                viewer.add_figure(fig)
            etime = datetime.now()

            full_logger.message(f"Global registration Done")
            full_logger.message(f"SSD Globally registered image pairs: {ssd(source_transformed,target_image)}\n"
                            f"elapsed time: {(etime-stime).total_seconds()}")
            short_logger.message(f"SSD Globally registered image pairs: {ssd(source_transformed,target_image)}\n"
                            f"elapsed time: {(etime-stime).total_seconds()}")
            
            source_image = source_transformed.copy()
            target_image = target_image.copy()

    # Fine Registration
    if finereg:
        # feature Extraction
        full_logger.message(f"Starting Fine registration")
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
        data = np.float32([ k.pt for k in keypoints_source ]).reshape(-1,2)
        if nclusters=='auto':
            full_logger.message(f"Finding the Optimum number of clusters")
            short_logger.message(f"Finding the Optimum number of clusters")
            n_components = np.arange(2, max_clusers, 1)
            bics = []
            # sils = []
            for n in n_components:
                gmm = GaussianMixture(n_components=n, weights_init=np.ravel(np.ones([n,1])/n),
                              random_state=0, covariance_type='full', n_init=10, max_iter=500)
                gmm.fit(data)
                bics.append(gmm.bic(data))
                # sils.append(silhouette_score(data,gmm.fit_predict(data)))
            max_curvature_idx = np.argmax(np.abs(grad2(bics)))
            full_logger.message(f"BIC Values:{bics}\n"
                                f"max curvature: {max_curvature_idx}")
            fig, ax = plot_bics(n_vector=n_components,
                                    bic_vector=bics,
                                    bic_grad_vector=grad2(bics),
                                    index=max_curvature_idx,
                                    save=save,
                                    saveAddress=outFolder,
                                    fileName="BayesianInformationCriterion")
            if show:    
                viewer.add_figure(fig)
            nclusters = n_components[max_curvature_idx+1]
            full_logger.message(f"Optimal Cluster Numbers: {nclusters}")
            short_logger.message(f"Optimal Cluster Numbers: {nclusters}")

        full_logger.message(f"Fitting GMM to {nclusters} Clusters")
        short_logger.message(f"clusters: {nclusters}")
        nclusters = int(nclusters)
        gmm = GaussianMixture(n_components=nclusters,  weights_init=np.ravel(np.ones([nclusters,1])/nclusters),
                              random_state=0, covariance_type='full', n_init=10, max_iter=500)
        gmm.fit(data)
        full_logger.message(f"GMM fitting done\n"
                            f"GMM means: {gmm.means_}")
        short_logger.message(f"GMM means: {gmm.means_}")


        gmm_map, masks= plot_gmm_map(img=i1,
                                      nclusters=nclusters,
                                      gmm=gmm,
                                      img_sz=i1.shape)
        gmm_map_final = gmm_map.copy()
        Transforms = []
        Transforms_DCN = []
        valid_centers = np.zeros([nclusters,1])
        cluster_centers = gmm.means_
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

                src_pts_cluster = np.float32([ keypoints_source_mask[m.queryIdx].pt for m in matches ]).reshape(-1,2)
                dst_pts_cluster = np.float32([ keypoints_target_mask[m.trainIdx].pt for m in matches ]).reshape(-1,2)
                try:
                    model_robust, inlierIndex = ransac((src_pts_cluster, dst_pts_cluster), EuclideanTransform, min_samples=3,
                                            residual_threshold=9, max_trials=2000)
                    if model_robust and not np.any(np.isnan(model_robust.params)):
                        M = model_robust.params
                        full_logger.log_transform_info(index= index,
                                               M = M,
                                               inlierIndex=inlierIndex)
                        short_logger.log_transform_info(index= index,
                                               M = M,
                                               inlierIndex=inlierIndex)
                        valid_centers[index] = 1

                        # check validity
                        if (M[0,2]> source_image.shape[0]*valid_weight):
                            full_logger.message(f"Tx: {M[0,2]} > {source_image.shape[0]*valid_weight} too large")
                            valid_centers[index] = 0
                        if (M[1,2] > source_image.shape[1]*valid_weight):
                            full_logger.message(f"Ty: {M[1,2]} > {source_image.shape[1]*valid_weight} too large")
                            valid_centers[index] = 0
                        if (np.abs(np.arccos(M[0,0])) > np.pi/8):
                            full_logger.message(f"Theta: {np.degrees(np.abs(np.arccos(M[0,0])))} > {np.degrees(np.pi/8)} too large")
                            valid_centers[index] = 0
                    else:
                        full_logger.log_error(f"cluster {index} Transform not found!")
                        model_robust.params
                        inlierIndex = [False * src_pts_cluster.shape[0]]
                except ValueError as ve:
                    full_logger.log_error(f"cluster {index} Transform not found!")
                    model_robust.params
                    inlierIndex = [False * src_pts_cluster.shape[0]]

                Transforms.append(M)
                Transforms_DCN.append(conv_DCN(np.linalg.inv(M)))
                x,y = int(cluster_centers[index][0]), int(cluster_centers[index][1])
                if valid_centers[index]==1:
                    gmm_map = annotate_transform(image=gmm_map,
                                                 Transform=[index, x, y, M[0,2], M[1,2], np.degrees((np.arccos(M[0,0])))],
                                                 color=(255,255,255))
                else:
                    gmm_map = annotate_transform(image=gmm_map,
                                                 Transform=[index, x, y, M[0,2], M[1,2], np.degrees((np.arccos(M[0,0])))],
                                                 color=(0,0,0))
        else:
            for index, mask in enumerate(masks):
                keypoints_source_mask, descriptors_source_mask = detector.detectAndCompute(source_image,mask)
                keypoints_target_mask, descriptors_target_mask = detector.detectAndCompute(target_image,mask)
                matches = bf.match(descriptors_source_mask,descriptors_target_mask)
                matches = sorted(matches, key = lambda x:x.distance)
                src_pts_cluster = np.float32([ keypoints_source_mask[m.queryIdx].pt for m in matches ]).reshape(-1,2)
                dst_pts_cluster = np.float32([ keypoints_target_mask[m.trainIdx].pt for m in matches ]).reshape(-1,2)
                try:
                    model_robust, inlierIndex = ransac((src_pts_cluster, dst_pts_cluster), EuclideanTransform, min_samples=3,
                                                residual_threshold=9, max_trials=2000)
                    if model_robust and not np.any(np.isnan(model_robust.params)):
                        M = model_robust.params
                        full_logger.log_transform_info(index= index,
                                               M = M,
                                               inlierIndex=inlierIndex)
                        short_logger.log_transform_info(index= index,
                                               M = M,
                                               inlierIndex=inlierIndex)
                        valid_centers[index] = 1

                        # check validity
                        if (M[0,2]> source_image.shape[0]*valid_weight):
                            full_logger.message(f"Tx: {M[0,2]} > {source_image.shape[0]*valid_weight} too large")
                            valid_centers[index] = 0
                        if (M[1,2] > source_image.shape[1]*valid_weight):
                            full_logger.message(f"Ty: {M[1,2]} > {source_image.shape[1]*valid_weight} too large")
                            valid_centers[index] = 0
                        if (np.abs(np.arccos(M[0,0])) > np.pi/8):
                            full_logger.message(f"Theta: {np.degrees(np.abs(np.arccos(M[0,0])))} > {np.degrees(np.pi/8)} too large")
                            valid_centers[index] = 0
                    else:
                        full_logger.log_error(f"cluster {index} Transform not found!")
                        M = model_robust.params
                        inlierIndex = [False * src_pts_cluster.shape[0]]
                except ValueError as ve:
                    full_logger.log_error(f"cluster {index} Transform not found!")
                    M = model_robust.params
                    inlierIndex = [False * src_pts_cluster.shape[0]]
                
                Transforms.append(M)
                Transforms_DCN.append(conv_DCN(np.linalg.inv(M)))
                x,y = int(cluster_centers[index][0]), int(cluster_centers[index][1])
                if valid_centers[index]==1:
                    gmm_map = annotate_transform(image=gmm_map,
                                                 Transform=[index, x, y, M[0,2], M[1,2], np.degrees((np.arccos(M[0,0])))],
                                                 color=(255,255,255))
                else:
                    gmm_map = annotate_transform(image=gmm_map,
                                                 Transform=[index, x, y, M[0,2], M[1,2], np.degrees((np.arccos(M[0,0])))],
                                                 color=(0,0,0))
        fig, ax = plot_image(img=gmm_map,
                                save=save,
                                saveAddress=outFolder,
                                fileName="GMMOverlayAll")
        if show:
            viewer.add_figure(fig)

        # fixing invalid transforms
        finalTransforms = Transforms.copy()
        finalTransforms_DCN = Transforms_DCN.copy()
        if 0 not in valid_centers:
            full_logger.message("All Transforms Valid")
            short_logger.message("All Transforms Valid")
        else:
            full_logger.message("Fixing Invalid Transforms")
            short_logger.message("Fixing Invalid Transforms")
            points = np.array(gmm.means_)/np.array([source_image.shape[1],source_image.shape[0]])
            G = create_weighted_graph(points)
            fig, ax = plot_graph(G=G,
                                    save=save,
                                    saveAddress=outFolder,
                                    fileName="ClusterGraph")
            if show:
                viewer.add_figure(fig)
            for index, center in enumerate(gmm.means_):
                if valid_centers[index] == 0:
                    replacements = find_direct_neighbor(G, index)
                    correct_replacements = []
                    for i, r in enumerate(replacements):
                        if valid_centers[r[0]] != 0:
                            correct_replacements.append(replacements[i])
                    full_logger.message(f"Cluster {index} is replaced by valid neighbors: {[i for i,r in correct_replacements]}")
                    selected_Transforms = [Transforms_DCN[i] for i,r in correct_replacements]
                    weights = [r for i, r in correct_replacements]
                    weights = weights/sum(weights)
                    finalTransforms_DCN[index] = blend_DCN(selected_Transforms, weights)
                    finalTransforms[index] = finalTransforms_DCN[index].toTransform()
                M = finalTransforms[index]
                x,y = int(cluster_centers[index][0]), int(cluster_centers[index][1])
                gmm_map_final = annotate_transform(image=gmm_map_final,
                                            Transform=[index, x, y, M[0,2], M[1,2], np.degrees((np.arccos(M[0,0])))],
                                                color=(255,255,255))
                full_logger.log_transform_info(index= index,
                                               M = M,
                                               inlierIndex=[0])
                short_logger.log_transform_info(index= index,
                                               M = M,
                                               inlierIndex=[0])
            fig, ax = plot_image(img=gmm_map_final,
                                save=save,
                                saveAddress=outFolder,
                                fileName="GMMOverlayFixed")
            if show:
                viewer.add_figure(fig)
        # Blending Trasnforms
        stime = datetime.now()
        full_logger. message(f"Starting to blend transforms")
        height, width = source_image.shape[:2]
        xs, ys = np.meshgrid(np.arange(width), np.arange(height))
        positions = np.column_stack([xs.flatten(), ys.flatten()])
        probs = gmm.predict_proba(positions)
        probs = probs.reshape(height, width, -1)
        all_transforms = np.empty((height, width, 3, 3))
        def blend_and_transform(probabilities):
            blended = blend_DCN(finalTransforms_DCN, probabilities)
            return blended.toTransform()
        vectorized_blend_transform = np.vectorize(blend_and_transform, signature='(n)->(m,m)')
        all_transforms = vectorized_blend_transform(probs)
        etime = datetime.now()
        full_logger.message(f"Blending Ended, time elapsed: {(etime-stime).total_seconds()}, transforms shape: {all_transforms.shape}")
        short_logger.message(f"Blending Transforms: {(etime-stime).total_seconds()}")

        full_logger.message(f"Applying Transforms")
        stime = datetime.now()
        homogeneous_coords = np.stack([xs, ys, np.ones_like(xs)], axis=-1)
        transformed_coords = np.einsum('...ij,...j->...i', all_transforms, homogeneous_coords)
        mapx = transformed_coords[..., 0]
        mapy = transformed_coords[..., 1]
        source_image_fine = cv2.remap(source_image, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_CUBIC).astype(np.uint8)
        if fix:
            _, mask = cv2.threshold(source_image_fine, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            kernel = np.ones((3, 3), np.uint8)
            mask_inv_dilated = cv2.dilate(mask_inv, kernel, iterations=1)
            source_image_fine[mask_inv_dilated == 255] = [255]
        
        fig, ax = plot_image(img=source_image_fine,
                                save=save,
                                saveAddress=outFolder,
                                fileName="FinalImage")
        if show:
            viewer.add_figure(fig)

        fig, ax = plot_3image(images=(source_image, source_image_fine, target_image),
                                save=save,
                                saveAddress=outFolder,
                                fileName="FinalTarget")
        if show:    
            viewer.add_figure(fig)

        fig, ax = plot_overlay(source=source_image_fine,
                                target=target_image,
                                save=save,
                                saveAddress=outFolder,
                                fileName="FinalOverlay")
        if show:    
            viewer.add_figure(fig)

        etime = datetime.now()
        full_logger.message(f"SSD Fine registered image pairs: {ssd(source_image_fine,target_image)}\n"
                         f"elapsed time: {(etime-stime).total_seconds()}")
        short_logger.message(f"SSD Fine registered image pairs: {ssd(source_image_fine,target_image)}\n"
                          f"elapsed time: {(etime-stime).total_seconds()}")
    return nclusters, viewer, source_image_fine