import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.transform import EuclideanTransform
from skimage.measure import ransac
from datetime import datetime
from sklearn.mixture import GaussianMixture
import shutil
from DCN import DCN


def overlay(im1,im2, alpha=0.5):
    im_out = np.zeros_like(im1)
    im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    im_out[:,:,0] = im1
    im_out[:,:,1] = im2
    return im_out


def findTransform(kp1,kp2):
    if type(kp1[0]) == cv2.KeyPoint:
        t_array = []
        for k in kp1:
            t_array.append([k.pt[0], k.pt[1]])
        src_pts = np.float32(t_array).reshape(-1,2)
        t_array = []
        for k in kp2:
            t_array.append([k.pt[0], k.pt[1]])
        dst_pts = np.float32(t_array).reshape(-1,2)
    else:
        src_pts = kp1
        dst_pts = kp2
    if len(src_pts)>2:
        model_robust, inlierIndex = ransac((src_pts, dst_pts), EuclideanTransform, min_samples=2,
                                residual_threshold=9, max_trials=2000)
        M = model_robust.params
        # M, inlierIndex = cv2.estimateAffinePartial2D(src_pts, dst_pts, method = cv2.RANSAC, ransacReprojThreshold=9, confidence = 0.99, refineIters=0)
        # M, inlierIndex = cv2.findHomography(srcPoints=src_pts, dstPoints=dst_pts, method = cv2.RANSAC, ransacReprojThreshold=1, confidence = 0.99)
        return M, inlierIndex, True
    else:
        return None, None, False
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

def regPair(source, target, outFolder, colorScale, detector, threshold, matcher, nclusters, save, fix, show, logging):
    logging.info(f"Registering{source} and {target}, saving {save}, output {outFolder}")
    # Loading Images
    source_image = loadImage(source, logging)
    target_image = loadImage(target, logging)
    source_image, target_image = changeImageColor(source_image, target_image, colorScale, logging)
    logging.info(f"images shape: {source_image.shape}, {target_image.shape}")
    if show:
        plt.figure(figsize=(20,10))
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
    else:
        logging.error(f"Error: Undefined detector {detector}")
        raise ValueError(f"Detector must be SIFT or AKAZE, but got {detector}") 
    
    # Feature Extarction
    logging.info(f"Starting  Global registration")
    stime = datetime.now()
    i1 = source_image.copy()
    i2 = target_image.copy()
    logging.info(f"SSD original image pairs: {ssd(i1,i2)}")
    keypoints_source, descriptors_source = detector.detectAndCompute(source_image,None)
    keypoints_target, descriptors_target = detector.detectAndCompute(target_image,None)
    keypoints_image = np.hstack((cv2.drawKeypoints(i1,keypoints_source,i1),
                                cv2.drawKeypoints(i2,keypoints_target,i2)))
    if show:
        plt.figure(figsize=(20,10))
        plt.imshow(keypoints_image)
        plt.title("All Keypoints in Source and Target Images")
        plt.axis('off')
        plt.show(block=False)
    if save:
        plt.imsave(f"{outFolder}/keypoints.jpg", keypoints_image)

    # Matching
    if matcher =="L2":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        logging.info(f"Matcher: {matcher}")
    elif matcher == "Hamming":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        logging.info(f"Matcher: {matcher}")
    else:
        logging.error(f"Error: Undefined mathcer {matcher}")
        raise ValueError(f"Matcher must be L2 or Hamming, but got {matcher}")
    matches = bf.match(descriptors_source,descriptors_target)
    matches = sorted(matches, key = lambda x:x.distance)
    if show:
        plt.figure(figsize=(20, 10))
        draw_all_matches(source_image, target_image, matches, keypoints_source, keypoints_target, save, outFolder)
        plt.title("All Matches between Source and Target Images")
        plt.axis('off')
        plt.show(block=False)

    logging.info(f"Number of Matched Keypoints: {len(matches)}")
    src_pts = np.float32([ keypoints_source[m.queryIdx].pt for m in matches ]).reshape(-1,2) #i1
    dst_pts = np.float32([ keypoints_target[m.trainIdx].pt for m in matches ]).reshape(-1,2) #i2
    # Transform
    logging.info(f"Finding Euclidean Transform")
    model_robust, inlierIndex = ransac((src_pts, dst_pts), EuclideanTransform, min_samples=2,
                                    residual_threshold=9, max_trials=2000)
    M = model_robust.params
    logging.info(f"Euclidean Transform found \n {M}\n inliers:{np.sum(inlierIndex)}")
    if show:

        plt.figure(figsize=(20, 10))
        draw_inliers(source_image.copy(), target_image.copy(), 
                    keypoints_source, keypoints_target, inlierIndex, save, outFolder)
        plt.title("Source and Target Images with Inlier Keypoints and Connecting Lines")
        plt.axis('off')

    source_transformed = cv2.warpPerspective(source_image, M, (i2.shape[1],i2.shape[0]),flags=cv2.INTER_LINEAR)
    if show:
        plt.figure(figsize=(20,10))
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
            plt.figure(figsize=(20,10))
            plt.imshow(source_transformed)
            plt.title("Source Transformed Fixed")
            plt.axis('off')
            plt.show(block=False)
    #Save
    plt.imsave(f"{outFolder}/SourceTransformed.jpg", source_transformed)
    
    etime = datetime.now()
    if show:
        plt.figure(figsize=(20,10))
        plt.imshow(np.hstack((source_transformed,target_image)))
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
        plt.figure(figsize=(20,10))
        plt.imshow(overlay(source_image, target_image))
        plt.title(f"Overlay Source Transformed , Target Image, ssd={ssd(source_image,target_image)}")
        plt.axis('off')
    i1 = source_image.copy()
    i2 = target_image.copy()
    keypoints_source, descriptors_source = detector.detectAndCompute(source_image,None)
    keypoints_target, descriptors_target = detector.detectAndCompute(target_image,None)
    
    logging.info(f"Number of Clusters: {nclusters}")
    logging.info("Fitting GMM to keypoints")
    gmm = GaussianMixture(n_components=nclusters, random_state=0,covariance_type='full')
    gmm.fit(np.float32([ k.pt for k in keypoints_source ]).reshape(-1,2))
    logging.info("GMM fitting done")
    logging.info(f"GMM means: {gmm.means_}")
    
    height, width = i1.shape[:2]
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    positions = np.column_stack([xs.flatten(), ys.flatten()])
    colors = np.random.randint(100, 200, size=(nclusters, 3), dtype=np.uint8)
    pred = gmm.predict(positions)
    predicted_image = pred.reshape(height, width)
    color_image = colors[predicted_image]

    class_centers = np.array([np.mean(np.argwhere(predicted_image == i), axis=0) for i in range(nclusters)])
    class_centers = class_centers.astype(int)
    for idx, center in enumerate(class_centers):
        cv2.circle(color_image, (center[1], center[0]), 10, (0,0,0), -1)
    if show:
        plt.figure(figsize=(20,10))
        plt.imshow(cv2.addWeighted(i1,0.2,color_image,0.8,0))
        plt.title("GMM Space overlayed")
        plt.axis('off')
        plt.show(block=False)
    if save:
        plt.imsave(f"{outFolder}/GMM_overlay.jpg", color_image)
    if show:
        plt.figure(figsize=(20,10))
        plt.imshow(color_image)
        plt.title("GMM Space overlayed")
        plt.axis('off')
        plt.show(block=False)

    masks = [(predicted_image == i).astype(np.uint8) * 255 for i in range(nclusters)]
    if show:
        fig = plt.figure(figsize=(20,10))
        fig2 = plt.figure(figsize=(20,10))
    Transforms = []
    Transforms_DCN = []
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
        model_robust, inlierIndex = ransac((src_pts_cluster, dst_pts_cluster), EuclideanTransform, min_samples=2,
                                    residual_threshold=9, max_trials=2000)
        if show:
            plt.figure(fig2.number)
            plt.subplot(nclusters//2 + 1,2, index+1)
            draw_inliers(source_image.copy(), target_image.copy(), 
                        keypoints_source_mask, keypoints_target_mask, inlierIndex, False, outFolder)
            plt.title(f"cluster {index} inliers")
            plt.axis('off')
        M = model_robust.params
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

    # Prob
    stime = datetime.now()
    logging.info(f"Starting to blend transforms")
    probs = gmm.predict_proba(positions)
    probs = probs.reshape(height, width, -1)
    all_transforms = np.empty((height, width, 3, 3))
    def blend_and_transform(probabilities):
        blended = blend_DCN(Transforms_DCN, probabilities)
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
        plt.figure(figsize=(20,10))
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
        plt.figure(figsize=(20,10))
        plt.imshow(source_image_fineT)
        plt.title("Source Transformed (fine) Fixed")
        plt.axis('off')
        plt.show(block=False)
    #Save
    plt.imsave(f"{outFolder}/SourceTransformed_fineT.jpg", source_image_fineT)
    
    if show:
        plt.figure(figsize=(20,10))
        plt.imshow(np.hstack((source_image_fineT,target_image)))
        plt.title("Source Transformed (fine) + Target Image")
        plt.axis('off')
        plt.show(block=False)
    if show:
        plt.figure(figsize=(20,10))
        plt.imshow(overlay(source_image_fineT, target_image))
        plt.title(f"Overlay Source Transformed (fine) , Target Image, ssd={ssd(source_image_fineT,target_image)}")
        plt.axis('off')
        plt.show(block=True)
    if save:
        plt.imsave(f"{outFolder}/source_target_overlay.jpg", overlay(source_image_fineT, target_image))
    
    logging.info(f"SSD Fine registered image pairs: {ssd(source_image_fineT,target_image)}")

    logging.info(f"Fine registration Done")

def createFinalOutputs(sFolder, dFolder, nFiles):

    os.makedirs(dFolder, exist_ok=True)

    for folder_name in range(nFiles):
        current_folder = os.path.join(sFolder, str(folder_name))
        source_file = os.path.join(current_folder, 'SourceTransformed_fineT.jpg')
        new_file_name = f'{folder_name + 1}.jpg'
        destination_file = os.path.join(dFolder, new_file_name)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, destination_file)