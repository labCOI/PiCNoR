import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import os


from itertools import combinations
import networkx as nx

import shutil
from DCN import DCN




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
    if len (img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1.copy()
        img2_gray = img2.copy()
    composite[..., 0] = img1_gray  # Red channel
    composite[..., 1] = img2_gray  # Green channel
    composite[..., 2] = img2_gray  # Blue channel
    return composite

def plot_image(img, save, saveAddress, fileName, patches=None):
    """
    show and save an image

    Parameters:
        img: image,
        save: boolean to allow saving,
        saveAddress: saving directory,
        fileName: file name for saving

    Returns:
    Figure & Axes
    """
    ratios = [4 / 3]
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, ax = plt.subplots(
    1, 1, figsize=figsize, dpi=100, gridspec_kw={"width_ratios": ratios}
    )
    ax.imshow(img)
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    ax.set_axis_off()
    ax.set_title(fileName)
    if patches:
        fig.legend(handles=patches[1], labels=patches[0], loc='upper right', bbox_to_anchor=(1.15, 1))
    if save:
        plt.imsave(f"{saveAddress}/{fileName}.png", img)
    return fig, ax

def plot_overlay(source, target, save, saveAddress, fileName):
    """
    show and save the overlay of source and target images

    Parameters:
        source: source image,
        target: target image,
        save: boolean to allow saving,
        saveAddress: saving directory,
        fileName: file name for saving

    Returns:
    Figure & Axes
    """
    ratios = [4 / 3]
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, ax = plt.subplots(
    1, 1, figsize=figsize, dpi=100, gridspec_kw={"width_ratios": ratios}
    )
    ov = overlay(source, target)
    ax.imshow(ov)
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    ax.set_axis_off()
    ax.set_title(fileName)  
    if save:
        plt.imsave(f"{saveAddress}/{fileName}.png", ov)
    return fig, ax

def plot_keypoints(images, keypoints, save, saveAddress, fileName, color='lime'):
    """
    show and save the images with their features

    Parameters:
        images: a tuple of two images,
        keypoints: a tuple of two sets of feature points
        save: boolean to allow saving,
        saveAddress: saving directory,
        fileName: file name for saving

    Returns:
    Figure & Axes
    """
    ratios = [4 / 3]*2
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, ax = plt.subplots(
    1, 2, figsize=figsize, dpi=100, gridspec_kw={"width_ratios": ratios}
    )
    # source image
    ax[0].imshow(images[0])
    ax[0].get_yaxis().set_ticks([])
    ax[0].get_xaxis().set_ticks([])
    ax[0].set_axis_off()
    ax[0].set_title("source")    

    #source keypoints
    ax[0].scatter(keypoints[0][:, 0], keypoints[0][:, 1], c=color, s=4, linewidths=0, alpha=0.8)
    
    # target image
    ax[1].imshow(images[1])
    ax[1].get_yaxis().set_ticks([])
    ax[1].get_xaxis().set_ticks([])
    ax[1].set_axis_off()
    ax[1].set_title("target")

    # target keypoints
    ax[1].scatter(keypoints[1][:, 0], keypoints[1][:, 1], c=color, s=4, linewidths=0, alpha=0.8)

    fig.tight_layout()
    fig.suptitle(fileName)

    if save:
        plt.savefig(f"{saveAddress}/{fileName}.png")
    return fig, ax

def plot_matches(images, keypoints, save, saveAddress, fileName, color='tomato'):
    """
    show and save the images with their matched features

    Parameters:
        images: a tuple of two images,
        keypoints: a tuple of two sets of feature points
        save: boolean to allow saving,
        saveAddress: saving directory,
        fileName: file name for saving

    Returns:
    Figure & Axes
    """
    ratios = [4 / 3]*2
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, ax = plt.subplots(
    1, 2, figsize=figsize, dpi=100, gridspec_kw={"width_ratios": ratios}
    )

    # source image
    ax[0].imshow(images[0])
    ax[0].get_yaxis().set_ticks([])
    ax[0].get_xaxis().set_ticks([])
    ax[0].set_axis_off()
    ax[0].set_title("source")    

    # source keypoints
    ax[0].scatter(keypoints[0][:, 0], keypoints[0][:, 1], c=color, s=4, linewidths=0, alpha=0.8)
    
    # target image
    ax[1].imshow(images[1])
    ax[1].get_yaxis().set_ticks([])
    ax[1].get_xaxis().set_ticks([])
    ax[1].set_axis_off()
    ax[1].set_title("target")

    # target keypoints
    ax[1].scatter(keypoints[1][:, 0], keypoints[1][:, 1], c=color, s=4, linewidths=0, alpha=0.8)

    # drawing lines
    for i in range(len(keypoints[0])):
        line = matplotlib.patches.ConnectionPatch(
                xyA=(keypoints[0][i, 0], keypoints[0][i, 1]),
                xyB=(keypoints[1][i, 0], keypoints[1][i, 1]),
                coordsA=ax[0].transData,
                coordsB=ax[1].transData,
                axesA=ax[0],
                axesB=ax[1],
                zorder=1,
                color=color,
                linewidth=0.4,
                clip_on=True,
                alpha=0.5,
                label=None,
                picker=5.0,
            )
        line.set_annotation_clip(True)
        fig.add_artist(line)
    fig.tight_layout()
    fig.suptitle(fileName)

    if save:
        plt.savefig(f"{saveAddress}/{fileName}.png")
    return fig, ax

def plot_3image(images, save, saveAddress, fileName):
    """
    show and save the images with their features

    Parameters:
        images: a tuple of 3 images,
        save: boolean to allow saving,
        saveAddress: saving directory,
        fileName: file name for saving

    Returns:
    Figure & Axes
    """
    ratios = [4 / 3]*3
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, ax = plt.subplots(
    1, 3, figsize=figsize, dpi=100, gridspec_kw={"width_ratios": ratios}
    )

    # source image
    ax[0].imshow(images[0])
    ax[0].get_yaxis().set_ticks([])
    ax[0].get_xaxis().set_ticks([])
    ax[0].set_axis_off()
    ax[0].set_title("source")    

    # source transformed image
    ax[1].imshow(images[1])
    ax[1].get_yaxis().set_ticks([])
    ax[1].get_xaxis().set_ticks([])
    ax[1].set_axis_off()
    ax[1].set_title("source transformed")   
    
    # target image
    ax[2].imshow(images[2])
    ax[2].get_yaxis().set_ticks([])
    ax[2].get_xaxis().set_ticks([])
    ax[2].set_axis_off()
    ax[2].set_title("target")

    fig.tight_layout()
    fig.suptitle(fileName)
    if save:
        plt.savefig(f"{saveAddress}/{fileName}.png")
    return fig, ax

def ssd(image1, image2):
    """
    calculates sum of square differences for two images

    Parameters:
        image1: first image,
        image2: second image

    Returns:
    SSD of inputs
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape.")
    image1_normalized = cv2.normalize(image1, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image2_normalized = cv2.normalize(image2, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    squared_differences = np.square(image1_normalized - image2_normalized)
    ssd_error = np.sum(squared_differences)
    return ssd_error

def grad2(v):
    """
    calculates second order derivative of the input vector

    Parameters:
        v: Vector

    Returns:
    Vector of gradients
    """
    a = np.array([(v[i+1] + v[i-1] - 2*v[i]) for i in range(1,len(v)-1)])
    return a

def plot_bics(n_vector,bic_vector, bic_grad_vector, mark, save, saveAddress, fileName):
    """
    show and save BIC vectors with its gradient

    Parameters:
        n_vector: x-axis vector,
        bic_vector: vector containing bic values,
        bic_grad_vector: vector containing bic gradient values,
        mark: max value,
        save: boolean to allow saving,
        saveAddress: saving directory,
        fileName: file name for saving

    Returns:
    Figure & Axes
    """
    ratios = [4 / 3]*2
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, ax = plt.subplots(
    1, 2, figsize=figsize, dpi=100, gridspec_kw={"width_ratios": ratios}
    )
    
    # BIC
    index = np.where(n_vector == mark)[0]
    ax[0].plot(n_vector,bic_vector,'bo-',markersize=5)
    ax[0].set_xlim([1, 20])
    ax[0].get_xaxis().set_ticks(n_vector)
    ax[0].set_title("BIC")    
    ax[0].plot(n_vector[index], bic_vector[index], 'ro', markersize=7)
    
    # BIC Grad
    index = np.where(n_vector[1:-1] == mark)[0]
    ax[1].plot(n_vector[1:-1],bic_grad_vector,'go-',markersize=3)
    ax[1].set_xlim([1, 20])
    ax[1].get_xaxis().set_ticks(n_vector)
    ax[1].set_title("BIC 2nd order gradient")    
    ax[1].plot(n_vector[1:-1][index], bic_grad_vector[index], 'ro', markersize=5)

    fig.tight_layout()
    fig.suptitle(fileName)

    if save:
        plt.savefig(f"{saveAddress}/{fileName}.png")
    return fig, ax

def plot_gmm_map(img,nclusters, gmm, img_sz):
    """
    show and save gmm areas

    Parameters:
        img: original image
        nclusters: number of clusters,
        gmm: Fitted Gaussian Mixture Model,
        img_sz: tuple containing image height and width,

    Returns:
    GMM map and GMM mask
    """
    height, width = img_sz[:2]
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    positions = np.column_stack([xs.flatten(), ys.flatten()])
    colors = np.random.randint(50, 230, size=(nclusters, 3), dtype=np.uint8)
    pred = gmm.predict(positions)
    predicted_image = pred.reshape(height, width)
    masks = [(predicted_image == i).astype(np.uint8) * 255 for i in range(nclusters)]           
    color_image = colors[predicted_image]
    color_image = cv2.addWeighted(img,0.35,color_image,0.65,0)

    return color_image, masks

def annotate_transform(image, Transform, color=(0,0,0)):
    """
    Annotates an image with transformation parameters (Tx, Ty, Rotation)
    at specified location.

    Parameters:
        image (ndarray): The image array to annotate.
        Transform (list): A list of tuples, where each tuple contains:
                          (index, x, y, Tx, Ty, Rotation)
                          - index: Cluster Number
                          - x, y: Coordinates of the cluster center
                          - Tx, Ty: Transformation values in x and y directions
                          - Rotation: Rotation in degrees

    Returns:
        ndarray: The annotated image.
    """
    annotated_image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    index, x, y, tx, ty, rotation = Transform
    annotation = f"{tx:.2f}\n{ty:.2f}\n{rotation:.2f}"
    cv2.putText(annotated_image, f"{index}", (x, y-25), font, font_scale, color, font_thickness, cv2.LINE_AA)
    cv2.putText(annotated_image, f"{tx:.2f}", (x, y-15), font, font_scale, color, font_thickness, cv2.LINE_AA)
    cv2.putText(annotated_image, f"{ty:.2f}", (x, y), font, font_scale, color, font_thickness, cv2.LINE_AA)
    cv2.putText(annotated_image, f"{rotation:.2f}", (x, y+15), font, font_scale, color, font_thickness, cv2.LINE_AA)
    cv2.circle(annotated_image, (x, y), 3, color, -1)
    return annotated_image

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

def plot_graph(G, save, saveAddress, fileName):
    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = nx.get_edge_attributes(G, 'weight')

    ratios = [4 / 3]
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, ax = plt.subplots(
    1, 1, figsize=figsize, dpi=100, gridspec_kw={"width_ratios": ratios}
    )
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='skyblue', node_size=700, font_size=12, font_weight='bold', edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    ax.set_axis_off()
    ax.set_title(fileName)
    if save:
        plt.savefig(f"{saveAddress}/{fileName}.png")
    return fig, ax

def find_direct_neighbor(G, target_node):
    neighbors = []
    for neighbor in G.neighbors(target_node):
        edge_weight = G[target_node][neighbor]['weight']
        neighbors.append(([neighbor, 1 / edge_weight if edge_weight != 0 else float('inf')]))
    return neighbors


def conv_DCN(M):
    p0 = np.sqrt( M[0,0] + M[1,0]*1j)
    p1 = (M[0,2] + M[1,2]*1j)/(2*p0)
    return DCN(p0,p1)

def blend_DCN(Cs,w) -> DCN:
    c_hat_temp = [ci.scalar_mult(wi) for wi,ci in zip(w,Cs)]
    c_hat = sum(c_hat_temp)
    c_hat = c_hat/c_hat.norm()
    return c_hat


def createFinalOutputs(sFolder, dFolder, nFiles):

    os.makedirs(dFolder, exist_ok=True)

    for folder_name in range(nFiles):
        current_folder = os.path.join(sFolder, str(folder_name))
        source_file = os.path.join(current_folder, 'FinalImage.png')
        new_file_name = f'{folder_name}.jpg'
        destination_file = os.path.join(dFolder, new_file_name)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, destination_file)

