
# Python Imports

# Package Imports
import numpy as np
import matplotlib
import torch

# Project Imports
from osm.parser import Groups


map_colors = {
    "building": (84, 155, 255),
    "parking": (255, 229, 145),
    "playground": (150, 133, 125),
    "grass": (188, 255, 143),
    "park": (0, 158, 16),
    "forest": (0, 92, 9),
    "water": (184, 213, 255),
    "fence": (238, 0, 255),
    "wall": (0, 0, 0),
    "hedge": (107, 68, 48),
    "kerb": (255, 234, 0),
    "building_outline": (0, 0, 255),
    "cycleway": (0, 251, 255),
    "path": (8, 237, 0),
    "road": (255, 0, 0),
    "tree_row": (0, 92, 9),
    "busway": (255, 128, 0),
    "void": [int(255 * 0.9)] * 3,
}


class Colormap:
    colors_areas = np.stack([map_colors[k] for k in ["void"] + Groups.areas])
    colors_ways = np.stack([map_colors[k] for k in ["void"] + Groups.ways])

    @classmethod
    def apply(cls, rasters, return_grayscale=False):

        # If we are a tensor then we need to convert to numpy
        if(torch.is_tensor(rasters)):
            rasters = rasters.cpu().detach().numpy()


        if(len(rasters.shape) == 3):
            img = np.where(rasters[1, ..., None] > 0, cls.colors_ways[rasters[1]], cls.colors_areas[rasters[0]],)
            img = img / 255.0

            # Convert to grayscale
            if(return_grayscale):
               img[...] = np.mean(img, axis=-1, keepdims=True) 

            return img

        elif(len(rasters.shape) == 4):

            # Do each image once
            imgs = []
            for i in range(rasters.shape[0]):
                img = np.where(rasters[i, 1, ..., None] > 0, cls.colors_ways[rasters[i, 1]], cls.colors_areas[rasters[i, 0]],)
                img = img / 255.0
                imgs.append(img)

            # Stack the images 
            imgs = np.stack(imgs)

            # Convert to grayscale
            if(return_grayscale):
               imgs[...] = np.mean(imgs, axis=-1, keepdims=True) 

            return imgs

        else:
            assert(False)

    # @classmethod
    # def add_colorbar(cls):
    #     ax2 = plt.gcf().add_axes([1, 0.1, 0.02, 0.8])
    #     color_list = np.r_[cls.colors_areas[1:], cls.colors_ways[1:]] / 255.0
    #     cmap = mpl.colors.ListedColormap(color_list[::-1])
    #     ticks = np.linspace(0, 1, len(color_list), endpoint=False)
    #     ticks += 1 / len(color_list) / 2
    #     cb = mpl.colorbar.ColorbarBase(
    #         ax2,
    #         cmap=cmap,
    #         orientation="vertical",
    #         ticks=ticks,
    #     )
    #     cb.set_ticklabels((Groups.areas + Groups.ways)[::-1])
    #     ax2.tick_params(labelsize=15)


# def plot_nodes(idx, raster, fontsize=8, size=15):
#     ax = plt.gcf().axes[idx]
#     ax.autoscale(enable=False)
#     nodes_xy = np.stack(np.where(raster > 0)[::-1], -1)
#     nodes_val = raster[tuple(nodes_xy.T[::-1])] - 1
#     ax.scatter(*nodes_xy.T, c="k", s=size)
#     for xy, val in zip(nodes_xy, nodes_val):
#         group = Groups.nodes[val]
#         add_text(
#             idx,
#             group,
#             xy + 2,
#             lcolor=None,
#             fs=fontsize,
#             color="k",
#             normalized=False,
#             ha="center",
#         )
