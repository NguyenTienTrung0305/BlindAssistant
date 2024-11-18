import os
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# labels
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

label_map = {k: v+1 for v, k in enumerate(voc_labels)} # start from 1, 'aeroplane: 1'
label_map['background'] = 0 # no object
rev_lable_map = {v:k for k, v in label_map.items()} # '1: aeroplane'

# bounding box colors
boundingbox_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']

label_colors_map = {k: boundingbox_colors[i] for i,k in enumerate(label_map.keys())} # 0: '#e6194b'; 1: '#3cb44b'



def xy_to_cxcy(xy):
    # (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h)
    # xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    return torch.cat([
                       (xy[:,0] + xy[:, 2])/2, # cx
                       (xy[:,1] + xy[:, 3])/2, # cy
                       xy[:, 2] - xy[:, 0],    # w
                       xy[:, 3] - xy[:, 1],    # h
                      ], dim=1)

def cxcy_to_xy(cxcy):
    # (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max)
    # cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    return torch.cat([
                       cxcy[:,0] - cxcy[:, 2]/2, # x_min
                       cxcy[:,1] - cxcy[:, 3]/2, # y_min
                       cxcy[:,0] + cxcy[:, 2]/2, # x_max
                       cxcy[:,1] + cxcy[:, 3]/2, # y_max
                      ], dim=1)

# Mỗi prior box có thể có một tỷ lệ chiều rộng và chiều cao rất khác nhau, điều này có thể khiến việc dự đoán trở nên 
#   phức tạp hơn và mô hình khó học các mối quan hệ giữa các bounding box thực tế và các prior box
# Chênh lệch giữa các giá trị của các bounding box thực tế và prior box có thể rất lớn
# => ground-center-size coordinates giải quyết 2 vấn đề này
def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    # Convert center-size coordinates to ground-center-size coordinates
    # cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    # priors_cxcy: prior boxes in center-size coordinates, a tensor of size (n_priors, 4)
    return torch.cat([
                       (cxcy[:,0] - priors_cxcy[:,0]) / (priors_cxcy[:,2] / 10), # gc_x
                       (cxcy[:,1] - priors_cxcy[:,1]) / (priors_cxcy[:,3] / 10), # gc_y
                       torch.log(cxcy[:,2] / priors_cxcy[:,2]) * 5,              # log_w
                       torch.log(cxcy[:,3] / priors_cxcy[:,3]) * 5,              # log_h
                      ], dim=1)

def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    return torch.cat([
                        torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                        torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h
                    ], dim=1)



def find_intersection(set1, set2):
    """_summary_

    Args:
        set1 (_type_): (n1, 4)
        set2 (_type_): (n2, 4)

    Returns:
        _type_: _description_
    """
    lower_bound = torch.max(set1[:, :2].unsqueeze(1), set2[:, :2].unsqueeze(0)) #(n1,n2,2)
    upper_bound = torch.min(set1[:, 2:].unsqueeze(1), set2[:, 2:].unsqueeze(0)) #(n1,n2,2)
    intersection_ = torch.clamp(upper_bound - lower_bound, min=0) #(n1,n2,2)
    return intersection_[:, :, 0] * intersection_[:, :, 1] # (n1,n2)

def find_jaccard_overlap(set1, set2):
    intersec = find_intersection(set1, set2) #(n1,n2)
    areas1 = (set1[:, 2] - set1[:, 0]) * (set1[:, 3] - set1[:, 1]) #(n1)
    areas2 = (set2[:, 2] - set2[:, 0]) * (set2[:, 3] - set2[:, 1]) #(n2)
    union = areas1.unsqueeze(1) + areas2.unsqueeze(0) - intersec #(n1,n2)
    return intersec / union




