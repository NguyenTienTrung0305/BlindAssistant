import torch
from torch import nn
from util import *
import torch.nn.functional as F
from math import sqrt
from itertools import product
import torchvision
import torchvision.models as models
from torchvision.models import mobilenet_v2
import torchvision.transforms as transforms 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomMobilenet(nn.Module):
    """
     base convolutions to produce lower-level feature maps.
    """
    def __init__(self):
        super(CustomMobilenet, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6) 

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        # Load pretrained layers
        self.load_pretrained_layers()

    def forward(self, image):
        out = F.relu(self.conv1_1(image))  # (N, 64, 300, 300)
        out = F.relu(self.conv1_2(out))  # (N, 64, 300, 300)
        out = self.pool1(out)  # (N, 64, 150, 150)

        out = F.relu(self.conv2_1(out))  # (N, 128, 150, 150)
        out = F.relu(self.conv2_2(out))  # (N, 128, 150, 150)
        out = self.pool2(out)  # (N, 128, 75, 75)

        out = F.relu(self.conv3_1(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_2(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_3(out))  # (N, 256, 75, 75)
        out = self.pool3(out)  # (N, 256, 38, 38)

        out = F.relu(self.conv4_1(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_2(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_3(out))  # (N, 512, 38, 38)
        conv4_3_feats = out  # (N, 512, 38, 38)
        out = self.pool4(out)  # (N, 512, 19, 19)

        out = F.relu(self.conv5_1(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_2(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_3(out))  # (N, 512, 19, 19)
        out = self.pool5(out)  # (N, 512, 19, 19)

        out = F.relu(self.conv6(out))  # (N, 1024, 19, 19)

        conv7_feats = F.relu(self.conv7(out))  # (N, 1024, 19, 19)

        # Lower-level feature maps
        return conv4_3_feats, conv7_feats

    def load_pretrained_layers(self):
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        for i, param in enumerate(param_names[:-4]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")
        
class SupportConvolutions(nn.Module):
    def __init__(self):
        super(SupportConvolutions, self).__init__()
        # self.conv1 = nn.Conv2d(1280, 512, kernel_size=1, stride=1, padding=0)
        
        self.conv1_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)
        self.conv1_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
         
        self.conv2_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0) 
        self.conv2_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        self.conv3_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0) 
        self.conv3_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  

        self.conv4_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv4_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  
        
        self.init_w_b()
    
    def init_w_b(self):
        for i in self.children():
            if isinstance(i, nn.Conv2d):
                nn.init.xavier_uniform_(i.weight)
                nn.init.constant_(i.bias, 0.)
                
                
    def forward(self, feature_l_2):
        # out_cv = self.conv1(feature_l_2) # (N, 512, 10, 10)
        # out_cv = F.relu(out_cv)
        # featuremap_1 = out_cv
        
        out_cv1_1 = self.conv1_1(feature_l_2) # (N, 256, 19, 19)
        out_cv1_1 = F.relu(out_cv1_1)
        
        out_cv1_2 = self.conv1_2(out_cv1_1) # (N, 512, 10, 10)
        out_cv1_2 = F.relu(out_cv1_2) 
        featuremap_1 = out_cv1_2
        
        # out_cv2_1 = self.conv2_1(out_cv)
        out_cv2_1 = self.conv2_1(out_cv1_2) # (N, 128, 10, 10)
        out_cv2_1 = F.relu(out_cv2_1)
        
        out_cv2_2 = self.conv2_2(out_cv2_1) # (N, 256, 5, 5)
        out_cv2_2 = F.relu(out_cv2_2)
        featuremap_2 = out_cv2_2
        
        out_cv3_1 = self.conv3_1(out_cv2_2) # (N, 128, 5, 5)
        out_cv3_1 = F.relu(out_cv3_1)
        
        out_cv3_2 = self.conv3_2(out_cv3_1) # (N, 256, 3, 3)
        out_cv3_2 = F.relu(out_cv3_2)
        featuremap_3 = out_cv3_2
        
        out_cv4_1 = self.conv4_1(out_cv3_2) # (N, 128, 3, 3)
        out_cv4_1 = F.relu(out_cv4_1)
        
        out_cv4_2 = self.conv4_2(out_cv4_1) # (N, 256, 1, 1)
        out_cv4_2 = F.relu(out_cv4_2)
        featuremap_4 = out_cv4_2
        
        return featuremap_1, featuremap_2, featuremap_3, featuremap_4


    
class PredictBox(nn.Module):
    def __init__(self, n_classes):
        super(PredictBox, self).__init__()
        self.n_classes = n_classes
        
        # số lượng anchor box cho các features map
        n_boxes = {'feature_l_1': 4, # small object
                   'feature_l_2': 6,
                   'featuremap_1': 6, 
                   'featuremap_2': 6, # md object
                   'featuremap_3': 4,
                   'featuremap_4': 4} # large object
        
        # n_boxes['feature_l_1'] * 4: (mỗi anchor box cần 4 giá trị để tạo ra bounding box)
        # self.loc_feature_l_1 = nn.Conv2d(32, n_boxes['feature_l_1'] * 4, kernel_size=3, padding=1)
        # self.loc_feature_l_2 = nn.Conv2d(1280, n_boxes['feature_l_2'] * 4, kernel_size=3, padding=1)
        self.loc_feature_l_1 = nn.Conv2d(512, n_boxes['feature_l_1'] * 4, kernel_size=3, padding=1)
        self.loc_feature_l_2 = nn.Conv2d(1024, n_boxes['feature_l_2'] * 4, kernel_size=3, padding=1)
        self.loc_featuremap_1 = nn.Conv2d(512, n_boxes['featuremap_1'] * 4, kernel_size=3, padding=1)
        self.loc_featuremap_2 = nn.Conv2d(256, n_boxes['featuremap_2'] * 4, kernel_size=3, padding=1)
        self.loc_featuremap_3 = nn.Conv2d(256, n_boxes['featuremap_3'] * 4, kernel_size=3, padding=1)
        self.loc_featuremap_4 = nn.Conv2d(256, n_boxes['featuremap_4'] * 4, kernel_size=3, padding=1)
        
        # n_boxes['feature_l_1'] * n_classes: (mỗi anchor box thuộc 1 class trong n_classses)
        # self.cl_feature_l_1 = nn.Conv2d(32, n_boxes['feature_l_1'] * n_classes, kernel_size=3, padding=1)
        # self.cl_feature_l_2 = nn.Conv2d(1280, n_boxes['feature_l_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_feature_l_1 = nn.Conv2d(512, n_boxes['feature_l_1'] * n_classes, kernel_size=3, padding=1)
        self.cl_feature_l_2 = nn.Conv2d(1024, n_boxes['feature_l_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_featuremap_1 = nn.Conv2d(512, n_boxes['featuremap_1'] * n_classes, kernel_size=3, padding=1)
        self.cl_featuremap_2 = nn.Conv2d(256, n_boxes['featuremap_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_featuremap_3 = nn.Conv2d(256, n_boxes['featuremap_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_featuremap_4 = nn.Conv2d(256, n_boxes['featuremap_4'] * n_classes, kernel_size=3, padding=1)
        
        self.init_w_b()
        
    def init_w_b(self):
        for i in self.children():
            if isinstance(i, nn.Conv2d):
                nn.init.xavier_uniform_(i.weight)
                nn.init.constant_(i.bias, 0.)
        
    
    def forward(self, feature_l_1, feature_l_2, featuremap_1, featuremap_2, featuremap_3, featuremap_4):
        batch_size = feature_l_1.size(0)
        
        # dự đoán bounding box
        l_feature_l_1 = self.loc_feature_l_1(feature_l_1)
        
        # Thay đổi thứ tự chiều tensor (batch_size, n_box * 4 , height, width) => (batch_size, height, width, n_box * 4)
        l_feature_l_1 = l_feature_l_1.permute(0, 2, 3, 1).contiguous()
        
        # Dùng view để thay đổi shape của tensor, chỉ thay đổi shape, các giá trị không thay đổi, thứ tự giá trị sẽ được fill từ trái -> phải, trên xuống dưới
        # (batch_size, height * width * n_box * 4 / 4, 4) (mục tiêu là biểu diễn toàn bộ anchor box về 1 chiều, mỗi anchor box có 4 giá trị)
        l_feature_l_1 = l_feature_l_1.view(batch_size, -1, 4)
        
        l_feature_l_2 = self.loc_feature_l_2(feature_l_2)
        l_feature_l_2 = l_feature_l_2.permute(0, 2, 3, 1).contiguous()
        l_feature_l_2 = l_feature_l_2.view(batch_size, -1, 4)
        
        l_featuremap_1 = self.loc_featuremap_1(featuremap_1)
        l_featuremap_1 = l_featuremap_1.permute(0, 2, 3, 1).contiguous()
        l_featuremap_1 = l_featuremap_1.view(batch_size, -1, 4)
        
        l_featuremap_2 = self.loc_featuremap_2(featuremap_2)
        l_featuremap_2 = l_featuremap_2.permute(0, 2, 3, 1).contiguous()
        l_featuremap_2 = l_featuremap_2.view(batch_size, -1, 4)
        
        l_featuremap_3 = self.loc_featuremap_3(featuremap_3)
        l_featuremap_3 = l_featuremap_3.permute(0, 2, 3, 1).contiguous()
        l_featuremap_3 = l_featuremap_3.view(batch_size, -1, 4)
        
        l_featuremap_4 = self.loc_featuremap_4(featuremap_4)
        l_featuremap_4 = l_featuremap_4.permute(0, 2, 3, 1).contiguous()
        l_featuremap_4 = l_featuremap_4.view(batch_size, -1, 4)
        
        
        
        # dự đoán class ứng với bounding box
        cl_feature_l_1 = self.cl_feature_l_1(feature_l_1)
        cl_feature_l_1 = cl_feature_l_1.permute(0, 2, 3, 1).contiguous()
        cl_feature_l_1 = cl_feature_l_1.view(batch_size, -1, self.n_classes) # (biểu diễn toàn bộ bounding box trên 1 chiều, mỗi bounding box là classs biểu diễn dưới dạng (0 0 0 1 0 0 0 ....))
        
        cl_feature_l_2 = self.cl_feature_l_2(feature_l_2)
        cl_feature_l_2 = cl_feature_l_2.permute(0, 2, 3, 1).contiguous()
        cl_feature_l_2 = cl_feature_l_2.view(batch_size, -1, self.n_classes)
        
        cl_featuremap_1 = self.cl_featuremap_1(featuremap_1)
        cl_featuremap_1 = cl_featuremap_1.permute(0, 2, 3, 1).contiguous()
        cl_featuremap_1 = cl_featuremap_1.view(batch_size, -1, self.n_classes)
        
        cl_featuremap_2 = self.cl_featuremap_2(featuremap_2)
        cl_featuremap_2 = cl_featuremap_2.permute(0, 2, 3, 1).contiguous()
        cl_featuremap_2 = cl_featuremap_2.view(batch_size, -1, self.n_classes)
        
        cl_featuremap_3 = self.cl_featuremap_3(featuremap_3)
        cl_featuremap_3 = cl_featuremap_3.permute(0, 2, 3, 1).contiguous()
        cl_featuremap_3 = cl_featuremap_3.view(batch_size, -1, self.n_classes)
        
        cl_featuremap_4 = self.cl_featuremap_4(featuremap_4)
        cl_featuremap_4 = cl_featuremap_4.permute(0, 2, 3, 1).contiguous()
        cl_featuremap_4 = cl_featuremap_4.view(batch_size, -1, self.n_classes)
        
        predict_location = torch.cat([l_feature_l_1, l_feature_l_2, l_featuremap_1, l_featuremap_2, l_featuremap_3, l_featuremap_4], dim=1) # (N, 7166,4)
        predict_classes = torch.cat([cl_feature_l_1, cl_feature_l_2, cl_featuremap_1, cl_featuremap_2, cl_featuremap_3, cl_featuremap_4], dim=1) # (N, 7166, n_classes)
        
        return predict_location, predict_classes
    
    
class SSD300(nn.Module):
    def __init__(self, n_classes):
        super(SSD300, self).__init__()
        
        self.n_classes = n_classes
        self.backbone = CustomMobilenet()
        self.addconv = SupportConvolutions()
        self.predconv = PredictBox(n_classes=n_classes)
        
        self.rescale = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale, 20)
        
        self.priors_cxcy = self.create_prior_boxes()
        
    def forward(self, image):
        fm_l1, fm_l2 = self.backbone(image)
        
        # rescale fm_l1 after L2 norm
        norm = fm_l1.pow(2).sum(dim=1, keepdim=True).sqrt()
        fm_l1 = fm_l1 / norm
        fm_l1 = fm_l1 * self.rescale
        
        fm1, fm2, fm3, fm4 = self.addconv(fm_l2)
        
        pdlocation, pdclasses = self.predconv(fm_l1, fm_l2, fm1, fm2, fm3, fm4)
        
        return pdlocation, pdclasses
    
    def create_prior_boxes(self):
        # Feature Map Dimensions
        fmap_dims = {'fm_l1': 38,
                     'fm_l2': 19,
                     'fm1': 10,
                     'fm2': 5,
                     'fm3': 3,
                     'fm4': 1
                    }

        # bouding box có kích thước bằng % kích thước input_image
        obj_scales = {'fm_l1': 0.1,
                      'fm_l2': 0.2,
                      'fm1': 0.375,
                      'fm2': 0.55,
                      'fm3': 0.725,
                      'fm4': 0.9
                    }

        # shape bounding box
        # 1: vuông
        # 2: dài ngang
        # 0.5: dài dọc
        # 3: dài ngang lớn hơn
        # 0.33: dài dọc hơn
        aspect_ratios = {'fm_l1': [1., 2., 0.5],
                         'fm_l2': [1., 2., 3., 0.5, 0.333],
                         'fm1': [1., 2., 3., 0.5, 0.333],
                         'fm2': [1., 2., 3., 0.5, 0.333],
                         'fm3': [1., 2., 0.5],
                         'fm4': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())
        
        prior_boxes = []
        
        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap] # chuan hoa ve [0-1]
                    cy = (j + 0.5) / fmap_dims[fmap] 
                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)]) # (cx, cy, w, h)
                        
                        # additional_scale: giúp mạng phát hiện các đối tượng có kích thước ở giữa hai feature map liền kề, 
                            # tăng khả năng bao phủ các đối tượng có kích thước đa dạng mà không cần thêm các feature map hoặc 
                            # tăng số lượng hộp prior lên quá nhiều
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])
                            
        
        prior_boxes = torch.FloatTensor(prior_boxes).to(device) # (7166, 4), 7166: số lượng bounding box
        prior_boxes.clamp_(0, 1) # đảm bảo giá trị không vượt quá [0,1]
        return prior_boxes
    

    def detect_object(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy .size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2) # (N, 7166, n_classes) (Batchsize, n_priors, n_classes)
        
        all_images_boxs = list()
        all_images_labels = list()
        all_images_scores = list()
        
        for i in range(batch_size):
            
            ### Mô hình không trực tiếp dự đoán giá trị tọa độ góc mà dự đoán các giá trị offset
            # giá trị offset là độ lệch giữa tọa độ của prior box và true_box (prior box là những hộp cố định)
            # mô hình sẽ đưa ra dự đoán cho các giá trị offset (predicted_box) và fit giá trị này với giá trị offset 
            # cxcy_to_xy(gcxgcy_to_cxcy()) chuyển từ tọa độ góc sang tọa độ (cx,cy,w,h) sau đó chuyển sang offset
            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy )) 

            image_box = list()
            image_label = list()
            image_score = list()
            
            max_scores, best_label = predicted_scores[i].max(dim=1) # (7166), hàm max trả về điểm số cao nhất và chỉ số tương ứng của nó
            
            # xử lý từng đối tượng
            for c in range(1, self.n_classes):
                # predicted_scores[i] với kích thước (n_priors, n_classes), cú pháp [:, c] trích ra điểm dự đoán cho lớp c của tất cả các prior boxes.
                class_scores = predicted_scores[i][:, c] # (7166), 
                class_higher_min_scores = class_scores > min_score  # các hộp có xác suất cao hơn min_score
                n_class_higher_min_scores = class_higher_min_scores.sum().item() # tổng số lượng các hộp có xác suất cao hơn min_score
                
                # không có hộp nào vượt qua ngưỡng, chuyển sang đối tượng tiếp theo
                if n_class_higher_min_scores == 0:
                    continue
                
                class_scores = class_scores[class_higher_min_scores] # (n_qualified), trả về các prior boxes sau khi được lọc
                class_decode_locs = decoded_locs[class_higher_min_scores] # (n_qualified, 4)

                # sắp xếp các prior boxes theo thứ tự giảm dần 
                class_scores, sort_idx = class_scores.sort(dim=0, descending=True)
                class_decode_locs = class_decode_locs[sort_idx]
                
                
                # NMS
                overlap = find_jaccard_overlap(class_decode_locs, class_decode_locs) # (n_qualified, n_qualified)
                
                # Khởi tạo filter để lọc các prior boxes
                suppress = torch.zeros((n_class_higher_min_scores), dtype=torch.uint8).to(device)
                
                for box in range(class_decode_locs.size(0)):
                    # nếu hộp đã được cho là chồng chéo thì không xét nữa
                    if suppress[box] == 1:
                        continue
                    # overlap[box] > max_overlap sẽ tạo ra một tensor boolean, với giá trị True (1) cho các hộp có độ chồng chéo lớn hơn max_overlap 
                        # và False (0) cho những hộp không chồng chéo
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    suppress[box] = 0 # đánh dấu hộp hiện tại là không bị loại bỏ (vì duyệt prior boxs theo điểm số giảm dần)
                
                # Chỉ lưu những prior boxes thỏa mãn NMS
                image_box.append(class_decode_locs[1-suppress]) # (n_final_boxes, 4), (n_final_boxes của mỗi object là khác nhau)
                image_label.append(torch.LongTensor((1-suppress).sum().item() * [c]).to(device)) # tạo danh sách gồm n_final_boxes phần tử c (c= 1,2,3,4...)
                image_score.append(class_scores[1-suppress]) # (n_final_boxes)
            
            # nếu không có đối tượng nào được tìm thấy trong ảnh, thêm các giá trị mặc định để đảm bảo xử lý bước sau không bị lỗi
            if len(image_box) == 0:
                image_box.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_label.append(torch.LongTensor([0]).to(device))
                image_score.append(torch.FloatTensor([0.]).to(device))
            
            
            # nối danh sách chứa các tensor thành danh sách chứa single tensor
            image_box = torch.cat(image_box, dim=0) # (n_objects, 4) , object có thể giống nhau
            image_label = torch.cat(image_label, dim=0) # (n_objects)
            image_score = torch.cat(image_score, dim=0) # (n_objects)
            n_objects = image_score.size(0)
            
            # lấy top_k đối tượng có điểm số cao nhấts
            if n_objects > top_k:
                image_score , sort_idx = image_score.sort(dim=0, descending=True)
                image_score = image_score[:top_k] 
                image_box = image_box[sort_idx][:top_k] # (top_k, 4)
                image_label = image_label[sort_idx][:top_k]
            
            all_images_boxs.append(image_box)
            all_images_labels.append(image_label)
            all_images_scores.append(image_score)
            
        return all_images_boxs, all_images_labels, all_images_scores
    
    
class CPLoss(nn.Module):
    """
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores
    """
    
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(CPLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        
        # CrossEntropyLoss tự động thực hiện softmax sau đó mới tính cross entropy
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False) 

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """

        Args:
            predicted_locs (_type_): (N, 7166, 4)
            predicted_scores (_type_): (N, 7166, n_classes)
            boxes (_type_): true bounding boxes (n_objects, 4)
            labels (_type_): true lable (n_objects)
        """
        
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)
        
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device) # # (N, 7166, 4)
        true_classes = torch.zeros((batch_size,n_priors), dtype=torch.long).to(device) # # (N, 7166)
        
        # for each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)
            
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)  # (n_objects, 7166)
            
            # Tìm bounding box (object box) tốt nhất cho mỗi prior box
            # SSD dự đoán cho một số lượng cố định prior boxes, vì vậy cần ánh xạ từng prior box tới bounding box thật phù hợp nhất
            # overlap_for_each_prior: Giá trị IoU cao nhất cho từng prior box
            # object_for_each_prior: Chỉ số bounding box thực tế tương ứng với giá trị IoU cao nhất
            # object_for_each_prior: chứa các giá trị từ [0 -> n_object - 1]
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (7166)
            
            # Tìm prior box tốt nhất cho mỗi bounding box (object box)
            # prior_for_each_object trả về (29,39,60)
            #  => best IoU của object_0 là prior box thứ 29
            #  => best IoU của object_1 là prior box thứ 39
            #  => best IoU của object_2 là prior box thứ 60
            _, prior_for_each_object = overlap.max(dim=1)
            
            # Để đảm bảo mỗi bounding box luôn có ít nhất 1 prior box ánh xạ tới => mô hình mới học được cách dự đoán hộp cho bounding box đó
            # torch.LongTensor(range(n_objects)).to(device) =>  (0,1,.. n_object-1) đại diện cho thứ tự objects
            # Với ví dụ ở trên, có thể prior box thứ 29 không ánh xạ đến object_0 => chúng ta ép prior box thứ 29 phải ánh xạ đến object_0 
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)
            
            # Thiết lập IoU của các prior boxes "tốt nhất" bằng 1.0 để đảm bảo chúng luôn được chọn trong huấn luyện.
            overlap_for_each_prior[prior_for_each_object] = 1.

            # gán nhãn thực tế cho prior boxes
            # sử dụng Indexing : Sử dụng một danh sách hoặc tensor khác làm chỉ số để truy cập nhiều phần tử cùng lúc
            # Ví dụ: labels[i] = [3, 3, 5, 6], object_for_each_prior = [0, 2, 1, 3, 0]
                # => labels[i][object_for_each_prior] sẽ lấy giá trị labels[i] dựa trên chỉ số [0,2,1,1,0] => kq: [3,5,3,6,3]
            # label_for_each_prior có giá trị từ 1 -> n_classes 
            label_for_each_prior = labels[i][object_for_each_prior]  # (7166)
            
            # Các prior box nào có overlap_for_each_prior < threshold thì gán nhãn là 0 ( no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (7166)

            # Store
            true_classes[i] = label_for_each_prior

            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (7166, 4)
        
        # (N, 7166), giá trị của true_classes != 0 thì gán bằng 1, không thì bằng 0
        # positive_priors: chi biết prior box có tiềm năng chứa object không, 1: có ,0: không
        positive_priors = true_classes != 0  
        
        
        # LOCATION LOSS
        ### Mô hình không trực tiếp dự đoán giá trị tọa độ góc mà dự đoán các giá trị offset
        # giá trị offset là độ lệch giữa tọa độ của prior box và true_box (prior box là những hộp cố định)
        # mô hình sẽ đưa ra dự đoán cho các giá trị offset (predicted_box) và cố gắng fit giá trị này với giá trị offset 
        locations_loss  = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])

        
        # CONFIDENCE LOSS
        n_positives = positive_priors.sum(dim=1) # (N,); n_positives: tổng số lượng priors_box có tiềm năng chứa object
        n_hard_negatives = self.neg_pos_ratio * n_positives # (N,) giá trị trong n_hard_negatives sẽ tỉ lệ với n_positives = neg_pos_ratio
        
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1)) # (N * 7166)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors) # (N, 7166)
        
        # loss for positive priors
        conf_loss_pos = conf_loss_all[positive_priors]
        
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0  # (N, 7166); bỏ qua loss của positive_priors
        
        # quan tâm đến loss của neg_prior, loss càng lớn càng quan tâm vì nó ảnh hưởng lớn
        # Tại sao cần neg_prior
            # + số lượng prior boxes không chứa đối tượng (negatives) thường áp đảo số lượng prior boxes chứa đối tượng (positives).
            # + Nếu không xử lý, mô hình sẽ ưu tiên học cách dự đoán "no object" cho tất cả các prior boxes vì điều này giúp giảm loss nhanh chóng, 
            #     nhưng lại không học được cách phát hiện chính xác các đối tượng.
            # => mục đích phải phân loại đúng các prior boxes không phát hiện đối tượng để giảm thiểu sai số
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True) # (N, 7166)
        
        # N rows, mỗi rows có giá trị tử 0 đến n_priors - 1
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 7166)
        
        # N rows, mỗi rows chứa giá trị 1 hoặc 0
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 7166)
        
        # Chọn n_hard_negatives thằng priors_box có loss giảm dần
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float() 

        # TOTAL LOSS

        return conf_loss + self.alpha * locations_loss