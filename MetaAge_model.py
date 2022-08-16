import torch.nn as nn
import copy
import torch

FACE_MODEL_DIM=2048

W_CAT_DIM=4096 #the dimension of tensor, with which the face feature is cancatted
AGE_DIM=101

class MetaAge(nn.Module):
    def __init__(self, vgg_model, face_feature_model, device, MLP_dim=[FACE_MODEL_DIM+W_CAT_DIM+AGE_DIM, 8192, 4096]):
        super(MetaAge, self).__init__()

        last_fc_w = copy.deepcopy(vgg_model.classifier[-1].weight)
        new_classifier = nn.Sequential(*list(vgg_model.classifier.children())[:-1])
        vgg_model.classifier = new_classifier

        self.vgg_features = vgg_model
        self.last_fc = nn.Parameter(last_fc_w)

        print("MLP dim:", MLP_dim)
        MLP_trans = []
        for i in range(len(MLP_dim) - 1):
            MLP_trans.append(nn.BatchNorm1d(MLP_dim[i]))
            MLP_trans.append(nn.ReLU())
            MLP_trans.append(nn.Linear(in_features=MLP_dim[i], out_features=MLP_dim[i + 1]))
      
        self.MLP_trans = nn.Sequential(*MLP_trans)
        self.face_feature_model=face_feature_model
        self.device = device

    def forward(self, x224):
        self.vgg_features.training = self.training

        x_face_feature = x224
        face_feature=self.face_feature_model(x_face_feature) #bn 2048
        face_feature=face_feature.detach()

        new_w = self.mix_w_face_feature_cat(face_feature) 

        feature_x = self.vgg_features(x224)  # bn 4096
        feature_x_us = feature_x.unsqueeze(2) #bn 4096 1
        new_w_x = torch.bmm(new_w, feature_x_us)
        new_w_x = torch.squeeze(new_w_x, 2) #bn 101

        return new_w_x

    def mix_w_face_feature_cat(self,face_feature):
        if AGE_DIM==1:
          age = torch.Tensor(list(range(101))) #101
          age = age.unsqueeze(1) #101 1
        elif AGE_DIM==101:
          age = torch.eye(101) #101 101
        age = age.to(device=self.device)

        age_expand = age.expand([face_feature.shape[0], age.shape[0], age.shape[1]]) #bn 101 x

        w_expand = self.last_fc.expand([face_feature.shape[0], self.last_fc.shape[0], self.last_fc.shape[1]]) #bn 101 4096
        face_feature = face_feature.unsqueeze(1)
        face_feature_expand = face_feature.expand([face_feature.shape[0], w_expand.shape[1], face_feature.shape[2]]) #bn 101 2048
        w_age_f_cat = torch.cat([face_feature_expand, w_expand, age_expand], 2) #bn 101 4096+2048+x
        w_age_f_cat_view = w_age_f_cat.view(-1, w_age_f_cat.shape[-1]) #bn*101 4096+2048+x
        new_w = self.MLP_trans(w_age_f_cat_view) #bn*101 4096
        new_w = new_w.view(w_expand.shape) #bn 101 4096
        new_w=new_w+w_expand
        return new_w

