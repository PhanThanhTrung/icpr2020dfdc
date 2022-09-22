import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from scipy.special import expit
from torch.utils.model_zoo import load_url
from torchmetrics import ConfusionMatrix

sys.path.append('..')

from architectures import fornet, weights
from blazeface import BlazeFace, FaceExtractor, VideoReader
from isplutils import utils

net_model = 'EfficientNetAutoAttB4'
train_db = 'DFDC'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
face_policy = 'scale'
face_size = 224
frames_per_video = 32

model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]
net = getattr(fornet,net_model)().eval().to(device)
net.load_state_dict(load_url(model_url,map_location=device,check_hash=True))

transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)

facedet = BlazeFace().to(device)
facedet.load_weights("./blazeface/blazeface.pth")
facedet.load_anchors("./blazeface/anchors.npy")
videoreader = VideoReader(verbose=False)
video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn=video_read_fn,facedet=facedet)

cnt = 0
all_video_path = glob.glob('./dataset/*/*')
data_column = ['real', 'deepfake']
all_prediction, groundtruth = [], []
total = len(all_video_path)
confusion_matrix = ConfusionMatrix(num_classes=2)
for video_path in all_video_path:
    vid_faces = face_extractor.process_video(video_path)
    try:
        im_real_face = vid_faces[0]['faces'][0]
    except:
        continue
    # import ipdb; ipdb.set_trace()
    faces_real_t = torch.stack( [ transf(image=frame['faces'][0])['image'] for frame in vid_faces if len(frame['faces'])] )

    with torch.no_grad():
        faces_real_pred = net(faces_real_t.to(device)).cpu().numpy().flatten()

    print('Average score for {} video: {:.4f}'.format(video_path, expit(faces_real_pred.mean())))
    if expit(faces_real_pred.mean()) > 0.5:
        all_prediction.append(1)
    else:
        all_prediction.append(0)
    if video_path.split('/')[-2] == 'real':
        groundtruth.append(0)
    else:
        groundtruth.append(1)
    

pred = torch.tensor(all_prediction).unsqueeze(dim = 1)
groundtruth = torch.tensor(groundtruth).unsqueeze(dim = 1)

output = confusion_matrix(pred, groundtruth).detach().cpu().numpy()

df_cm = pd.DataFrame(output, data_column, data_column)
plt.figure(figsize=(10, 10))
sns.set(font_scale=0.9)  # for label size
heatmap = sns.heatmap(df_cm, annot=True, fmt='g', annot_kws={
    "size": 11})  # font size
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=75)
title = "Confusion matrix"
plt.title(title)
plt.xlabel("Predicted")
plt.ylabel("True label")
plt.tight_layout()
plt.savefig("confusion_matrix.jpg")        
        
