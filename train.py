from base_models import VGG16
import torch

## test
from utils import SSDAugmentation, VOCDetection, detection_collate
from layers import MultiBoxLoss
import torch.utils.data as data

from RefineDet import refine_det

import json
conf = json.loads(open('conf.json').read())
model_conf = conf['model_conf']
bbox_conf  = conf['bbox_conf']
train_conf = conf['train_conf']
backbone = VGG16()
model = refine_det(backbone, model_conf).cuda()
#model.weight_initialize()
print(model)

dataset = VOCDetection(root='../../data/VOCdevkit/', transform=SSDAugmentation(512))
dl = data.DataLoader(dataset = dataset, collate_fn=detection_collate, batch_size=12, shuffle=True)
criterion = MultiBoxLoss(conf)
optimizer = torch.optim.SGD(params = model.parameters(), momentum=train_conf['momentum'], weight_decay = train_conf['weight_decay'], lr = train_conf['lr'])

model.train()
for epoch in range(train_conf['num_epochs']):
    num_itrs = 1
    t_all, t_alc, t_oll, t_olc, t_loss = 0,0,0,0,0
    for x, y in dl:
        optimizer.zero_grad()
        al, ac, ol, oc = model(x)
        arm_loss_loc, arm_loss_conf = criterion(al,ac,ol,oc, y, mode='ARM')
        odm_loss_loc, odm_loss_conf = criterion(al,ac,ol,oc, y, mode='ODM')
        arm_loss = arm_loss_loc + arm_loss_conf
        odm_loss = odm_loss_loc + odm_loss_conf
        loss = arm_loss + odm_loss

        t_all += arm_loss_loc.data
        t_alc += arm_loss_conf.data
        t_oll += odm_loss_loc.data
        t_olc += odm_loss_conf.data
        t_loss += loss.data

        #if num_itrs % 10 == 0:
        print("[E: %03d][I: %05d] LOSS: %.4f(ALL: %.4f, ACL: %.4f, OLL: %.4f, OCL: %.4f)" 
                    %(epoch, num_itrs, t_loss/num_itrs, t_all/num_itrs, t_alc/num_itrs, t_oll/num_itrs, t_olc/num_itrs))

        loss.backward()
        optimizer.step()
        num_itrs += 1

