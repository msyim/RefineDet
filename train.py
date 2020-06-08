from base_models import VGG16
import torch

## test
from utils import SSDAugmentation, VOCDetection, VOCAnnotationTransform, detection_collate, get_anchors, BaseTransform
from layers import MultiBoxLoss
from test import test_net
import torch.utils.data as data

from RefineDet import refine_det

import json
import sys

torch.autograd.set_detect_anomaly(True)

conf = json.loads(open('conf.json').read())
bbox_conf  = conf['bbox_conf']
train_conf = conf['train_conf']
dset_conf  = conf['dataset_conf']
backbone = VGG16()
model = refine_det(backbone, conf).cuda()
#model.weight_initialize()
print(model)

if train_conf['trained_weights'] != '':
    model.load_state_dict(torch.load(train_conf['trained_weights']))

dataset = VOCDetection(root=dset_conf['dataset_root'], transform=SSDAugmentation(512))
testset = VOCDetection(dset_conf['dataset_root'], [('2007', 'test')],
                           BaseTransform(bbox_conf['image_size'], mean = (104, 117, 123)),
                           VOCAnnotationTransform())
dl = data.DataLoader(dataset = dataset, collate_fn=detection_collate, batch_size=13, shuffle=True)
criterion = MultiBoxLoss(conf)
optimizer = torch.optim.SGD(params = model.parameters(), momentum=train_conf['momentum'], weight_decay = train_conf['weight_decay'], lr = train_conf['lr'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, eps=1e-8)

max_map = 0.0

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
        loss.backward()
        optimizer.step()

        sys.stdout.write("\r[E: %03d][I: %05d] LOSS: %.4f(ALL: %.4f, ACL: %.4f, OLL: %.4f, OCL: %.4f)" 
                    %(epoch, num_itrs, t_loss/num_itrs, t_all/num_itrs, t_alc/num_itrs, t_oll/num_itrs, t_olc/num_itrs))    
        num_itrs += 1

    sys.stdout.write('\n')
    model.eval()
    with torch.no_grad():
        MAP = test_net(model, testset, anchors=get_anchors(bbox_conf))
        if MAP > max_map:
            max_map = MAP
            torch.save(model.state_dict(), './saved_models/best_accuracy.pth')
        print('Best MAP: %f, cur MAP: %f' % (max_map, MAP))
    model.train()
    scheduler.step(t_loss)


