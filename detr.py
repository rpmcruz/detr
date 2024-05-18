import torch, torchvision
torch.set_float32_matmul_precision('high')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

########################################### MODEL ###########################################

def Backbone():
    return torch.nn.Sequential(*list(torchvision.models.resnet50(weights='DEFAULT').children())[:-2])

class DETR(torch.nn.Module):
    # https://github.com/facebookresearch/detr
    def __init__(self, num_classes, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        # num_classes: notice that class=0 is background (no-class)
        # num_objects must be bigger than hidden features (32 for resnet50, default is 100)
        super().__init__()
        self.conv = torch.nn.LazyConv2d(hidden_dim, 1)
        self.transformer = torch.nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, batch_first=True)
        self.classes = torch.nn.Linear(hidden_dim, num_classes)
        self.bboxes = torch.nn.Linear(hidden_dim, 4)
        self.query_pos = torch.nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = torch.nn.Parameter(torch.rand(100//2, hidden_dim//2))
        self.col_embed = torch.nn.Parameter(torch.rand(100//2, hidden_dim//2))
        self.nheads = nheads

    def forward(self, x):
        h = self.conv(x)
        N, _, H, W = h.shape
        pos = torch.cat([
            self.col_embed[None, :W].repeat(H, 1, 1),
            self.row_embed[:H, None].repeat(1, W, 1),
        ], -1).flatten(0, 1)[None]
        h = self.transformer(pos + h.flatten(2).permute(0, 2, 1), self.query_pos[None].repeat(N, 1, 1))
        # user should apply softmax
        classes = self.classes(h)
        # predicted bounding boxes are in cxcywh format => convert to xyxy
        bboxes = torch.sigmoid(self.bboxes(h))
        bboxes = torch.stack((
            bboxes[..., 0] - bboxes[..., 2]/2,
            bboxes[..., 1] - bboxes[..., 3]/2,
            bboxes[..., 0] + bboxes[..., 2]/2,
            bboxes[..., 1] + bboxes[..., 3]/2,
        ), -1)
        return bboxes, classes

########################################### LOSS ###########################################

def detr_loss_per_batch(pred_bboxes: torch.Tensor, pred_classes: torch.Tensor, true_bboxes: torch.Tensor, true_classes: torch.Tensor):
    device = pred_bboxes.device
    lambda_iou = 5  # default parameter values from the paper
    lambda_l1 = 2
    # all loss permutations between predicted and true objects
    # the last true bbox is the "no object"
    n_pred = len(pred_bboxes)
    n_true = 1 + min(len(true_bboxes), len(pred_bboxes))
    # repeat preds and trues in a different fashion so they match
    pred_bboxes = torch.repeat_interleave(pred_bboxes, n_true, 0)
    pred_classes = torch.repeat_interleave(pred_classes, n_true, 0)
    true_bboxes = torch.cat((true_bboxes, torch.zeros(1, 4, device=device)), 0).repeat(n_pred, 1)
    true_classes = torch.cat((true_classes, torch.zeros(1, dtype=torch.int64, device=device)), 0).repeat(n_pred)
    real_bboxes = torch.cat((torch.ones(n_true-1, device=device), torch.zeros(1, device=device))).repeat(n_pred)
    # compute losses
    loss_iou = torchvision.ops.generalized_box_iou_loss(pred_bboxes, true_bboxes, reduction='none')
    loss_l1 = torch.mean(torch.abs(pred_bboxes - true_bboxes), 1)
    loss_class = torch.nn.functional.cross_entropy(pred_classes, true_classes, reduction='none')
    total_losses = loss_class + real_bboxes*(lambda_iou*loss_iou + lambda_l1*loss_l1)
    total_losses = torch.reshape(total_losses, (n_pred, n_true))
    # Hungarian algorithm: find the minimum matches
    # quick and dirty solution
    final_loss = 0
    no_objects_included = True
    for _ in range(n_pred):
        if no_objects_included and total_losses.shape[0] <= total_losses.shape[1]-1:
            # if we still have true objects to match, focus on these objects
            no_objects_included = False
            total_losses = total_losses[:, :-1].contiguous()
        loss, i = torch.min(total_losses.view(-1), dim=0)
        final_loss += loss
        # remove predicted row
        pred_i = i // total_losses.shape[1]
        total_losses = torch.cat((total_losses[:pred_i], total_losses[pred_i+1:]), 0)
        # remove true column (only if not "no object")
        true_i = i % total_losses.shape[1]
        if not no_objects_included or true_i < total_losses.shape[1]-1:
            total_losses = torch.cat((total_losses[:, :true_i], total_losses[:, true_i+1:]), 1)
        total_losses = total_losses.contiguous()
    return final_loss

def detr_loss(pred_bboxes, pred_classes, true_bboxes, true_classes):
    return sum(detr_loss_per_batch(pb, pc, tb, tc) for pb, pc, tb, tc in zip(pred_bboxes, pred_classes, true_bboxes, true_classes)) / len(pred_bboxes)

########################################### DATASET ###########################################

class VOC:
    labels = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
        'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
        'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor',
    ]
    def __init__(self, root, train, transform):
        image_set = 'train' if train else 'val'
        self.ds = torchvision.datasets.VOCDetection(root, image_set=image_set)
        self.transform = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        image, xml = self.ds[i] 
        w = int(xml['annotation']['size']['width'])
        h = int(xml['annotation']['size']['height'])
        bboxes = torchvision.tv_tensors.BoundingBoxes(torch.tensor([(
            float(object['bndbox']['xmin']), float(object['bndbox']['ymin']),
            float(object['bndbox']['xmax']), float(object['bndbox']['ymax']),
        ) for object in xml['annotation']['object']]), format='XYXY', canvas_size=(h, w))
        # note that classes start in 1 (0 is background)
        classes = torch.tensor([self.labels.index(object['name'])+1 for object in xml['annotation']['object']])
        if self.transform:
            image, bboxes = self.transform(image, bboxes)
        # normalize bboxes to 0-1
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / bboxes.canvas_size[1]
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / bboxes.canvas_size[0]
        return image, bboxes, classes

def my_collate(batch):
    return torch.stack([i[0] for i in batch], 0), [i[1] for i in batch], [i[2] for i in batch]

from torchvision.transforms import v2
aug = v2.Compose([
    v2.PILToTensor(),
    v2.Resize((int(224*1.1), int(224*1.1))),
    v2.RandomCrop((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(0.2, 0.2),
    v2.ToDtype(torch.float32, True),
])

ds = VOC('/data/toys', True, aug)
ds = torch.utils.data.DataLoader(ds, 8, True, num_workers=4, pin_memory=True, collate_fn=my_collate)

########################################### TRAIN ###########################################

# we instantiate the backbone and the main model separately to use different learning rates, like the paper
backbone = Backbone().to(device)
model = DETR(len(VOC.labels)+1).to(device)
model.train()
model_opt = torch.optim.AdamW(model.parameters(), 1e-4, weight_decay=1e-4)
backbone_opt = torch.optim.AdamW(backbone.parameters(), 1e-5, weight_decay=1e-4)
model.backbone = backbone

from time import time
from tqdm import tqdm
epochs = 300
drop_lr_epoch = 200
for epoch in range(epochs):
    tic = time()
    for images, true_bboxes, true_classes in tqdm(ds):
        images = images.to(device)
        true_bboxes = [t.to(device) for t in true_bboxes]
        true_classes = [t.to(device) for t in true_classes]
        pred_bboxes, pred_classes = model(backbone(images))
        loss = detr_loss(pred_bboxes, pred_classes, true_bboxes, true_classes)
        model_opt.zero_grad()
        backbone_opt.zero_grad()
        loss.backward()
        model_opt.step()
        backbone_opt.step()
    if epoch == drop_lr_epoch:
        for opt in [model_opt, backbone_opt]:
            for param_group in opt.param_groups:
                param_group['lr'] /= 10
    toc = time()
    print(f'Epoch {epoch+1}/{epochs} - {toc-tic:.0f}s - Loss: {loss}')
    torch.save(model, 'model.pth')
    # evaluate
    import matplotlib.pyplot as plt
    from matplotlib import patches
    plt.clf()
    plt.imshow(images[0].permute(1, 2, 0).cpu())
    h, w = images[0].shape[1:]
    for (x1, y1, x2, y2), label in zip(true_bboxes[0].cpu(), true_classes[0].cpu()):
        plt.gca().add_patch(patches.Rectangle((x1*w, y1*h), (x2-x1)*w, (y2-y1)*h, linewidth=1, edgecolor='g', facecolor='none'))
        plt.text(x1*w, y1*h, VOC.labels[label-1], c='g')
    for (x1, y1, x2, y2), label, prob in zip(pred_bboxes[0].cpu().detach(), pred_classes[0].argmax(1).cpu().detach(), torch.softmax(pred_classes[0], 1).amax(1).cpu().detach()):
        if label == 0: continue
        plt.gca().add_patch(patches.Rectangle((x1*w, y1*h), (x2-x1)*w, (y2-y1)*h, linewidth=1, edgecolor='r', facecolor='none'))
        plt.text(x1*w, y1*h, f'{VOC.labels[label-1]} {100*prob:.0f}%', c='r')
    plt.title(f'Epoch {epoch+1}')
    plt.savefig(f'result-{epoch+1}.png')