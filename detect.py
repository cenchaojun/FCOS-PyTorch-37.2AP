import cv2
from model.fcos import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
from dataset.VOC_dataset import VOCDataset
import time
import matplotlib.patches as patches
import  matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_img(image,input_ksize):
    '''
    resize image and bboxes 
    Returns
    image_paded: input_ksize  
    bboxes: [None,4]
    '''
    min_side, max_side    = input_ksize
    h,  w, _  = image.shape

    smallest_side = min(w,h)
    largest_side=max(w,h)
    scale=min_side/smallest_side
    if largest_side*scale>max_side:
        scale=max_side/largest_side
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    pad_w=32-nw%32
    pad_h=32-nh%32

    image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
    image_paded[:nh, :nw, :] = image_resized
    return image_resized
    
def convertSyncBNtoBN(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    for name, child in module.named_children():
        module_output.add_module(name,convertSyncBNtoBN(child))
    del module
    return module_output
if __name__=="__main__":
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    class Config():
        #backbone
        pretrained=False
        freeze_stage_1=True
        freeze_bn=True

        #fpn
        fpn_out_channels=256
        use_p5=True
        
        #head
        # 之前是80个类别,现在改为1个了
        class_num=1
        use_GN_head=True
        prior=0.01
        add_centerness=True
        cnt_on_reg=False

        #training
        strides=[8,16,32,64,128]
        limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]

        #inference
        score_threshold=0.4
        nms_iou_threshold=0.1
        max_detection_boxes_num=600

    model=FCOSDetector(mode="inference",config=Config)
    # model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # print("INFO===>success convert BN to SyncBN")
    # 如果遇到了权重不匹配，需要把并行训练这个模式给关掉
    # model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("./fewcheckpoint/model_30.pth",map_location=torch.device('cpu')))
    # model=convertSyncBNtoBN(model)
    # print("INFO===>success convert SyncBN to BN")
    model=model.cuda().eval()
    print("===>success loading model")

    import os
    root="/home/cen/PycharmProjects/dataset/20201203dataset/crop512valdatasetimage/"
    names=os.listdir(root)
    for name in names:
        img_bgr=cv2.imread(root+name)
        img_pad=preprocess_img(img_bgr,[512,512])
        img=cv2.cvtColor(img_pad.copy(),cv2.COLOR_BGR2RGB)
        img1=transforms.ToTensor()(img)
        img1= transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225],inplace=True)(img1)
        img1=img1.cuda()
        

        start_t=time.time()
        with torch.no_grad():
            out=model(img1.unsqueeze_(dim=0))
        end_t=time.time()
        cost_t=1000*(end_t-start_t)
        print("===>success processing img, cost time %.2f ms"%cost_t)
        # print(out)
        scores,classes,boxes=out

        boxes=boxes[0].cpu().numpy().tolist()
        classes=classes[0].cpu().numpy().tolist()
        scores=scores[0].cpu().numpy().tolist()
        # 这个是源代码写的，有点问题，不方便检查，所以要自己重新写一下
        for i,box in enumerate(boxes):
            pt1=(int(box[0]),int(box[1]))
            pt2=(int(box[2]),int(box[3]))
            # TODO 需要将text的大小降低下来，要不然太大了，不好看
            # TODO 再次检查一些
            img_pad=cv2.rectangle(img_pad,pt1,pt2,(0,255,0),thickness=3)
            textLabel = "%s %.3f"%(VOCDataset.CLASSES_NAME[int(classes[i])],scores[i])
            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,0.4,1)
            textOrg = (int(box[0]), int(box[1]) )
            cv2.rectangle(img_pad, (textOrg[0] - 2, textOrg[1]+baseLine - 2), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 255, 0), 1)
            cv2.rectangle(img_pad, (textOrg[0] - 2,textOrg[1]+baseLine - 2), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 255, 0), -1)
            cv2.putText(img=img_pad,text=textLabel,org=textOrg,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4,color=(0, 0, 0),thickness=1)
            # cv2.putText(img_pad, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
        cv2.imwrite('./fewcrop512test/{}'.format(name),img_pad)
        #
        # plt.figure()
        # fig, ax = plt.subplots(1)
        # ax.imshow(img)
        # for i,box in enumerate(boxes):
        #     pt1=(int(box[0]),int(box[1]))
        #     pt2=(int(box[2]),int(box[3]))
        #     img_pad=cv2.rectangle(img_pad,pt1,pt2,(0,255,0))
        #     b_color = colors[int(classes[i]) - 1]
        #     bbox = patches.Rectangle((box[0],box[1]),width=box[2]-box[0],height=box[3]-box[1],linewidth=1,facecolor='none',edgecolor=b_color)
        #     ax.add_patch(bbox)
        #     plt.text(box[0], box[1], s="%s %.3f"%(VOCDataset.CLASSES_NAME[int(classes[i])],scores[i]), color='white',
        #              verticalalignment='top',
        #              bbox={'color': b_color, 'pad': 0})
        # plt.axis('off')
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        # plt.savefig('./out_put_60epoch/{}'.format(name), bbox_inches='tight', pad_inches=0.0)
        # plt.close()





