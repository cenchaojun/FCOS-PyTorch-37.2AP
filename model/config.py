class DefaultConfig():
    #backbone
    backbone = 'resnet50'  # vovnet39 or resnet50
    pretrained=True
    freeze_stage_1=True
    freeze_bn= True

    #fpn
    fpn_out_channels=256
    use_p5=True
    
    #head
    class_num=1
    use_GN_head=True
    prior=0.01
    add_centerness=True
    cnt_on_reg=True

    #training
    # strides=[4,8,16,32,64,128]
    # limit_range=[[-1,32],[32,64],[64,128],[128,256],[256,512],[512,999999]]
    #training
    strides=[8,16,32,64,128]
    limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]

    #inference
    score_threshold=0.5
    nms_iou_threshold=0.1
    max_detection_boxes_num=1000