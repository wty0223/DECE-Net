import torch
import cv2
import time
import os
import os.path as osp
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import ArgumentParser
from collections import OrderedDict
from IOUEval import iouEval
from PIL import Image
import loadData as ld
import DECE-Net as net
import cv2
import transforms as myTransforms
import dataset as myDataLoader
from torch.nn.parallel.scatter_gather import gather


@torch.no_grad()
def validate(args, model, val_loader,crossVal,image_list,label_list):
    iou_eval_val = iouEval(args.classes)
    total_batches = len(val_loader)
    for iter, (input, contour, target) in enumerate(val_loader):
        start_time = time.time()

        if args.gpu:
            input = input.cuda()
            contour = contour.cuda()
            target = target.cuda()
        input_var = torch.autograd.Variable(input)
        contour_var = torch.autograd.Variable(contour)
        target_var = torch.autograd.Variable(target)

        output = model(input_var,contour_var)

        torch.cuda.synchronize()
        time_taken = time.time() - start_time
        print('Segmentation for {}/{} takes {:.3f}s per image'.format(iter, len(image_list), time_taken))
        # compute the confusion matrix
        if args.gpu and torch.cuda.device_count() > 1:
            output = gather(output, 0, dim=0)[0]
        else:
            output = output[0]

      
        iou_eval_val.add_batch(output.max(1)[1].data.cpu().numpy(), target_var.data.cpu().numpy())
        out_numpy = (output.max(1)[1].data.cpu().numpy() * 255).astype(np.uint8)
        out_numpy=np.squeeze(out_numpy,0)


        name = image_list[iter].split('/')[-1]
        if not osp.isdir(osp.join(args.savedir, args.data_name)):
            os.mkdir(osp.join(args.savedir, args.data_name))
        if not osp.isdir(osp.join(args.savedir, args.data_name, args.model_name)):
            os.mkdir(osp.join(args.savedir, args.data_name, args.model_name))
        if not osp.isdir(osp.join(args.savedir, args.data_name, args.model_name, 'crossVal'+str(crossVal))):
            os.mkdir(osp.join(args.savedir, args.data_name, args.model_name, 'crossVal'+str(crossVal)))
        cv2.imwrite(osp.join(args.savedir, args.data_name, args.model_name, 'crossVal'+str(crossVal), name[:-4] + '.png'), out_numpy)

    overall_acc, per_class_acc, per_class_iu, mIOU = iou_eval_val.get_metric()
    print('Overall Acc (Val): %.4f\t mIOU (Val): %.4f' % (overall_acc, mIOU))
    return mIOU




def main_te(args, crossVal, pretrained):
    image_list = list()
    label_list = list()
    with open(osp.join(args.data_dir, 'COVID-19-' + args.data_name + '/dataList/'+'val'+str(crossVal)+'.txt')) as text_file:
        for line in text_file:
            line_arr = line.split()
            image_list.append(osp.join(args.data_dir, line_arr[0].strip()))
            label_list.append(osp.join(args.data_dir, line_arr[1].strip()))
    dataLoad = ld.LoadData(args.data_dir, args.classes)
    data = dataLoad.processData(crossVal, args.data_name)

    model = net.Net(args.classes, aux=False)
    if not osp.isfile(pretrained):
        print('Pre-trained model file does not exist...')
        exit(-1)
    state_dict = torch.load(pretrained)
    #'''
    new_keys = []
    new_values = []
    for key, value in zip(state_dict.keys(), state_dict.values()):
        if 'pred' not in key or 'pred1' in key:
            new_keys.append(key)
            new_values.append(value)
    new_dict = OrderedDict(list(zip(new_keys, new_values)))
    model.load_state_dict(new_dict)
    #'''
    #model.load_state_dict(state_dict,True)

    if args.gpu:
        model = model.cuda()

    # set to evaluation mode
    model.eval()

    if not osp.isdir(args.savedir):
        os.mkdir(args.savedir)

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(args.width, args.height),
        myTransforms.ToTensor()
    ])
    valLoader = torch.utils.data.DataLoader(
        myDataLoader.Dataset(data['valIm'], data['valAnnot'], transform=valDataset),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    mIOU = validate(args, model, valLoader,crossVal,image_list,label_list)
    return mIOU


def main(args):
    crossVal = 5
    maxEpoch = [51,62,45,78,72]#
    mIOUList = []
    avgmIOU = 0

    for i in range(crossVal):

        pthName = 'model_' + args.model_name + '_crossVal' + str(i+1) + '_' + str(maxEpoch[i]) + '.pth'
        pretrainedModel = args.pretrained + args.data_name + '/' + args.model_name + '/' + pthName
        #pretrainedModel = args.pretrained + args.data_name + '/' + 'MiniSeg' + '/' + pthName
        mIOU = "{:.4f}".format(main_te(args, i, pretrainedModel))
        mIOU = float(mIOU)
        mIOUList.append(mIOU)
        avgmIOU = avgmIOU + mIOU/5
    print(mIOUList)
    print(args.model_name, args.data_name, "{:.4f}".format(avgmIOU))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="./datasets", help='Data directory')
    parser.add_argument('--width', type=int, default=512, help='Width of RGB image')
    parser.add_argument('--height', type=int, default=512, help='Height of RGB image')
    parser.add_argument('--savedir', default='./outputs', help='directory to save the results')
    parser.add_argument('--model_name', default='DECE-Net', help='Model name')
    parser.add_argument('--data_name', default='CT00', help='Model name')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU')
    parser.add_argument('--pretrained', default='./new_results_crossVal_mod80/', help='Pretrained model')
    parser.add_argument('--classes', default=2, type=int, help='Number of classes in the dataset')

    args = parser.parse_args()
    print('Called with args:')
    print(args)
    main(args)
