import argparse

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import json
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import provider
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.dirname(ROOT_DIR))
#import part_seg_model as model
import part_seg_model_multi_gap_deformoff_k3_new as model
from matplotlib import pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,ConfusionMatrixDisplay

#~~~~~~~~~~~~~~Configure Tensorflow GPU Options~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#Limit the model to use only a single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
#BASE_DIR2= 
results_dir='/app/plots'
os.makedirs(results_dir, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model_path', default='/results/epoch_145.ckpt', help='Model checkpoint path')
FLAGS = parser.parse_args()
#results_dir=os.path.join(ROOT_DIR,'CM')

hdf5_data_dir = os.path.join(BASE_DIR, 'wheat_comb_r')

# DEFAULT SETTINGS
#pretrained_model_path = FLAGS.model_path  # os.path.join(BASE_DIR, './pretrained_model/model.ckpt')
pretrained_model_path = os.path.join(BASE_DIR, 'results/epoch_145.ckpt')

print(pretrained_model_path)

#hdf5_data_dir = os.path.join(BASE_DIR2, 'wheat_comb_w')
gpu_to_use = FLAGS.gpu
output_verbose = True   # If true, output all color-coded part segmentation obj files
output_dir = os.path.join(BASE_DIR, 'test_results_deform')
# MAIN SCRIPT
point_num = 8192            # the max number of points in the all testing data shapes
batch_size = 1

testing_file_list=os.path.join(hdf5_data_dir,'test_hdf5_file_list.txt')
test_file_list=provider.getDataFiles(testing_file_list)
num_test_file=len(test_file_list)

NUM_OBJ_CATS = 1
NUM_PART_CATS = 2
def printout(flog, data):
  print(data)
  flog.write(data + '\n')

def placeholder_inputs():
    pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, point_num, 3))
    input_label_ph = tf.placeholder(tf.float32, shape=(batch_size, NUM_OBJ_CATS))
    labels_ph = tf.placeholder(tf.int32, shape=(batch_size))
    seg_ph = tf.placeholder(tf.int32, shape=(batch_size, point_num))
    return pointclouds_ph, input_label_ph, labels_ph, seg_ph

def convert_label_to_one_hot(labels):
    label_one_hot = np.zeros((labels.shape[0], NUM_OBJ_CATS))
    for idx in range(labels.shape[0]):
        label_one_hot[idx, labels[idx]] = 1
    return label_one_hot

bn_decay=None
def predict():
  is_training = False
  
  with tf.device('/gpu:'+str(gpu_to_use)):
    pointclouds_ph, input_label_ph,labels_ph, seg_ph = placeholder_inputs()
    is_training_ph = tf.placeholder(tf.bool, shape=())

    seg_pred,_,_ = model.get_model(pointclouds_ph, input_label_ph, \
        cat_num=NUM_OBJ_CATS, part_num=NUM_PART_CATS, is_training=is_training_ph, \
        batch_size=batch_size, num_point=point_num, weight_decay=0.0, bn_decay=None)
    
  saver = tf.train.Saver()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True

  with tf.Session(config=config) as sess:
    if not os.path.exists(output_dir):
      os.mkdir(output_dir)

    flog = open(os.path.join(output_dir, 'log.txt'), 'w')

    printout(flog, 'Loading model %s' % pretrained_model_path)
    saver.restore(sess, pretrained_model_path)
    printout(flog, 'Model restored.')
    
    total_acc = 0.0
    total_seen=0
    total_iou0=0.0
    total_iou1=0.0
    total_io=0.0
    all_preds = []
    all_gts = []

    for i in range(num_test_file):
            print(num_test_file)
            cur_test_filename=os.path.join(hdf5_data_dir,test_file_list[i])
            cur_data,cur_labels,cur_seg=provider.loadDataFile_with_seg(cur_test_filename)
            cur_labels=np.squeeze(cur_labels)
            cur_labels_one_hot=convert_label_to_one_hot(cur_labels)
            num_data=len(cur_labels)  
            num_batch=num_data//batch_size


            #go through each batch and predict result
            for j in range(num_batch):
                begidx = j * batch_size
                endidx = (j + 1) * batch_size
                feed_dict = {pointclouds_ph: cur_data[begidx: endidx, ...],input_label_ph: cur_labels_one_hot[begidx: endidx, ...],is_training_ph: is_training}
                
                pred_seg_res= sess.run([seg_pred],feed_dict=feed_dict)
               
                pred_seg_res=np.asarray(pred_seg_res)
                pred_seg_res=pred_seg_res.reshape(1,point_num,NUM_PART_CATS)
                pred_seg_val=np.argmax(pred_seg_res,axis=2)
                cur_seg_val=cur_seg[begidx:endidx,...]

                mask = np.int32(pred_seg_val == cur_seg_val)
                all_preds.extend(pred_seg_val.flatten())
                all_gts.extend(cur_seg_val.flatten())

                #accuracy
                sec_acc=np.mean(pred_seg_val==cur_seg_val)
                print('seg acc',sec_acc)
                printout(flog, 'seg accuracy is: %f' % sec_acc)
                total_acc +=sec_acc
                total_seen +=1
                print(cur_data.shape,point_num)
                #IOU calculations
                for oid in range(2):
                    total_iou=0.0

                    n_pred=np.sum(pred_seg_val==oid)
                    n_gt=np.sum(cur_seg_val==oid)
                    n_intersect=np.sum(np.int32(cur_seg_val==oid)*mask)
                    n_union=n_pred+ n_gt - n_intersect
                    if n_union==0:
                        total_iou=1
                    else:
                        total_iou=n_intersect*1.0/n_union
                        total_io=total_io+ total_iou
                        #print('total',total_io)
 
                     #print('iou for seg {} is:{}'.format(oid,total_iou))
                    printout(flog, 'iou for seg {} is:{}'.format(oid,total_iou))
                    #print('total_io',total_io)
                    if oid==0:
                        total_iou0=total_iou+total_iou0
                    else:
                        total_iou1=total_iou+total_iou1
                 #seg plotting
                '''
                color=['g','r','c','m']
                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 12))
                for i in range(point_num):
                    cd=None
                    x=cur_data[begidx: endidx,i,0]
                    y=cur_data[begidx: endidx,i,1]
                    z=cur_data[begidx: endidx,i,2]
                    seg=int(pred_seg_val[:,i])
                    ori_seg=int(cur_seg_val[:,i])
                    if seg==ori_seg:
                        cd=2
                    else:
                        cd=3
                    ax[0].scatter(x,y,c=color[ori_seg],s=3)
                    ax[1].scatter(x,y,c=color[seg],s=3)
                    ax[2].scatter(x,y,c=color[cd],s=3)

                    ax[0].set_title('ground truth')
                    ax[1].set_title('prediction')
                    ax[2].set_title('difference')
                

    
                plt.savefig(os.path.join(results_dir, f'result_plot_{j}.png'))
                plt.clf()
                print('plot saved')
            '''
            avg_iou0=total_iou0/total_seen
            avg_iou1=total_iou1/total_seen
            avg_iou=total_io/(total_seen*2)
            #print('avg iou for seg 0 is {}'.format(avg_iou0))
            #print('avg iou for seg 1 is {}'.format(avg_iou1))
            #print('total files seen are',total_seen)
            #print('avg accuracy is', total_acc/total_seen)
    #printout(flog,'avg iou  {}'.format(total_io))
    print('avgiou',avg_iou)
    
           
    precision = precision_score(all_gts, all_preds, average='binary', pos_label=1, zero_division=0)
    recall = recall_score(all_gts, all_preds, average='binary', pos_label=1, zero_division=0)
    f1 = f1_score(all_gts, all_preds, average='binary', pos_label=1, zero_division=0)
    cm = confusion_matrix(all_gts, all_preds)
       
    precision=precision_score(all_gts, all_preds, average='macro', zero_division=0)
    recall=recall_score(all_gts, all_preds, average='macro', zero_division=0)
    f1=f1_score(all_gts, all_preds, average='macro', zero_division=0)
        
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)
    print('Confusion Matrix:\n', cm)
    cm_normalized = confusion_matrix(all_gts, all_preds, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title('Normalized Confusion Matrix')
    plt.savefig((os.path.join(results_dir, 'confusion_matrix_normalized_g_w.png')))
    plt.close()
'''
    printout(flog,'avg iou for seg 0 is {}'.format(avg_iou0))
    printout(flog,'avg iou for seg 1 is {}'.format(avg_iou1))
    printout(flog,'avg iou is {}'.format(avg_iou))
    printout(flog,'total files seen{}'.format(total_seen))
    printout(flog,'avg accuracy is{}'.format( total_acc/total_seen))
    
    '''
    printout(flog, 'Precision: %.5f' % precision)
    printout(flog, 'Recall: %.5f' % recall)
    printout(flog, 'F1-score: %.5f' % f1)
    printout(flog, 'Confusion Matrix:\n%s' % str(cm))
'''

    flog.close()            
               

       

with tf.Graph().as_default():
    predict()




                
                   
                    
                    
                    
                    

                    
                

