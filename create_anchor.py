import os 


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "2"


import tensorflow as tf 
import numpy as np 

from np_anchors import create_target_assigner, batch_assign_targets, Anchor
from object_detection import np_box_list as box_list
from retinanet import retinanet


# format: ymin,xmin, ymax, xmax
test_gt_boxes_list= [[
    [0.1, 0.1, 0.2,0.2],
    [0.3,0.3, 0.7,0.7],
    [0.8,0.8,0.9,0.9]
] ]

test_gt_labels_list= [ [
    1,2
] ]

test_gt_boxes_list = np.array(test_gt_boxes_list,dtype=np.float32)
test_gt_labels_list = np.array(test_gt_labels_list, dtype=np.int32)





def _get_feature_map_shape(feature_map_list):
    output=[]
    for feature_map in feature_map_list:
        print(feature_map)
        shape = feature_map.get_shape().as_list()
        print(shape)
        output.append( (shape[1], shape[2]) )

    return output

def _assign_targets(gt_boxes_list, gt_labels_list, target_assigner, anchors):
        """
        Assign gt targets
        Args:
             gt_boxes_list: a list of 2-D tensor of shape [num_boxes, 4] containing coordinates of gt boxes
             gt_labels_list: a list of 2-D one-hot tensors of shape [num_boxes, num_classes] containing gt classes
        Returns:
            batch_cls_targets: class tensor with shape [batch_size, num_anchors, num_classes]
            batch_reg_target: box tensor with shape [batch_size, num_anchors, 4]
            match_list: a list of matcher.Match object encoding the match between anchors and gt boxes for each image
                        of the batch, with rows corresponding to gt-box and columns corresponding to anchors
        """
        gt_boxlist_list = [box_list.BoxList(boxes) for boxes in gt_boxes_list]
        # gt_labels_with_bg = [tf.pad(gt_class, [[0, 0], [1, 0]], mode='CONSTANT')
        #                       for gt_class in gt_labels_list]
        anchors_boxlist = box_list.BoxList(anchors)
        print("inside _assign_targets, anchors_boxlist data={}".format(anchors_boxlist.get()) )
        return batch_assign_targets(target_assigner,anchors_boxlist,gt_boxlist_list,gt_labels_list)


image_shape=(224,224)

num_classes=2
num_scales = 2
aspect_ratios = (1.0, 2.0, 0.5)
anchor_scale = 4.0




num_anchors_per_loc = num_scales * len(aspect_ratios)

inputs = tf.placeholder(tf.float32, shape=(1,224,224,3))

prediction_dict = retinanet(inputs, num_classes, num_anchors_per_loc, is_training=True)



feature_map_shape_list = _get_feature_map_shape(prediction_dict["feature_map_list"])
anchor_generator = Anchor(feature_map_shape_list=feature_map_shape_list,
                                img_size=image_shape,
                                anchor_scale=anchor_scale,
                                aspect_ratios=aspect_ratios,
                                scales_per_octave=num_scales)

anchors = anchor_generator.boxes

print("anchors shape:{}".format(anchors.shape))

unmatched_class_label = tf.constant((num_classes + 1) * [0], tf.float32)
target_assigner = create_target_assigner(unmatched_cls_target=unmatched_class_label)



_assign_targets(test_gt_boxes_list, test_gt_labels_list, target_assigner, anchors)