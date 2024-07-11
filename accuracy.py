# Copyright (c) OpenMMLab. All rights reserved.
import os
import numpy as np
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_score,
    recall_score, f1_score, roc_curve, accuracy_score,
    log_loss, precision_recall_curve, average_precision_score
)
from datetime import datetime
import matplotlib.pyplot as plt

diganostic_save_path = "C:/Users/u251245/CVEpilepsy/test_figures_aidan/diagnostics/"
image_save_path = "C:/Users/u251245/CVEpilepsy/test_figures_aidan/"

def confusion_matrix(y_pred, y_real, normalize=None):
    """Compute confusion matrix.

    Args:
        y_pred (list[int] | np.ndarray[int]): Prediction labels.
        y_real (list[int] | np.ndarray[int]): Ground truth labels.
        normalize (str | None): Normalizes confusion matrix over the true
            (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized. Options are
            "true", "pred", "all", None. Default: None.

    Returns:
        np.ndarray: Confusion matrix.
    """
    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
        if y_pred.dtype == np.int32:
            y_pred = y_pred.astype(np.int64)
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(
            f'y_pred must be list or np.ndarray, but got {type(y_pred)}')
    if not y_pred.dtype == np.int64:
        raise TypeError(
            f'y_pred dtype must be np.int64, but got {y_pred.dtype}')

    if isinstance(y_real, list):
        y_real = np.array(y_real)
        if y_real.dtype == np.int32:
            y_real = y_real.astype(np.int64)
    if not isinstance(y_real, np.ndarray):
        raise TypeError(
            f'y_real must be list or np.ndarray, but got {type(y_real)}')
    if not y_real.dtype == np.int64:
        raise TypeError(
            f'y_real dtype must be np.int64, but got {y_real.dtype}')

    label_set = np.unique(np.concatenate((y_pred, y_real)))
    num_labels = len(label_set)
    max_label = label_set[-1]
    label_map = np.zeros(max_label + 1, dtype=np.int64)
    for i, label in enumerate(label_set):
        
        label_map[label] = i

    y_pred_mapped = label_map[y_pred]
    y_real_mapped = label_map[y_real]

    confusion_mat = np.bincount(
        num_labels * y_real_mapped + y_pred_mapped,
        minlength=num_labels**2).reshape(num_labels, num_labels)

    with np.errstate(all='ignore'):
        if normalize == 'true':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=1, keepdims=True))
        elif normalize == 'pred':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=0, keepdims=True))
        elif normalize == 'all':
            confusion_mat = (confusion_mat / confusion_mat.sum())
        confusion_mat = np.nan_to_num(confusion_mat)

    return confusion_mat


def mean_class_accuracy(scores, labels):
    """Calculate mean class accuracy.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.

    Returns:
        np.ndarray: Mean class accuracy.
    """
    pred = np.argmax(scores, axis=1)
    cf_mat = confusion_matrix(pred, labels).astype(float)

    cls_cnt = cf_mat.sum(axis=1)
    cls_hit = np.diag(cf_mat)

    mean_class_acc = np.mean(
        [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)])

    return mean_class_acc


def modified_auc(scores, labels):
    roc_points = []
    for threshold in np.arange(0, 1.0001, 0.0001):
        rates = perf_metrics(scores, labels, threshold)
        roc_points.append(rates)
    
    roc_points = np.array(roc_points)
    fpr_array = roc_points[:, 0]
    tpr_array = roc_points[:, 1]
    
    auc = np.trapz(tpr_array, x=fpr_array)
    
    return auc



# anthony's original
# def top_k_accuracy(scores, labels, topk=(1, )):
#     """Calculate top k accuracy score.

#     Args:
#         scores (list[np.ndarray]): Prediction scores for each class.
#         labels (list[int]): Ground truth labels.
#         topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

#     Returns:
#         list[float]: Top k accuracy score for each k.
#     """
#     pred = np.argmax(scores, axis=1)
#     print(f"\nRaw scores:\t{scores}\n\n\nLabels:\t{labels}\n\n\nPred:\t{pred}\n\n\nRaw Size:\t{np.size(scores)}\n\n\nLabel Size:\t{np.size(labels)}\n\n\nPred Size:\t{np.size(pred)}\n")
#     return accuracy_score(labels, pred)

def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    # print(f"Labels: {labels}\nScores Arrays: {scores}")
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        # print(f"Prediction: {max_k_preds}")
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        #print(f"\nRaw scores:\t{scores}\n\n\nLabels:\t{labels}\n\n\nPred:\t{max_k_preds}\n\n\nRaw Size:\t{np.size(scores)}\n\n\nLabel Size:\t{np.size(labels)}\n\n\nPred Size:\t{np.size(max_k_preds)}\n")

        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res
    
def perf_metrics(scores, labels, threshold):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    # print(f"\nScores:\n{scores}")
    positive_scores = [row[1] for row in scores]
    pred=[0 if x<threshold else 1 for x in positive_scores]
    
    #We find the True positive rate and False positive rate based on the threshold
    start = 0
    
    vid = {}
    for count in range(len(labels)-1):
        if labels[count] == 0 and labels[count+1] == 1:
            vid[(count, 0)] = {'labels': labels[start:count+1], 'scores': pred[start:count+1]}
            start = count + 1
        if count+1 == (len(labels)-1) and labels[count+1] == 1:
            vid[(count, 1)] = {'labels': labels[start:count+1], 'scores': pred[start:count+1]}
        if labels[count] == 1 and labels[count+1] == 0:
            vid[(count, 1)] = {'labels': labels[start:count+1], 'scores': pred[start:count+1]}
            start = count + 1 

    for x in vid:
        misc, binary = x 
        
        tru = vid[x]['labels']
        pred_ = vid[x]['scores']
        
        if binary == 0:
            for x in range(len(tru)):
                if tru[x] == pred_[x]:
                    tn += 1 
                else:
                    fp += 1
        if binary == 1:
            for x in range(len(pred_)-1):
                """if tru[x] == pred_[x]:
                    tp += 1
                else:
                    fn += 1"""
                if pred_[x] == pred_[x+1] and pred_[x] == 1:
                    tp += 1
                    break
    if tp == 0:
        tpr = 0
    else:
        tpr = tp/(tp+fn)
    if fp == 0:
        fpr = 0
    else:
        fpr = fp/(tn+fp)

    return [fpr,tpr]


def roc_auc(scores, labels):
    pred = [item for sublist in scores for item in sublist]
    print(pred)
    print(labels)
    return roc_auc_score(labels, pred)


def modified_acc(scores, labels):
    pred = np.argmax(scores, axis=1)
    # pred = np.round([item for sublist in scores for item in sublist])
    
    start = 0
    
    vid = {}
    for count in range(len(labels)-1):
        if labels[count] == 0 and labels[count+1] == 1:
            vid[(count, 0)] = {'labels': labels[start:count+1], 'scores': pred[start:count+1]}
            start = count + 1
        if count+1 == (len(labels)-1) and labels[count+1] == 1:
            vid[(count, 1)] = {'labels': labels[start:count+1], 'scores': pred[start:count+1]}
        if labels[count] == 1 and labels[count+1] == 0:
            vid[(count, 1)] = {'labels': labels[start:count+1], 'scores': pred[start:count+1]}
            start = count + 1 
        
    correct = 0
    total = 0
    for x in vid:
        misc, binary = x 
        
        tru = vid[x]['labels']
        pred_ = vid[x]['scores']
        
        if binary == 0:
            for x in range(len(tru)):
                if tru[x] == pred_[x]:
                    correct += 1 
                total += 1 
        if binary == 1:
            for x in range(len(pred_)-2):
                if pred_[x] == pred_[x+1] and pred_[x+1] == pred_[x+2] and pred_[x] == 1:
                    correct += 1
                    break
            total +=1 
    
    return correct/total
            
        
            

def binary_precision_recall_curve(y_score, y_true):
    """Calculate the binary precision recall curve at step thresholds.

    Args:
        y_score (np.ndarray): Prediction scores for each class.
            Shape should be (num_classes, ).
        y_true (np.ndarray): Ground truth many-hot vector.
            Shape should be (num_classes, ).

    Returns:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.
        thresholds (np.ndarray): Different thresholds at which precision and
            recall are tested.
    """
    assert isinstance(y_score, np.ndarray)
    assert isinstance(y_true, np.ndarray)
    assert y_score.shape == y_true.shape

    # make y_true a boolean vector
    y_true = (y_true == 1)
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind='mergesort')[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    # There may be ties in values, therefore find the `distinct_value_inds`
    distinct_value_inds = np.where(np.diff(y_score))[0]
    threshold_inds = np.r_[distinct_value_inds, y_true.size - 1]
    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_inds]
    fps = 1 + threshold_inds - tps
    thresholds = y_score[threshold_inds]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def pairwise_temporal_iou(candidate_segments,
                          target_segments,
                          calculate_overlap_self=False):
    """Compute intersection over union between segments.

    Args:
        candidate_segments (np.ndarray): 1-dim/2-dim array in format
            ``[init, end]/[m x 2:=[init, end]]``.
        target_segments (np.ndarray): 2-dim array in format
            ``[n x 2:=[init, end]]``.
        calculate_overlap_self (bool): Whether to calculate overlap_self
            (union / candidate_length) or not. Default: False.

    Returns:
        t_iou (np.ndarray): 1-dim array [n] /
            2-dim array [n x m] with IoU ratio.
        t_overlap_self (np.ndarray, optional): 1-dim array [n] /
            2-dim array [n x m] with overlap_self, returns when
            calculate_overlap_self is True.
    """
    candidate_segments_ndim = candidate_segments.ndim
    if target_segments.ndim != 2 or candidate_segments_ndim not in [1, 2]:
        raise ValueError('Dimension of arguments is incorrect')

    if candidate_segments_ndim == 1:
        candidate_segments = candidate_segments[np.newaxis, :]

    n, m = target_segments.shape[0], candidate_segments.shape[0]
    t_iou = np.empty((n, m), dtype=np.float32)
    if calculate_overlap_self:
        t_overlap_self = np.empty((n, m), dtype=np.float32)

    for i in range(m):
        candidate_segment = candidate_segments[i, :]
        tt1 = np.maximum(candidate_segment[0], target_segments[:, 0])
        tt2 = np.minimum(candidate_segment[1], target_segments[:, 1])
        # Intersection including Non-negative overlap score.
        segments_intersection = (tt2 - tt1).clip(0)
        # Segment union.
        segments_union = ((target_segments[:, 1] - target_segments[:, 0]) +
                          (candidate_segment[1] - candidate_segment[0]) -
                          segments_intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments.
        t_iou[:, i] = (segments_intersection.astype(float) / segments_union)
        if calculate_overlap_self:
            candidate_length = candidate_segment[1] - candidate_segment[0]
            t_overlap_self[:, i] = (
                segments_intersection.astype(float) / candidate_length)

    if candidate_segments_ndim == 1:
        t_iou = np.squeeze(t_iou, axis=1)
    if calculate_overlap_self:
        if candidate_segments_ndim == 1:
            t_overlap_self = np.squeeze(t_overlap_self, axis=1)
        return t_iou, t_overlap_self

    return t_iou


def average_recall_at_avg_proposals(ground_truth,
                                    proposals,
                                    total_num_proposals,
                                    max_avg_proposals=None,
                                    temporal_iou_thresholds=np.linspace(
                                        0.5, 0.95, 10)):
    """Computes the average recall given an average number (percentile) of
    proposals per video.

    Args:
        ground_truth (dict): Dict containing the ground truth instances.
        proposals (dict): Dict containing the proposal instances.
        total_num_proposals (int): Total number of proposals in the
            proposal dict.
        max_avg_proposals (int | None): Max number of proposals for one video.
            Default: None.
        temporal_iou_thresholds (np.ndarray): 1D array with temporal_iou
            thresholds. Default: ``np.linspace(0.5, 0.95, 10)``.

    Returns:
        tuple([np.ndarray, np.ndarray, np.ndarray, float]):
            (recall, average_recall, proposals_per_video, auc)
            In recall, ``recall[i,j]`` is recall at i-th temporal_iou threshold
            at the j-th average number (percentile) of average number of
            proposals per video. The average_recall is recall averaged
            over a list of temporal_iou threshold (1D array). This is
            equivalent to ``recall.mean(axis=0)``. The ``proposals_per_video``
            is the average number of proposals per video. The auc is the area
            under ``AR@AN`` curve.
    """

    total_num_videos = len(ground_truth)

    if not max_avg_proposals:
        max_avg_proposals = float(total_num_proposals) / total_num_videos

    ratio = (max_avg_proposals * float(total_num_videos) / total_num_proposals)

    # For each video, compute temporal_iou scores among the retrieved proposals
    score_list = []
    total_num_retrieved_proposals = 0
    for video_id in ground_truth:
        # Get proposals for this video.
        proposals_video_id = proposals[video_id]
        this_video_proposals = proposals_video_id[:, :2]
        # Sort proposals by score.
        sort_idx = proposals_video_id[:, 2].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :].astype(
            np.float32)

        # Get ground-truth instances associated to this video.
        ground_truth_video_id = ground_truth[video_id]
        this_video_ground_truth = ground_truth_video_id[:, :2].astype(
            np.float32)
        if this_video_proposals.shape[0] == 0:
            n = this_video_ground_truth.shape[0]
            score_list.append(np.zeros((n, 1)))
            continue

        if this_video_proposals.ndim != 2:
            this_video_proposals = np.expand_dims(this_video_proposals, axis=0)
        if this_video_ground_truth.ndim != 2:
            this_video_ground_truth = np.expand_dims(
                this_video_ground_truth, axis=0)

        num_retrieved_proposals = np.minimum(
            int(this_video_proposals.shape[0] * ratio),
            this_video_proposals.shape[0])
        total_num_retrieved_proposals += num_retrieved_proposals
        this_video_proposals = this_video_proposals[:
                                                    num_retrieved_proposals, :]

        # Compute temporal_iou scores.
        t_iou = pairwise_temporal_iou(this_video_proposals,
                                      this_video_ground_truth)
        score_list.append(t_iou)

    # Given that the length of the videos is really varied, we
    # compute the number of proposals in terms of a ratio of the total
    # proposals retrieved, i.e. average recall at a percentage of proposals
    # retrieved per video.

    # Computes average recall.
    pcn_list = np.arange(1, 101) / 100.0 * (
        max_avg_proposals * float(total_num_videos) /
        total_num_retrieved_proposals)
    matches = np.empty((total_num_videos, pcn_list.shape[0]))
    positives = np.empty(total_num_videos)
    recall = np.empty((temporal_iou_thresholds.shape[0], pcn_list.shape[0]))
    # Iterates over each temporal_iou threshold.
    for ridx, temporal_iou in enumerate(temporal_iou_thresholds):
        # Inspect positives retrieved per video at different
        # number of proposals (percentage of the total retrieved).
        for i, score in enumerate(score_list):
            # Total positives per video.
            positives[i] = score.shape[0]
            # Find proposals that satisfies minimum temporal_iou threshold.
            true_positives_temporal_iou = score >= temporal_iou
            # Get number of proposals as a percentage of total retrieved.
            pcn_proposals = np.minimum(
                (score.shape[1] * pcn_list).astype(np.int), score.shape[1])

            for j, num_retrieved_proposals in enumerate(pcn_proposals):
                # Compute the number of matches
                # for each percentage of the proposals
                matches[i, j] = np.count_nonzero(
                    (true_positives_temporal_iou[:, :num_retrieved_proposals]
                     ).sum(axis=1))

        # Computes recall given the set of matches per video.
        recall[ridx, :] = matches.sum(axis=0) / positives.sum()

    # Recall is averaged.
    avg_recall = recall.mean(axis=0)

    # Get the average number of proposals per video.
    proposals_per_video = pcn_list * (
        float(total_num_retrieved_proposals) / total_num_videos)
    # Get AUC
    area_under_curve = np.trapz(avg_recall, proposals_per_video)
    auc = 100. * float(area_under_curve) / proposals_per_video[-1]
    return recall, avg_recall, proposals_per_video, auc


def get_weighted_score(score_list, coeff_list):
    """Get weighted score with given scores and coefficients.

    Given n predictions by different classifier: [score_1, score_2, ...,
    score_n] (score_list) and their coefficients: [coeff_1, coeff_2, ...,
    coeff_n] (coeff_list), return weighted score: weighted_score =
    score_1 * coeff_1 + score_2 * coeff_2 + ... + score_n * coeff_n

    Args:
        score_list (list[list[np.ndarray]]): List of list of scores, with shape
            n(number of predictions) X num_samples X num_classes
        coeff_list (list[float]): List of coefficients, with shape n.

    Returns:
        list[np.ndarray]: List of weighted scores.
    """
    assert len(score_list) == len(coeff_list)
    num_samples = len(score_list[0])
    for i in range(1, len(score_list)):
        assert len(score_list[i]) == num_samples

    scores = np.array(score_list)  # (num_coeff, num_samples, num_classes)
    coeff = np.array(coeff_list)  # (num_coeff, )
    weighted_scores = list(np.dot(scores.T, coeff).T)
    return weighted_scores


def softmax(x, dim=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e_x / e_x.sum(axis=dim, keepdims=True)


def interpolated_precision_recall(precision, recall):
    """Interpolated AP - VOCdevkit from VOC 2011.

    Args:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.

    Returns：
        float: Average precision score.
    """
    mprecision = np.hstack([[0], precision, [0]])
    mrecall = np.hstack([[0], recall, [1]])
    for i in range(len(mprecision) - 1)[::-1]:
        mprecision[i] = max(mprecision[i], mprecision[i + 1])
    idx = np.where(mrecall[1::] != mrecall[0:-1])[0] + 1
    ap = np.sum((mrecall[idx] - mrecall[idx - 1]) * mprecision[idx])
    return ap


def average_precision_at_temporal_iou(ground_truth,
                                      prediction,
                                      temporal_iou_thresholds=(np.linspace(
                                          0.5, 0.95, 10))):
    """Compute average precision (in detection task) between ground truth and
    predicted data frames. If multiple predictions match the same predicted
    segment, only the one with highest score is matched as true positive. This
    code is greatly inspired by Pascal VOC devkit.

    Args:
        ground_truth (dict): Dict containing the ground truth instances.
            Key: 'video_id'
            Value (np.ndarray): 1D array of 't-start' and 't-end'.
        prediction (np.ndarray): 2D array containing the information of
            proposal instances, including 'video_id', 'class_id', 't-start',
            't-end' and 'score'.
        temporal_iou_thresholds (np.ndarray): 1D array with temporal_iou
            thresholds. Default: ``np.linspace(0.5, 0.95, 10)``.

    Returns:
        np.ndarray: 1D array of average precision score.
    """
    ap = np.zeros(len(temporal_iou_thresholds), dtype=np.float32)
    if len(prediction) < 1:
        return ap

    num_gts = 0.
    lock_gt = dict()
    for key in ground_truth:
        lock_gt[key] = np.ones(
            (len(temporal_iou_thresholds), len(ground_truth[key]))) * -1
        num_gts += len(ground_truth[key])

    # Sort predictions by decreasing score order.
    prediction = np.array(prediction)
    scores = prediction[:, 4].astype(float)
    sort_idx = np.argsort(scores)[::-1]
    prediction = prediction[sort_idx]

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(temporal_iou_thresholds), len(prediction)),
                  dtype=np.int32)
    fp = np.zeros((len(temporal_iou_thresholds), len(prediction)),
                  dtype=np.int32)

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in enumerate(prediction):

        # Check if there is at least one ground truth in the video.
        if this_pred[0] in ground_truth:
            this_gt = np.array(ground_truth[this_pred[0]], dtype=float)
        else:
            fp[:, idx] = 1
            continue

        t_iou = pairwise_temporal_iou(this_pred[2:4].astype(float), this_gt)
        # We would like to retrieve the predictions with highest t_iou score.
        t_iou_sorted_idx = t_iou.argsort()[::-1]
        for t_idx, t_iou_threshold in enumerate(temporal_iou_thresholds):
            for jdx in t_iou_sorted_idx:
                if t_iou[jdx] < t_iou_threshold:
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[this_pred[0]][t_idx, jdx] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[t_idx, idx] = 1
                lock_gt[this_pred[0]][t_idx, jdx] = idx
                break

            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float32)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float32)
    recall_cumsum = tp_cumsum / num_gts

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for t_idx in range(len(temporal_iou_thresholds)):
        ap[t_idx] = interpolated_precision_recall(precision_cumsum[t_idx, :],
                                                  recall_cumsum[t_idx, :])

    return ap


# --- Other Functions --- #

def mmit_mean_average_precision(scores, labels):
    """Mean average precision for multi-label recognition. Used for reporting
    MMIT style mAP on Multi-Moments in Times. The difference is that this
    method calculates average-precision for each sample and averages them among
    samples.

    Args:
        scores (list[np.ndarray]): Prediction scores of different classes for
            each sample.
        labels (list[np.ndarray]): Ground truth many-hot vector for each
            sample.

    Returns:
        np.float: The MMIT style mean average precision.
    """
    results = []
    for score, label in zip(scores, labels):
        precision, recall, _ = binary_precision_recall_curve(score, label)
        ap = -np.sum(np.diff(recall) * np.array(precision)[:-1])
        results.append(ap)
    return np.mean(results)


def mean_average_precision(scores, labels):
    """Mean average precision for multi-label recognition.

    Args:
        scores (list[np.ndarray]): Prediction scores of different classes for
            each sample.
        labels (list[np.ndarray]): Ground truth many-hot vector for each
            sample.

    Returns:
        np.float: The mean average precision.
    """
    results = []
    scores = np.stack(scores).T
    labels = np.stack(labels).T

    for score, label in zip(scores, labels):
        precision, recall, _ = binary_precision_recall_curve(score, label)
        ap = -np.sum(np.diff(recall) * np.array(precision)[:-1])
        results.append(ap)
    results = [x for x in results if not np.isnan(x)]
    if results == []:
        return np.nan
    return np.mean(results)


# --- Other functions --- #

def aidan_auc(preds, labels):
    auc = roc_auc_score(labels, [pred[1] for pred in preds])
    return auc

def aidan_acc(preds, labels):
    scores = np.argmax(preds, axis=1)
    acc = accuracy_score(labels, scores)
    return acc

def save_diagnostics(preds, labels, save_path='diagnostic_save_path', image_save_path='image_save_path'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    scores = np.argmax(preds, axis=1)
    positive_preds = [pred[1] for pred in preds]

    cross_entropy = log_loss(labels, preds)
    cm = confusion_matrix(scores, labels)
    auc = roc_auc_score(labels, preds)
    precision = precision_score(labels, scores)
    recall = recall_score(labels, scores)
    f1 = f1_score(labels, scores)
    accuracy = accuracy_score(labels, scores)

    # Print confusion matrix
    print("Confusion Matrix:")
    print(cm)

    # Plot AUC-ROC and Precision-Recall curves as subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # AUC-ROC curve
    fpr, tpr, _ = roc_curve(labels, positive_preds)
    ax1.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc:.2f}')
    ax1.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc='lower right')

    # Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(labels, positive_preds)
    average_precision = average_precision_score(labels, positive_preds)
    ax2.plot(recall_vals, precision_vals, color='b', lw=2, label=f'AP = {average_precision:.2f}')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_image_file = os.path.join(save_path, f"roc_pr_curves_{current_time}.png")
    plt.savefig(combined_image_file)
    plt.close()

    # Save diagnostics to text file
    diagnostics_file = os.path.join(save_path, f"diagnostics_{current_time}.txt")
    with open(diagnostics_file, "a+") as file:
        file.write("Diagnostics Report\n")
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Confusion Matrix:\n{cm}\n\n")
        file.write(f"Cross-entropy loss:\n{cross_entropy:.4f}\n\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"AUC: {auc:.4f}\n")
        file.write(f"Combined ROC and Precision-Recall Curves saved at: {combined_image_file}\n")
        file.write(f"\n\n\nLabels:\n{labels}\nPreds:\n{preds}\nScores:\n{scores}")

    print(f"Diagnostics saved to {diagnostics_file}")
    print(f"Combined ROC and Precision-Recall Curves saved to {combined_image_file}")

