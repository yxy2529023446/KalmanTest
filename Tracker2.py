# -*- coding: utf-8 -*-
# 在mac的virtualenvs环境下需要加上以下2行
import matplotlib
matplotlib.use('TkAgg')

from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
import time


def get_iou(box1, box2):
    """
    计算两个检测框的IOU（Intersection Over Union）检测评价函数
    交并比检测框形式[x1,y1,x2,y2]
    """
    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])
    weight = max(0, xx2 - xx1)
    height = max(0, yy2 - yy1)
    s = weight * height
    s1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    s2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    iou = s / (s1 + s2 - s)
    return iou


def box2xysr(box):
    """
    输入检测框[x1, y1, x2, y2]
    输出检测框[x, y, s, r] ：(x, y)为中心点坐标，s为面积，r为长宽比
    """
    width = box[2] - box[0]
    height = box[3] - box[1]
    x = box[0] + width / 2
    y = box[1] + height / 2
    s = width * height
    r = width / height
    return np.array([x, y, s, r]).reshape((4, 1))


def xysr2box(xysr):
    """
    输入检测框[x, y, s, r]
    输出检测框[x1, y1, x2, y2, ...]
    """
    width = np.sqrt(abs(xysr[2] * xysr[3]))
    height = xysr[2] / width
    x1 = xysr[0] - width / 2
    y1 = xysr[1] - height / 2
    x2 = xysr[0] + width / 2
    y2 = xysr[1] + height / 2
    return np.array([x1, y1, x2, y2]).reshape((1, 4))


class KalmanBoxTracker:
    count = 0
    def __init__(self, box):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # 状态转移矩阵
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]], dtype=float)
        # 测量函数
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]], dtype=float)
        # 测量噪声
        self.kf.R *= 0.1
        # 过程噪声
        self.kf.Q *= 1
        # 协方差矩阵
        self.kf.P *= 1000
        # 初始状态
        self.kf.x[: 4] = box2xysr(box)

        self.time_since_update = 0
        self.id = self.__class__.count
        self.__class__.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(xysr2box(self.kf.x))
        return self.history[-1]

    def update(self, box):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(box2xysr(box))

    def get_state(self):
        return xysr2box(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.1):
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=float)

    for i, detection in enumerate(detections):
        for j, tracker in enumerate(trackers):
            iou_matrix[i, j] = get_iou(detection, tracker)
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for i, detction in enumerate(detections):
        if i not in matched_indices[:, 0]:
            unmatched_detections.append(i)

    unmatched_trackers = []
    for j, tracker in enumerate(trackers):
        if j not in matched_indices[:, 1]:
            unmatched_trackers.append(j)

    #根据低交并比值过滤
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold :
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
            np.delete(matched_indices, m[0])

    return matched_indices, np.array(unmatched_detections), np.array(unmatched_trackers)


class Xxx(object):
    def __init__(self, max_age=20, min_hits=1.5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        """
        参数:
          dets: 检测框列表，一个numpy队列[[x,y,w,h,label],[x,y,w,h,label],...]
        （该函数每帧调用一次，即使没有检测框校正）

        返回：
          目标框列表，一个numpy队列[[x,y,w,h,ID],[x,y,w,h,ID],...]
        （输入输出的队列大小不等）
        """
        self.frame_count += 1
        #根据检测获得预测
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

        #根据检测，更新以配对的
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1]==t)[0], 0]
                trk.update(dets[d, :][0])

        #给未匹配的检测建立新Tracker
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1)) # +1 as MOT benchmark requires positive
            i -= 1
            #删除寿命到期的Tracker
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

# 视频和数据
video_file = 'videos/144115192380048440_2017052310938_output2.mp4'
data_file = 'data/144115192380048440_2017052310938_output.mp4.txt'

mot_tracker = Xxx()

colors = np.random.rand(128, 3) * 255

track_line0, track_line1, track_line2, track_line3, track_line4 = [], [], [], [], []
track_line0_id = track_line1_id = track_line2_id = track_line3_id = track_line4_id = 0

# 开始的帧数
frame_count = 60000

total_targets = 0

f1 = open(data_file)
cap = cv2.VideoCapture(video_file)
success, frame = cap.read()
last_line = f1.readline().split()
while success:
    detections = []
    current_line = f1.readline().split()
    while last_line[-2] == current_line[-2]:
        if float(last_line[-1]) > 0.3:
            detections.append(last_line[:4])
        last_line = current_line
        current_line = f1.readline().split()
    if float(last_line[-1]) > 0.3:
        detections.append(last_line[:4])
    last_line = current_line

    detections = np.array(detections, dtype=float)
    if detections.any():
        detections[:, 2:4] += detections[:, 0:2]
    trackers = mot_tracker.update(detections)

    for tracker in trackers:
        tracker = tracker.astype(int)
        color = colors[tracker[4] % 128]
        cv2.rectangle(frame, (tracker[0], tracker[1]), (tracker[2]-tracker[0], tracker[3]-tracker[1]), color, 2)
        cv2.putText(frame, str(tracker[4]), (tracker[0], tracker[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        xxx = tracker[4] % 5
        if xxx == 0:
            color0 = colors[tracker[4] % 128]
            if track_line0_id != tracker[4]:
                track_line0_id = tracker[4]
                track_line0 = []
            track_line0.append((int(tracker[0] + (tracker[2] - 2 * tracker[0]) / 2),
                                 int(tracker[1] + (tracker[3] - 2 * tracker[1]) / 2)))
        if xxx == 1:
            color1 = colors[tracker[4] % 128]
            if track_line1_id != tracker[4]:
                track_line1_id = tracker[4]
                track_line1 = []
            track_line1.append((int(tracker[0] + (tracker[2] - 2 * tracker[0]) / 2),
                                 int(tracker[1] + (tracker[3] - 2 * tracker[1]) / 2)))
        if xxx == 2:
            color2 = colors[tracker[4] % 128]
            if track_line3_id != tracker[4]:
                track_line3_id = tracker[4]
                track_line2 = []
            track_line2.append((int(tracker[0] + (tracker[2] - 2 * tracker[0]) / 2),
                                 int(tracker[1] + (tracker[3] - 2 * tracker[1]) / 2)))
        if xxx == 3:
            color3 = colors[tracker[4] % 128]
            if track_line3_id != tracker[4]:
                track_line3_id = tracker[4]
                track_line3 = []
            track_line3.append((int(tracker[0] + (tracker[2] - 2 * tracker[0]) / 2),
                                 int(tracker[1] + (tracker[3] - 2 * tracker[1]) / 2)))
        if xxx == 4:
            color4 = colors[tracker[4] % 128]
            if track_line4_id != tracker[4]:
                track_line4_id = tracker[4]
                track_line4 = []
            track_line4.append((int(tracker[0] + (tracker[2] - 2 * tracker[0]) / 2),
                                 int(tracker[1] + (tracker[3] - 2 * tracker[1]) / 2)))

    if len(track_line0) == 150:
        track_line0.pop(0)
    if len(track_line1) == 150:
        track_line1.pop(0)
    if len(track_line2) == 150:
        track_line2.pop(0)
    if len(track_line3) == 1500:
        track_line3.pop(0)
    if len(track_line4) == 150:
        track_line4.pop(0)

    for i in range(len(track_line0) - 1):
        cv2.line(frame, track_line0[i], track_line0[i + 1], color0, 2)
    for i in range(len(track_line1) - 1):
        cv2.line(frame, track_line1[i], track_line1[i + 1], color1, 2)
    for i in range(len(track_line2) - 1):
        cv2.line(frame, track_line2[i], track_line2[i + 1], color2, 2)
    for i in range(len(track_line3) - 1):
        cv2.line(frame, track_line3[i], track_line3[i + 1], color3, 2)
    for i in range(len(track_line4) - 1):
        cv2.line(frame, track_line4[i], track_line4[i + 1], color4, 2)

    cv2.putText(frame, 'Frame:' + str(frame_count), (260, 20), cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 0), 2)
    cv2.putText(frame, 'Count:' + str(total_targets), (465, 20), cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 0), 2)
    cv2.imshow('TRACKING', frame)
    cv2.waitKey(1)

    success, frame = cap.read()

    frame_count = frame_count + 1

f1.close()
cap.release()