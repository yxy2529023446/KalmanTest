# -*- coding: utf-8 -*-

import pandas as pd  # 数据分析
import numpy as np  # 数据处理
import cv2
import tools as tl
from filterpy.kalman import KalmanFilter
from sklearn.utils.linear_assignment_ import linear_assignment
import time

# 数据和视频地址
video_file = 'video/144115194519432837_20170314_120930.mp4'
data_file = 'simple_data/data_70.csv'

# file = 'simple_data/complex.txt'
file = '144115194519432837_20170314_120930.mp4.txt'
# video_file = 'simple_data/in_one_1.mp4'
# data_file = 'simple_data/in_one_1.csv'

# video_file = 'simple_data/complex_1_75.mp4'
# data_file = 'simple_data/complex_1_75.csv'

# 读取数据和视频
data_final = pd.read_csv(data_file)
cap = cv2.VideoCapture(video_file)
# print data_final
# 读取视频第一帧
success,frame = cap.read()
cap_fps=cap.get(cv2.CAP_PROP_FPS)
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_0 = np.zeros((cap_width, cap_height, 3), np.int)

# 初始化检测框、帧数统计
track_window_data = [[0, 0, 0, 0]]*10
frame_count = 1
numframes = cap.get(7)

# 初始化数据驱使的检测框
data_flag = 0
data_number = 1
frame_cerrent = data_final['frame'][data_flag]
frame_next = data_final['frame'][data_flag + 1]
track_window_data[0] = [data_final['xmin'][data_flag],
                   data_final['ymin'][data_flag],
                   data_final['length'][data_flag],
                   data_final['height'][data_flag]]

# 扫描同一帧数的检测框
while (frame_cerrent == frame_next):
    data_flag = data_flag + 1
    track_window_data[data_number] = [data_final['xmin'][data_flag],
                                      data_final['ymin'][data_flag],
                                      data_final['length'][data_flag],
                                      data_final['height'][data_flag]]
    data_number = data_number + 1
    frame_next = data_final['frame'][data_flag + 1]

def iou(bb_test,bb_gt):
  """
  计算两个检测框的IOU（Intersection Over Union）检测评价函数，交并比
  检测框形式[x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
  return(o)

def convert_bbox_to_z(bbox):
  """
  输入检测框[x1,y1,x2,y2]
  输出检测框[x,y,s,r] ：(x,y)为中心点坐标，s为面积，r为长宽比
  """
  w = bbox[2]-bbox[0]
  h = bbox[3]-bbox[1]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s = w*h    #面积
  r = w/float(h)   #长宽比
  return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x,score=None):
  """
  输入检测框[x,y,s,r]
  输出检测框[x1,y1,x2,y2]
  """
  w = np.sqrt(x[2]*x[3])
  h = x[2]/w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  个体跟踪目标检测框的内部状态
  """
  count = 0
  def __init__(self,bbox):
    """
    初始化检测框
    """
    #定义连续速度模型
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #给一个不可观测初始速度一个高不确定性
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """

    根据检测结果更新速度状态
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    预测速度状态
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    获取当前估计的检测框值[x1,y1,x2,y2]
    """
    return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  给跟踪对象分配检测框[x1,y1,x2,y2]

  返回3个列表：匹配、未匹配的检测框、未匹配的预测框
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = iou(det,trk)
  matched_indices = linear_assignment(-iou_matrix)

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #根据低交并比值（low IOU）过滤匹配
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0],m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



class Sort(object):
  def __init__(self,max_age=18,min_hits=1.5):
    """
    设置关键参数
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

  def update(self,dets):
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
    trks = np.zeros((len(self.trackers),5))
    # kLine = np.zeros((len(self.trackers),15))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)

    #根据检测，更新以配对的
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0],0]
        trk.update(dets[d,:][0])

    #给未匹配的检测建立新Tracker
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
        # kLine = ((trk[0] + ((trk[2] - 2 * d[0]) / 2), trk[1] + ((trk[3] - 2 * trk[1]) / 2)))
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #删除寿命到期的Tracker
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))


# class KalmanLine(object):
#     def __init__(self, label):
#
#         self.label = label
#         self.lines = []

    # def getKalmanLine(no):
    #     kalmanLine = []
# 初始化Camshift算法驱使的检测框
cam_xmin = data_final['xmin'][0]
cam_ymin = data_final['ymin'][0]
cam_length = data_final['length'][0]-20
cam_height = data_final['height'][0]-40
track_window_cam = (cam_xmin, cam_ymin, cam_length, cam_height)
# print track_window_cam

# ROI-感兴趣区域-图像采集
roi = frame[cam_ymin:cam_ymin+cam_height, cam_xmin:cam_xmin+cam_length]
# cv2.imshow('test', roi)
# k = cv2.waitKey(0) & 0xff
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 1 )

mot_tracker = Sort()
total_time = 0.0
colours = np.random.rand(128, 3)
seq = np.loadtxt(file, delimiter=' ', usecols=(0,1,2,3,6,7)).astype('float')
np.set_printoptions(suppress=True)
seq_dets = seq[seq[:, 5] > 0.7]
KalmanLine_1 = []
KalmanLine_2 = []
KalmanLine_3 = []
KalmanLine_4 = []
KalmanLine_5 = []
Kid_1 = 1
Kid_2 = 2
Kid_3 = 3
Kid_4 = 4
Kid_5 = 5
Kno_1 = Kno_2 = Kno_3 =Kno_4 = Kno_5 = 0


while(success):
    success, frame = cap.read()
    img = frame.copy()

    dets = seq_dets[seq_dets[:, 4] == frame_count, 0:4]
    dets[:, 2:4] += dets[:, 0:2]
    # dets.astype('int')

    start_time = time.time()
    trackers = mot_tracker.update(dets)
    cycle_time = time.time() - start_time
    total_time += cycle_time

    for d in trackers:
        # cv2.rectangle(img,(d[0],d[1]),(d[2]-d[0],d[3]-d[1]), (0, 255, 0), 2)
        d = d.astype(np.uint32)
        color = (colours[d[4] % 128, :]*127)
        cv2.rectangle(img, (d[0], d[1]), (d[2] - d[0], d[3] - d[1]),color, 2)
        cv2.putText(img, str(d[4]), (d[0], d[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
        for case in tl.switch(d[4]%5):
            if case(0):
                color1 = (colours[d[4] % 128, :] * 127)
                if Kno_1 == d[4]:
                    KalmanLine_1.append((int(d[0] + ((d[2] - 2 * d[0]) / 2)),int(d[1] + ((d[3] - 2 * d[1]) / 2))))
                    # for i in range(len(KalmanLine_1) - 1):
                    #     cv2.line(img, KalmanLine_1[i], KalmanLine_1[i + 1], color, 2)
                else:
                    Kno_1 = d[4]
                    KalmanLine_1 = []
                    KalmanLine_1.append((int(d[0] + ((d[2] - 2 * d[0]) / 2)),int(d[1] + ((d[3] - 2 * d[1]) / 2))))
                    # for i in range(len(KalmanLine_1) - 1):
                    #     cv2.line(img, KalmanLine_1[i], KalmanLine_1[i + 1], color, 2)
                break
            if case(1):
                color2 = (colours[d[4] % 128, :] * 127)
                if Kno_2 == d[4]:
                    KalmanLine_2.append((int(d[0] + ((d[2] - 2 * d[0]) / 2)),int(d[1] + ((d[3] - 2 * d[1]) / 2))))
                    # for i in range(len(KalmanLine_2) - 1):
                    #     cv2.line(img, KalmanLine_2[i], KalmanLine_2[i + 1], color, 2)
                else:
                    Kno_2 = d[4]
                    KalmanLine_2 = []
                    KalmanLine_2.append((int(d[0] + ((d[2] - 2 * d[0]) / 2)),int(d[1] + ((d[3] - 2 * d[1]) / 2))))
                    # for i in range(len(KalmanLine_2) - 1):
                    #     cv2.line(img, KalmanLine_2[i], KalmanLine_2[i + 1], color, 2)
                break
            if case(2):
                color3 = (colours[d[4] % 128, :] * 127)
                if Kno_3 == d[4]:
                    KalmanLine_3.append((int(d[0] + ((d[2] - 2 * d[0]) / 2)),int(d[1] + ((d[3] - 2 * d[1]) / 2))))
                    # for i in range(len(KalmanLine_3) - 1):
                    #     cv2.line(frame, KalmanLine_3[i], KalmanLine_3[i + 1], color, 2)
                else:
                    Kno_3 = d[4]
                    KalmanLine_3 = []
                    KalmanLine_3.append((int(d[0] + ((d[2] - 2 * d[0]) / 2)),int(d[1] + ((d[3] - 2 * d[1]) / 2))))
                    # for i in range(len(KalmanLine_3) - 1):
                    #     cv2.line(frame, KalmanLine_3[i], KalmanLine_3[i + 1], color, 2)
                break
            if case(3):
                color4 = (colours[d[4] % 128, :] * 127)
                if Kno_4 == d[4]:
                    KalmanLine_4.append((int(d[0] + ((d[2] - 2 * d[0]) / 2)),int(d[1] + ((d[3] - 2 * d[1]) / 2))))
                    # for i in range(len(KalmanLine_4) - 1):
                    #     cv2.line(frame, KalmanLine_4[i], KalmanLine_4[i + 1], color, 2)
                else:
                    Kno_4 = d[4]
                    KalmanLine_4 = []
                    KalmanLine_4.append((int(d[0] + ((d[2] - 2 * d[0]) / 2)),int(d[1] + ((d[3] - 2 * d[1]) / 2))))
                    # for i in range(len(KalmanLine_4) - 1):
                    #     cv2.line(frame, KalmanLine_4[i], KalmanLine_4[i + 1], color, 2)
                break
            if case(4):
                color5 = (colours[d[4] % 128, :] * 127)
                if Kno_5 == d[4]:
                    KalmanLine_5.append((int(d[0] + ((d[2] - 2 * d[0]) / 2)),int(d[1] + ((d[3] - 2 * d[1]) / 2))))
                    # for i in range(len(KalmanLine_5) - 1):
                    #     cv2.line(frame, KalmanLine_5[i], KalmanLine_5[i + 1], color, 2)
                else:
                    Kno_5 = d[4]
                    KalmanLine_5 = []
                    KalmanLine_5.append((int(d[0] + ((d[2] - 2 * d[0]) / 2)),int(d[1] + ((d[3] - 2 * d[1]) / 2))))
                    # for i in range(len(KalmanLine_5) - 1):
                    #     cv2.line(frame, KalmanLine_5[i], KalmanLine_5[i + 1], color, 2)
                break
        if len(KalmanLine_1) == 80:
            del KalmanLine_1[0]
        if len(KalmanLine_2) == 80:
            del KalmanLine_2[0]
        if len(KalmanLine_3) == 80:
            del KalmanLine_3[0]
        if len(KalmanLine_4) == 80:
            del KalmanLine_4[0]
        if len(KalmanLine_5) == 80:
            del KalmanLine_5[0]

        # 数据驱使的检测框
    # if frame_count == frame_cerrent:
    #     x1, y1, w1, h1 = track_window_data[0]
    #     cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
    #
    #     for case in tl.switch(data_number):
    #         # if case(1):
    #         #     cv2.imshow('TRACKING', frame)
    #         #     break
    #         if case(2):
    #             x2, y2, w2, h2 = track_window_data[1]
    #             cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
    #             # cv2.imshow('TRACKING', frame)
    #             break
    #         if case(3):
    #             x2, y2, w2, h2 = track_window_data[1]
    #             x3, y3, w3, h3 = track_window_data[2]
    #             cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
    #             cv2.rectangle(frame, (x3, y3), (x3 + w3, y3 + h3), (255, 0, 0), 2)
    #             # cv2.imshow('TRACKING', frame)
    #             break
    #         if case(4):
    #             x2, y2, w2, h2 = track_window_data[1]
    #             x3, y3, w3, h3 = track_window_data[2]
    #             x4, y4, w4, h4 = track_window_data[3]
    #             cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
    #             cv2.rectangle(frame, (x3, y3), (x3 + w3, y3 + h3), (255, 0, 0), 2)
    #             cv2.rectangle(frame, (x4, y4), (x4 + w4, y4 + h4), (255, 0, 0), 2)
    #             # cv2.imshow('TRACKING', frame)
    #             break
    #
    #     track_window_data = [[0, 0, 0, 0]] * 10
    #     data_flag = data_flag + 1
    #     data_number = 1
    #     frame_cerrent = data_final['frame'][data_flag]
    #     frame_next = data_final['frame'][data_flag + 1]
    #     track_window_data[0] = [data_final['xmin'][data_flag],
    #                             data_final['ymin'][data_flag],
    #                             data_final['length'][data_flag],
    #                             data_final['height'][data_flag]]
    #     while (frame_cerrent == frame_next):
    #         data_flag = data_flag + 1
    #         track_window_data[data_number] = [data_final['xmin'][data_flag],
    #                                           data_final['ymin'][data_flag],
    #                                           data_final['length'][data_flag],
    #                                           data_final['height'][data_flag]]
    #         data_number = data_number + 1
    #         frame_next = data_final['frame'][data_flag + 1]




    for i in range(len(KalmanLine_1) - 1):
        cv2.line(img, KalmanLine_1[i], KalmanLine_1[i + 1], color1, 2)
    for i in range(len(KalmanLine_2) - 1):
        cv2.line(img, KalmanLine_2[i], KalmanLine_2[i + 1], color2, 2)
    for i in range(len(KalmanLine_3) - 1):
        cv2.line(img, KalmanLine_3[i], KalmanLine_3[i + 1], color3, 2)
    for i in range(len(KalmanLine_4) - 1):
        cv2.line(img, KalmanLine_4[i], KalmanLine_4[i + 1], color4, 2)
    for i in range(len(KalmanLine_5) - 1):
        cv2.line(img, KalmanLine_5[i], KalmanLine_5[i + 1], color5, 2)

    cv2.putText(img, str('Tracking Mode'), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 0, 0), 2)
    cv2.putText(img, str('Frame:') + str(frame_count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 0, 0), 2)
    cv2.putText(frame, str('Original'), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 0, 0), 2)
    cv2.putText(frame, str('Frame:') + str(frame_count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 0, 0), 2)

    if len(trackers) != 0:
        cv2.imwrite("simple_data/video/img_" + str(frame_count) + ".png", img)
        cv2.imwrite("simple_data/video/frame_" + str(frame_count) + ".png", frame)

    # 键盘控制
    cv2.imshow('TRACKING', img)
    # cv2.imshow('Original',frame)
    k = cv2.waitKey(1) & 0xff
    for case in tl.switch(k):
        if case(ord('n')):
            k = cv2.waitKey(1000) & 0xff
            continue
        if case(ord(' ')):
            k = cv2.waitKey(0) & 0xff
            continue
        if case(ord('c')):
            cv2.imwrite(str(frame_count) + "_img.jpg", img)
            cv2.imwrite(str(frame_count) + "_frame.jpg", frame)
            print(chr(k))
            continue
        if case(27):
            break
        if case():
            continue
    if k == 27:
        break

    frame_count = frame_count + 1


cv2.destroyAllWindows()
cap.release()
