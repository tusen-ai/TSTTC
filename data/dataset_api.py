# Interface for accessing TSTTC Dataset

# TuSimple TTC is a large-scale dataset for TTC estimation. This api is used to assist in
# loading parsing and visualizing the annotation in TSTTC. For more information, please refer to
# $TSTTC website https://open-dataset.tusen.ai/$, for the dataset and paper.

# The following API functions are defined:
#  TSTTC        - TSTTC api class that loads TSTTC annotation file and prepare data structures.
#  getAnnoIds   - Get ann ids that satisfy given filter conditions.
#  getImgSeqIds - Get image ids that satisfy given filter conditions.
#  loadAnnos    - Load anns with the specified ids.
#  loadImgSeqs  - Load image sequence anns with the specified ids.
#  showAnns     - Display the specified annotations.
#  loadRes      - Load algorithm results and create API for accessing them.
#  evalRes      - Evaluate the result file and return MiD & RTE.


import time
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import math
import json
import pickle
import skimage.io as io
import matplotlib.patches as patches
from collections import defaultdict

PKL_EXT = [".pkl"]


def load_json(f):
    return json.load(open(f, 'r'))
def load_pickle(f):
    return pickle.load(open(f, 'rb'))
def get_file_list(path, type_list):
    # get all file names in the type_list under the path
    file_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in type_list:
                file_names.append(apath)
    return file_names

def ttc_to_scale_ratio(ttc, fps=10):
    scale_ratio = 1 / ((1 / (ttc * fps) + 1) + 1e-6)
    return scale_ratio

def scale_ratio_to_ttc(scale_ratio, fps=10):
    ttc = 1 / ((fps * (1 / scale_ratio - 1)) + 1e-6)
    return ttc

class TSTTC:
    def __init__(self, dataset_dir=None,annotation_dir=None, sequence_len=6):
        '''
        :param dataset_dir: location of dataset directory, str or list of str
        :param annotation_dir: location of annotation directory, str or list of str
        :param sequence_len: sequence length
        '''
        self.annos,self.frameSeqs = dict(), dict()
        self.bagToAnnos, self.camToAnnos = defaultdict(list), defaultdict(list)
        if dataset_dir is not None:
            print('loading annotations into memory...')
            tic = time.time()
            self.dataset_dir = dataset_dir
            self.annos_dir = dataset_dir if annotation_dir is None else annotation_dir
            self.sequence_len = sequence_len
            self.cur_frameSeqs_id = 0
            self.cur_inst_id = 0
            self.raw_annos = self.loadRawAnnos()
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.createIndex()
            del self.raw_annos

    def loadRawAnnos(self):
        anno_list = []
        self.annos_dir = [self.annos_dir] if isinstance(self.annos_dir, str) else self.annos_dir
        self.dataset_dir = [self.dataset_dir] if isinstance(self.dataset_dir, str) else self.dataset_dir
        anno_names_list = []
        for anno_dir in self.annos_dir:
            anno_names_list.append(get_file_list(anno_dir, PKL_EXT))
        for anno_names in anno_names_list:
            subSet = []
            for anno_name in anno_names:
                subSet.append(load_pickle(anno_name))
            anno_list.append(subSet)
        return anno_list

    def createIndex(self):
        print('creating index...')
        for subSetIdx,subSetAnnos in enumerate(self.raw_annos):
            for bag_annos in subSetAnnos:
                for cam_annos in bag_annos:
                    #sort cam_annos by key
                    cam_annos = dict(sorted(cam_annos.items(), key=lambda item: item[0]))
                    cam_annos_list = list(cam_annos.items())
                    tmpSeqDict = dict()
                    for i in range(0, len(cam_annos)):
                        frame_annos = cam_annos_list[i]
                        for obj in frame_annos[1]:
                            if obj.id not in tmpSeqDict:
                                tmpSeqDict[obj.id] = [obj]
                            else:
                                if obj.ts - tmpSeqDict[obj.id][-1].ts == 1e8:
                                    tmpSeqDict[obj.id].append(obj)
                                else:
                                    tmpSeqDict[obj.id] = [obj]
                        anno_unit = []
                        remove_keys = []
                        for tmpObjId,obj in tmpSeqDict.items():
                            if len(obj) == self.sequence_len:
                                anno_unit.append(obj)
                                remove_keys.append(tmpObjId)
                        for key in remove_keys:
                            tmpSeqDict.pop(key)

                        if len(anno_unit) == 0: continue
                        for objSeqs in anno_unit:
                            for obj in objSeqs:
                                obj.bag_stamp = os.path.basename(obj.bag_stamp)
                                obj.frameSeq_id = self.cur_frameSeqs_id
                                obj.img_path = os.path.join(self.dataset_dir[subSetIdx], obj.bag_stamp, 'cam' + str(obj.cam_id),
                                                            str(obj.ts) + '.jpg')
                            objSeqs[-1].anno_id = self.cur_inst_id
                            self.annos[self.cur_inst_id] = objSeqs
                            self.cur_inst_id += 1

                        bag_stamp, cam_id = anno_unit[0][-1].bag_stamp, anno_unit[0][-1].cam_id
                        self.frameSeqs[self.cur_frameSeqs_id] = anno_unit
                        self.bagToAnnos[bag_stamp].append(anno_unit)
                        self.camToAnnos[cam_id].append(anno_unit)
                        self.cur_frameSeqs_id += 1
        print('index created!')
    def getAnnoIds(self,img_seqs_ids=[], ttc_range=[], area_range=[],
                    distance_range=[], lane_range=[], det_obj_ids = [], time_stamps=[],
                    cam_ids=[], bag_stamps=[]
                   ):
        '''
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param ttc_range        : get anns for given ttc range
        :param area_range       : get anns for given area range
        :param distance_range   : get anns for given distance range
        :param lane_range       : get anns for lane range, True for same, False for different
        :param det_obj_ids      : get anns for given det_obj_ids
        :param time_stamps      : get anns for given time_stamps
        :param cam_ids          : get anns for given cam_ids
        :param bag_stamps       : get anns for given bag_stamps
        :return: ids(int array) : integer array of ann ids
        '''
        img_seqs_ids = img_seqs_ids if type(img_seqs_ids) == list else [img_seqs_ids]
        ttc_range = ttc_range if type(ttc_range) == list else [ttc_range]
        area_range = area_range if type(area_range) == list else [area_range]
        distance_range = distance_range if type(distance_range) == list else [distance_range]
        lane_range = lane_range if type(lane_range) == list else [lane_range]
        det_obj_ids = det_obj_ids if type(det_obj_ids) == list else [det_obj_ids]
        time_stamps = time_stamps if type(time_stamps) == list else [time_stamps]
        cam_ids = cam_ids if type(cam_ids) == list else [cam_ids]
        bag_stamps = bag_stamps if type(bag_stamps) == list else [bag_stamps]

        if len(img_seqs_ids) == len(ttc_range) == len(area_range) == len(distance_range) \
                == len(lane_range) == len(det_obj_ids)== len(time_stamps) == len(cam_ids) == len(bag_stamps) == 0:
            return list(self.annos.keys())
        ids = []
        annos = self.annos.values()
        annos = annos if len(img_seqs_ids) == 0 else [anno for anno in annos if anno[-1].frameSeq_id in img_seqs_ids]
        annos = annos if len(ttc_range) == 0 else [anno for anno in annos if anno[-1].ttc_imu >= ttc_range[0] and anno[-1].ttc_imu <= ttc_range[1]]
        annos = annos if len(det_obj_ids) == 0 else [anno for anno in annos if anno[-1].id in det_obj_ids]
        annos = annos if len(lane_range) == 0 else [anno for anno in annos if anno[-1].same_lane in lane_range]
        annos = annos if len(time_stamps) == 0 else [anno for anno in annos if anno[-1].ts in time_stamps]
        annos = annos if len(cam_ids) == 0 else [anno for anno in annos if anno[-1].cam_id in cam_ids]
        annos = annos if len(bag_stamps) == 0 else [anno for anno in annos if anno[-1].bag_stamp in bag_stamps]

        tmp_annos = []
        for obj_seqs in annos:

            if len(area_range) != 0:
                tmp_area = (obj_seqs[-1].box2d[2] - obj_seqs[-1].box2d[0]) * (
                            obj_seqs[-1].box2d[3] - obj_seqs[-1].box2d[1]) * 1024 * 576
                if tmp_area < area_range[0] or tmp_area > area_range[1]: continue

            if len(distance_range) != 0:
                tmp_distance = abs(np.min(obj_seqs[-1].bbox_3d_lidar[:, 1]))
                if tmp_distance < distance_range[0] or tmp_distance > distance_range[1]: continue

            tmp_annos.append(obj_seqs)
        annos = tmp_annos


        for obj_seqs in annos:
            ids.append(obj_seqs[-1].anno_id)
        return list(ids)

    def getImgSeqIds(self, bag_stamps=[], cam_ids=[]):
        '''
        Get img ids that satisfy given filter conditions. default skips that filter
        :param bag_stamps       : get imgs for given bag stamps
        :param cam_ids          : get imgs for given cam ids
        :return: ids(int array) : integer array of img ids
        '''
        bag_stamps = bag_stamps if type(bag_stamps) == list else [bag_stamps]
        cam_ids = cam_ids if type(cam_ids) == list else [cam_ids]
        if len(bag_stamps) == len(cam_ids) == 0:
            return list(self.frameSeqs.keys())
        ids = []
        frameSeqs = self.frameSeqs.values()
        frameSeqs = frameSeqs if len(bag_stamps) == 0 else [frameSeq for frameSeq in frameSeqs if frameSeq[-1][-1].bag_stamp in bag_stamps]
        frameSeqs = frameSeqs if len(cam_ids) == 0 else [frameSeq for frameSeq in frameSeqs if frameSeq[-1][-1].cam_id in cam_ids]
        for frameSeq in frameSeqs:
            ids.append(frameSeq[-1][-1].frameSeq_id)
        return ids

    def loadAnnos(self, ids=[]):
        ids = ids if type(ids) == list else [ids]
        return [self.annos[id] for id in ids]

    def loadImgSeqs(self, ids=[]):
        ids = ids if type(ids) == list else [ids]
        return [self.frameSeqs[id] for id in ids]

    def showAnnos(self,annos):
        """
           Display the specified annotations.
           :param anns (array of object): annotations to display
           :return: None
        """
        if len(annos) == 0:return
        for anno in annos:
            for i in range(len(anno)):
                plt.figure()
                img = io.imread(anno[i].img_path)
                ttc = anno[i].ttc_imu
                box2d = anno[i].box2d
                #convert the box2d from 0~1 to 0~H or 0~W
                box2d[0] *= img.shape[1]
                box2d[1] *= img.shape[0]
                box2d[2] *= img.shape[1]
                box2d[3] *= img.shape[0]
                box2d = np.array(box2d).astype(np.int32)
                #draw box with plt
                ax = plt.gca()
                ax.imshow(img)
                rect = patches.Rectangle((box2d[0],box2d[1]),box2d[2]-box2d[0],box2d[3]-box2d[1],linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
                #add ttc text to the box
                ax.text(box2d[0],box2d[1],str(round(ttc,1)),color='r')
                plt.axis('off')
                plt.tight_layout()
                plt.show()

    def evalRes(self, resFile):
        """
        Load result file and return its performance.
        :param   resFile (str) or resFile (dict)     : file name of result file / list of results, [[anno_id, pred_ttc],...]
        :return:
        str type, ave mid and rte in different conditions
        """
        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str:
            results = load_json(resFile)
        else:
            results = resFile

        assert type(results) == dict, 'results in not dict of objects'
        annsIds = list(results.keys())
        annsIds = [int(anno_id) for anno_id in annsIds]
        assert set(annsIds) == set(self.getAnnoIds()), \
            'Results do not correspond to current tsttc set'
        #preset ttc conditaions: negative,crucial,small,large
        conditions_dict = {'negative_ttc(-20~0)':[-20,0],'crucial_ttc(0~3)':[0,3],'small_ttc(3~6)':[3,6],'large_ttc(6~20)':[6,20]}

        resDict = {}
        for anno_id in results:
            id = int(anno_id)
            ttc_pred = results[anno_id]
            assert ttc_pred != 0, 'ttc prediction cannot be zero'
            ttc_pred = max(min(ttc_pred, 20), -20) # clip
            ttc_imu = self.annos[id][-1].ttc_imu
            scale_pre,scale_imu = ttc_to_scale_ratio(ttc_pred),ttc_to_scale_ratio(ttc_imu)
            rte = abs((ttc_pred-ttc_imu)/ttc_imu)*100
            mid = float(abs(math.log(scale_imu) - math.log(scale_pre)) * 10**4)
            resDict[id] = {'rte': rte, 'mid': mid}
        aveRTE = sum([resDict[id]['rte'] for id in resDict])/ len(resDict)
        aveMID = sum([resDict[id]['mid'] for id in resDict])/ len(resDict)
        summary = ''
        summary += 'Average RTE: {:0.2f}\n'.format(aveRTE)
        summary += 'Average MID: {:0.2f}\n'.format(aveMID)

        for condition in conditions_dict:
            condition_ids = self.getAnnoIds(ttc_range=conditions_dict[condition])
            condition_res = [resDict[id] for id in condition_ids]
            condition_aveRTE = sum([ins_res['rte'] for ins_res in condition_res])/ len(condition_res)
            condition_aveMID = sum([ins_res['mid'] for ins_res in condition_res])/ len(condition_res)
            summary += 'Average RTE of {}: {:0.2f}\n'.format(condition,condition_aveRTE)
            summary += 'Average MID of {}: {:0.2f}\n'.format(condition,condition_aveMID)

        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

        return summary


