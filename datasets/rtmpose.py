from torch.utils.data import Dataset
import torch, os, glob
from collections import defaultdict
import os, random
from openxlab.model import download  # noqa
from mmpose.apis import MMPoseInferencer  # noqa
import pickle
import numpy as np
import copy, cv2
import torch.nn.functional as F
from PIL import Image


#download(model_repo='mmpose/RTMPose', model_name='dwpose-l')
#download(model_repo='mmpose/RTMPose', model_name='RTMW-x')
#download(model_repo='mmpose/RTMPose', model_name='RTMO-l')
#download(model_repo='mmpose/RTMPose', model_name='RTMPose-l-body8')
#download(model_repo='mmpose/RTMPose', model_name='RTMPose-m-face6')
project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
mmpose_path = project_path.split('/projects', 1)[0]

models = [
    'rtmpose | body', 'rtmo | body', 'rtmpose | face', 'dwpose | wholebody',
    'rtmw | wholebody'
]
cached_model = {model: None for model in models}

class Rtmpose(Dataset):
    
    def __init__(self, args, split='train', print_cls_idx=True):
        self.args = args
        self.split = split
          
        
        if args.olympic_split:
            self.dataset_splits = {
                'train': [1, 2, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                          41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
                'test': [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                         19, 20, 21, 22, 23, 24, 25, 26, 27]
            }
        else:
            self.dataset_splits = {
                'train': [1],
                'test': [1]
            }
        
        
        self.idx2class = {
            0: {'r_set', 'r-set'},
            1: {'l_set', 'l-set'},
            2: {'r_spike', 'r-spike'},
            3: {'l_spike', 'l-spike'},
            4: {'r_pass', 'r-pass'},
            5: {'l_pass', 'l-pass'},
            6: {'r_winpoint', 'r-winpoint'},
            7: {'l_winpoint', 'l-winpoint'}
        }
        self.class2idx = dict()
        if print_cls_idx:
            print('class index:') 
        for k in self.idx2class:
            for v in self.idx2class[k]:
                self.class2idx[v] = k
                if print_cls_idx:
                    print('{}: {}'.format(v, k))
        self.group_activities_weights = torch.FloatTensor([1., 1., 1., 1., 1., 1., 1., 1.]).cuda()
        
                    
        self.person_actions_all = pickle.load(
                open(os.path.join(self.args.dataset_dir, self.args.person_action_label_file_name), "rb"))
        self.person_actions_weights = torch.FloatTensor([0.2, 1., 1., 2., 3., 1., 4., 4., 0.2, 1.]).cuda()
        # ACTIONS = ['NA', 'blocking', 'digging', 'falling', 'jumping', 'moving', 'setting', 'spiking', 'standing', 'waiting']
        # { 'NA': 0,
        # 'blocking': 1, 
        # 'digging': 2, 
        #  'falling': 3, 
        #  'jumping': 4,
        #  'moving':5 , 
        #  'setting': 6, 
        #  'spiking': 7, 
        #  'standing': 8,
        #  'waiting': 9}
        
        
        
        self.annotations = []
        self.annotations_each_person = []
        self.clip_joints_paths = []
        self.clips = []
        if args.ball_trajectory_use:
            self.clip_ball_paths = []
        self.prepare(args.dataset_dir)
            
        if self.args.horizontal_flip_augment and self.split == 'train':
            self.classidx_horizontal_flip_augment = {
                0: 1,
                1: 0,
                2: 3,
                3: 2,
                4: 5,
                5: 4,
                6: 7,
                7: 6
            }
            if self.args.horizontal_flip_augment_purturb:
                self.horizontal_flip_augment_joint_randomness = dict()
                
        if self.args.horizontal_move_augment and self.split == 'train':
            self.horizontal_move_augment_joint_randomness = dict()
                
        if self.args.vertical_move_augment and self.split == 'train':
            self.vertical_move_augment_joint_randomness = dict()
            
        if self.args.agent_dropout_augment:
            self.agent_dropout_augment_randomness = dict()
        
        #input_type = 'image'    
        #self.predict(input_type=input_type)
        
        
        self.tdata = pickle.load(
            open(os.path.join(self.args.dataset_dir, self.args.tracklets_file_name), "rb"))

    def prepare(self, dataset_dir):
        """
        Prepare the following lists based on the dataset_dir, self.split
            - self.annotations 
            - self.annotations_each_person 
            - self.clip_joints_paths
            - self.clips
            (the following if needed)
            - self.clip_ball_paths
            - self.horizontal_flip_mask
            - self.horizontal_mask
            - self.vertical_mask
            - self.agent_dropout_mask
        """  
        annotations_thisdatasetdir = defaultdict()
        clip_joints_paths = []
        
        if(self.args.isvideo):
            videopath = os.path.join(dataset_dir, 'videos/clip')
            cvideo = 0    
            for file in os.listdir(videopath):
                self.getvideo(os.path.join(videopath, file), os.path.join(os.path.dirname(videopath), str(cvideo)), 50)
                cvideo = cvideo + 1

        for annot_file in glob.glob(os.path.join(dataset_dir, 'videos/*/annotations.txt')):
            video = annot_file.split('/')[-2]
            with open(annot_file, 'r') as f:
                lines = f.readlines()
            for l in lines:
                clip, label = l.split()[0].split('.jpg')[0], l.split()[1]
                annotations_thisdatasetdir[(video, clip)] = self.class2idx[label]  

        for video in self.dataset_splits[self.split]:
            clip_joints_paths.extend(glob.glob(os.path.join(dataset_dir, self.args.joints_folder_name, str(video), '*.pickle')))
            
        count = 0
        for path in clip_joints_paths:
            video, clip = path.split('/')[-2], path.split('/')[-1].split('.pickle')[0]
            self.clips.append((video, clip))
            self.annotations.append(annotations_thisdatasetdir[(video, clip)])
            self.annotations_each_person.append(self.person_actions_all[(int(video), int(clip))])
            if self.args.ball_trajectory_use:
                self.clip_ball_paths.append(os.path.join(dataset_dir, self.args.ball_trajectory_folder_name, video, clip + '.txt'))
            count += 1
        # print('total number of clips is {}'.format(count))

        self.clip_joints_paths += clip_joints_paths
      
        assert len(self.annotations) == len(self.clip_joints_paths)
        assert len(self.annotations) == len(self.annotations_each_person)
        assert len(self.clip_joints_paths) == len(self.clips)
        if self.args.ball_trajectory_use:
            assert len(self.clip_joints_paths) == len(self.clip_ball_paths)
        
        true_data_size = len(self.annotations)
        true_annotations = copy.deepcopy(self.annotations)
        true_annotations_each_person = copy.deepcopy(self.annotations_each_person)
        true_clip_joints_paths = copy.deepcopy(self.clip_joints_paths)
        true_clips = copy.deepcopy(self.clips)
        if self.args.ball_trajectory_use:
            true_clip_ball_paths = copy.deepcopy(self.clip_ball_paths)
        
        # if horizontal flip augmentation and is training
        if self.args.horizontal_flip_augment and self.split == 'train':
            self.horizontal_flip_mask = list(np.zeros(len(self.annotations))) + list(np.ones(true_data_size))
            self.annotations += true_annotations
            self.annotations_each_person += true_annotations_each_person
            self.clip_joints_paths += true_clip_joints_paths
            self.clips += true_clips
            if self.args.ball_trajectory_use:
                self.clip_ball_paths += true_clip_ball_paths
                
        # if horizontal move augmentation and is training
        if self.args.horizontal_move_augment and self.split == 'train':
            self.horizontal_mask = list(np.zeros(len(self.annotations))) + list(np.ones(true_data_size))
            self.annotations += true_annotations
            self.annotations_each_person += true_annotations_each_person
            self.clip_joints_paths += true_clip_joints_paths
            self.clips += true_clips
            if self.args.ball_trajectory_use:
                self.clip_ball_paths += true_clip_ball_paths
      
        # if vertical move augmentation and is training
        if self.args.vertical_move_augment and self.split == 'train':
            self.vertical_mask = list(np.zeros(len(self.annotations))) + list(np.ones(true_data_size))
            self.annotations += true_annotations
            self.annotations_each_person += true_annotations_each_person
            self.clip_joints_paths += true_clip_joints_paths
            self.clips += true_clips
            if self.args.ball_trajectory_use:
                self.clip_ball_paths += true_clip_ball_paths
                
        # if random agent dropout augmentation and is training
        if self.args.agent_dropout_augment and self.split == 'train':
            self.agent_dropout_mask = list(np.zeros(len(self.annotations))) + list(np.ones(true_data_size))
            self.annotations += true_annotations
            self.annotations_each_person += true_annotations_each_person
            self.clip_joints_paths += true_clip_joints_paths
            self.clips += true_clips
            if self.args.ball_trajectory_use:
                self.clip_ball_paths += true_clip_ball_paths
                    
    def __len__(self):
        return len(self.clip_joints_paths)
    
    def random_file_name(self, path):
        try:
            jpg_files = []
            files = os.listdir(path)
        
            for file_name in files:
                if not file_name.lower().endswith('.txt'):
                    jpg_files.append(file_name)
                    my_pic = str(random.choice(jpg_files))
            return my_pic
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return []
    
    def predict(self, input, 
                draw_heatmap=False,
                model_type='body',
                skeleton_style='mmpose',
                input_type='image'):
        """Visualize the demo images.

        Using mmdet to detect the human.
        """
        
        if model_type == 'rtmpose | face':
            if cached_model[model_type] is None:
                cached_model[model_type] = MMPoseInferencer(pose2d='face')
            model = cached_model[model_type]

        elif model_type == 'dwpose | wholebody':
            if cached_model[model_type] is None:
                cached_model[model_type] = MMPoseInferencer(
                    pose2d=os.path.join(
                        project_path, 'rtmpose/wholebody_2d_keypoint/'
                        'rtmpose-l_8xb32-270e_coco-wholebody-384x288.py'),
                    pose2d_weights='https://download.openmmlab.com/mmpose/v1/'
                    'projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-'
                    '384x288-2438fd99_20230728.pth')
            model = cached_model[model_type]

        elif model_type == 'rtmw | wholebody':
            if cached_model[model_type] is None:
                cached_model[model_type] = MMPoseInferencer(
                    pose2d=os.path.join(
                        project_path, 'rtmpose/wholebody_2d_keypoint/'
                        'rtmw-l_8xb320-270e_cocktail14-384x288.py'),
                    pose2d_weights='https://download.openmmlab.com/mmpose/v1/'
                    'projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-'
                    '384x288-20231122.pth')
            model = cached_model[model_type]

        elif model_type == 'rtmpose | body':
            if cached_model[model_type] is None:
                cached_model[model_type] = MMPoseInferencer(pose2d='rtmpose-l')
            model = cached_model[model_type]

        elif model_type == 'rtmo | body':
            if cached_model[model_type] is None:
                cached_model[model_type] = MMPoseInferencer(pose2d='rtmo')
            model = cached_model[model_type]
            draw_heatmap = False

        else:
            if model_type =='body':
                model_type = 'rtmpose | body'
                if cached_model[model_type] is None:
                    cached_model[model_type] = MMPoseInferencer(pose2d='rtmpose-l', device='cpu')
            model = cached_model[model_type]

        if input_type == 'image':

            result = next(
                model(
                    input,
                    return_vis=True,
                    draw_heatmap=draw_heatmap,
                    skeleton_style=skeleton_style))
            img = result['visualization'][0][..., ::-1]
            return img

        elif input_type == 'video':

            for _ in model(
                    input,
                    vis_out_dir='test.mp4',
                    draw_heatmap=draw_heatmap,
                    skeleton_style=skeleton_style):
                pass

            return 'test.mp4'

        return None    

    def getvideo(self, videopath, output_dir, step):
        cap = cv2.VideoCapture(videopath)
        framecount = 0
        keyframe = 0
        
        if not os.path.exists(f"{output_dir}/{keyframe:04d}"):
            os.makedirs( f"{output_dir}/{keyframe:04d}")
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = f"{output_dir}/{keyframe:04d}/{framecount:04d}.jpg"
            cv2.imwrite(frame_path, frame)
            image = Image.open(frame_path)
            image.thumbnail([self.args.image_w, self.args.image_h])
            image.save(frame_path, optimize=True, quality=85)
            framecount += 1    
            if(framecount%step == 0):
                keyframe += step
                if not os.path.exists(f"{output_dir}/{keyframe:04d}"):
                    os.makedirs( f"{output_dir}/{keyframe:04d}")
        cap.release()
        print(f"Frames extracted: {framecount}")
            

    
    def __getitem__(self, index):
        # index = 0
        current_joint_feats_path = self.clip_joints_paths[index] 
        (video, clip) = self.clips[index]
        label = self.annotations[index]
        person_labels = self.annotations_each_person[index]
        
        joint_raw = pickle.load(open(current_joint_feats_path, "rb"))
        # joint_raw: T: (N, J, 3)
        # 3: [joint_x, joint_y, joint_type]
        
        frames = sorted(joint_raw.keys())[self.args.frame_start_idx:self.args.frame_end_idx+1:self.args.frame_sampling]
                        
        person_labels = torch.LongTensor(person_labels[frames[0]].squeeze())  # person action remains to be the same across all frames 
        # person_labels: (N, )
        ball_feats = torch.zeros(len(frames), 6)
        del frames[0:9]              
        for tidx, frame in enumerate(frames):
           # print(os.path.join(self.args.dataset_dir,  video, clip, str(frame) + '.jpg'))    
            out = self.predict(input = os.path.join(self.args.dataset_dir, 'videos',  video, clip, str(frame) + '.jpg'))
            #out = self.predict(input =os.getcwd(), 'datasets/zip/volleyball/videos/1/9530/9510.jpg')

        out_copy = np.copy(out)
        out = torch.tensor(out_copy, dtype=torch.float32)
        assert not torch.isnan(out).any()
        return out, label, video, clip, person_labels, ball_feats