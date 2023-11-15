from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os
import pandas as pd

dump_frames_on_disk = True

class Pixel:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SegmentedObject: 
    def __init__(self, label, name_map, img_height, img_width):
        self.class_id, coords = label
        self.index = 0
        self.class_name = name_map[self.class_id]
        self.pixels = self.decode_coco_label_values_into_pixels(coords, img_height, img_width)
        self.pixel_depths = []
        self.centroid = Pixel(0, 0) 
        self.depth_distribution = [] 
        self.is_too_far = False
        self.average_depth = 0
        self.depth_stdev = 0

    def decode_coco_label_values_into_pixels(self, coords, img_height, img_width):
        # reshape the coordinates into a numpy array
        coords = np.array(coords).reshape(-1, 2)
        # scale the coordinates to the image size
        coords[:, 0] *= img_width
        coords[:, 1] *= img_height
        coords = np.round(coords).astype(int)

        # iterate through the polygon coordinates and set the corresponding pixel coordinates to the class ID
        pixels = [(x, y) for y in range(img_height) for x in range(img_width) if cv2.pointPolygonTest(coords, (x, y), False) >= 0]
        return pixels
    
    def compute_depth_map(self, img_depth):
        self.average_depth = 0
        centroid_x, centroid_y = 0, 0

        for x, y in self.pixels: 
            depth = np.sum(img_depth[y][x])
            self.average_depth += depth
            self.depth_distribution.append(depth)
            centroid_x += x
            centroid_y += y

        if len(self.pixels):
            centroid_x //= len(self.pixels)
            centroid_y //= len(self.pixels)
            self.average_depth /= len(self.pixels)
            self.depth_stdev = np.std(np.array(self.depth_distribution))
            if self.average_depth <= 10:
                self.is_too_far = True
                if self.depth_stdev <= 10:
                    self.depth_stdev= 0xFFFFFFFF


        self.centroid = Pixel(centroid_x, centroid_y)

class DepthTracer:
    def __init__(self):
        self.objects_depth_evolution = {}
        self.previous_depth_objects = []
        self.previous_indexes = []
        self.last_index = 0
        #self.closeness_threshold = 0.025
        self.closeness_threshold = 30 #pixels

    def similar_object(self, c1, c2, img_height, img_width):
        # calculate the maximum allowed movement based on the closeness threshold
        max_movement_x = self.closeness_threshold #img_width * self.closeness_threshold
        max_movement_y = self.closeness_threshold #img_height * self.closeness_threshold
        
        # check if the distance between the centroids is within the maximum allowed movement
        return c1.class_id == c2.class_id and abs(c1.centroid.x - c2.centroid.x) <= max_movement_x and abs(c1.centroid.y - c2.centroid.y) <= max_movement_y

    def get_closest_object(self, obj, close_objects):
        #need this mechanism in order to avoid false positives
        close_objects = sorted(close_objects, key = lambda x: (obj.centroid.x - x.centroid.x) + (obj.centroid.y - x.centroid.y))
        return close_objects[0]

    def add_depth_trace(self, segmented_objects, img_height, img_width, frame_count):
        # search the objects from the current frame in the previous frame. if they are the same, add trace, else add new entry
        indexes = []
        for obj in segmented_objects:
            # for each object search the objects which are within the given threshold distance
            close_objects = [old_obj for old_obj in self.previous_depth_objects if self.similar_object(obj, old_obj, img_height, img_width)]
            if close_objects:
                # eliminate false positives
                closest_object = self.get_closest_object(obj, close_objects)
                obj.index = closest_object.index
                indexes.append(obj.index)
                key = f"{closest_object.index}_{obj.class_name}"
                print(key)
                self.objects_depth_evolution[key][1] += 1 # increase frame interval right boundary
                self.objects_depth_evolution[key] = np.append(self.objects_depth_evolution[key], obj.average_depth)

            else:
                # if we have a new object, create a new key
                self.last_index+=1
                obj.index = self.last_index
                indexes.append(obj.index)
                key = f"{obj.index}_{obj.class_name}" # ensure unique id
                self.objects_depth_evolution[key] = np.array([frame_count, frame_count, obj.average_depth])

        self.previous_depth_objects = segmented_objects
        return indexes

    def write_csv(self):
        #for depth_trace in self.objects_depth_evolution:
        np.savez('data_visualization\\depth_traces.npz', **self.objects_depth_evolution)

class DepthAnalysis:
    def __init__(self):
        self.model = YOLO("yolov8x-seg.pt")  # load an official model
        #model.train(data="coco128.yaml", epochs=1, imgsz=640)
        self.color_video = cv2.VideoCapture("challenge_color_848x480.mp4")
        self.depth_video = cv2.VideoCapture("challenge_depth_848x480.mp4")
        self.out_video_color = cv2.VideoWriter("masks\\video_segmented_color.mp4", cv2.VideoWriter_fourcc('V','X','I','D'), 30, (848, 480))
        self.out_video_depth = cv2.VideoWriter("masks\\video_segmented_depth.mp4", cv2.VideoWriter_fourcc('V','X','I','D'), 30, (848, 480))
        self.simple_logger = open("masks\\logs.txt", 'w')
        self.conf_threashold = 0.5
        
        self.sync_map_depth_timestamps = {
            "0:12.00" : 3,
            "3:00.00" : 1,
            "6:55.0" : 2.5,
        }

        self.sync_map_depth_frames = [
            (12*30, 87), # at the frame count from left (color), stop the depth camera for the number on the right frames
            (3*60*30 + 87, 1*30),
            (6*60*30+55*30 + 87+1*30, 2.5*30)
        ]

    def free_objects(self):
        self.color_video.release()
        self.depth_video.release()
        self.out_video_color.release()
        self.out_video_depth.release()
        self.simple_logger.close()

    @staticmethod
    def get_segmented_objects(result, conf_threashold):
        # extract yolo labels: following few lines are taken from source code dive
        labels = []
        det, mask = result.boxes, result.masks # getting tensors 
        for j, d in enumerate(reversed(det)):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            if conf < conf_threashold:
                continue
            seg = mask.segments[len(det) - j - 1].copy() # reversed mask.segments
            seg = seg.reshape(-1)
            labels.append((cls.item(), seg))

        img_height, img_width = np.asarray(result.boxes.orig_shape.cpu())

        depth_objects = []
        for label in labels:
            depth_object = SegmentedObject(label, result.names, img_height, img_width)
            depth_objects.append(depth_object)

        return depth_objects, img_height, img_width
        
    @staticmethod
    def test_labels_on_img(img, image_path, segmented_objects):
        #test function to see if our pixels are extracted properly
        for item in segmented_objects:
            for x, y in item.pixels:
                img[y][x] = [0,0,255]

        cv2.imwrite(image_path, img)

    def dump_images(self, i, frame_color, frame_depth, res_img, segmented_objects):
        dump_dir = "masks"
        dump_nr = str(i).zfill(5)
        dump_name = os.path.join(dump_dir, dump_nr)
        
        if dump_frames_on_disk:
            cv2.imwrite(dump_name+"depth_o.png", frame_depth)
            #self.test_labels_on_img(frame_color, dump_name+"color.png", segmented_objects)#modifies the frame
            cv2.imwrite(dump_name+"color_seg.png", res_img)
            self.test_labels_on_img(frame_depth, dump_name+"depth.png", segmented_objects)#modifies the frame

        simple_log = ', '.join(['(({:d}, {:d}) : {:.2f})'.format(s.centroid.x, s.centroid.y, s.average_depth) for s in segmented_objects])
        self.simple_logger.write(str(i).zfill(5)+": "+simple_log+"\n")
    
    def is_skipping_frame(self, i):
        is_skipping = False
        for skip in self.sync_map_depth_frames:
            if i > skip[0] and i < skip[0] + skip[1]:
                is_skipping = True
        return is_skipping

    def calculate_depth_of_objects(self):
        grabbed_color, frame_color = self.color_video.read()
        grabbed_depth, frame_depth = self.depth_video.read()
        self.simple_logger.write("frame_count: [((x,y), depth), ...]\n")

        count_color_frames = 0
        count_depth_frames = 0
        
        depth_tracer = DepthTracer()

        quick_exec = [2400, 2700]
        #quick_exec = [0, 0xFFFFFFFF]#don't skip anything
        

        while grabbed_color and grabbed_depth:
            grabbed_color, frame_color = self.color_video.read()
            count_color_frames+=1

            #sync videos with this trick (manual timestamps) - only properly syncs some portions
            skip_depth_frame = self.is_skipping_frame(count_color_frames)
            if not skip_depth_frame:
                grabbed_depth, frame_depth = self.depth_video.read()
                count_depth_frames+=1

            if count_color_frames < quick_exec[0]:
                continue
            if count_color_frames > quick_exec[1]:
                break

            # copy_frame_depth = frame_depth.copy()

            result = self.model.predict(source = frame_color, save = True, conf = 0.5, save_txt =True)[0]
            # correlate the depth with the segmentation, for frames where there is an identified object
            # don't dump frames where there is no identified object 
            if result.boxes and not skip_depth_frame:
                segmented_objects, img_height, img_width = self.get_segmented_objects(result, self.conf_threashold)
                # call compute_depth_map on every object
                list(map(lambda s : s.compute_depth_map(frame_depth), segmented_objects)) 
                # add depth info in segmented frame -> modified a bit of source code for this (it does not seem to provide support for custom labels)                
                indexes = depth_tracer.add_depth_trace(segmented_objects, img_height, img_width, count_color_frames)
                res_img = result.plot(indexes = [i for i in indexes], depths = [s.average_depth for s in segmented_objects]) 
                self.dump_images(count_color_frames, frame_color, frame_depth, res_img, segmented_objects)
            else:
                res_img = result.plot()
            

            self.out_video_color.write(res_img)
            self.out_video_depth.write(frame_depth)

        depth_tracer.write_csv()
        self.free_objects()
        

#if __name__ == '__main__':
depth_analysis = DepthAnalysis()
depth_analysis.calculate_depth_of_objects()

