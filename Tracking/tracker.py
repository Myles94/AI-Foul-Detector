from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort_realtime.deep_sort.detection import Detection
import numpy as np
import os

class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = None

        encoder_model_filename = 'C:\\Users\\tarhe\\Desktop\\AI Foul Detector\\Tracking\\model_data\\mars-small128.pb'
        assert os.path.exists(encoder_model_filename), "Encoder model file not found!"

        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSort(metric)
        self.encoder = self.create_box_encoder(encoder_model_filename)

    def create_box_encoder(self, model_filename, batch_size=1):
        # This should be implemented according to how `deep_sort_realtime` expects the encoder to be created
        from deep_sort_realtime.deep_sort import generate_detections
        return generate_detections.create_box_encoder(model_filename, batch_size)

    def update(self, frame, detections):
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self.update_tracks()
            return

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            id = track.track_id
            tracks.append(Track(id, bbox))
        self.tracks = tracks

class Track:
    track_id = None
    bbox = None

    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox
