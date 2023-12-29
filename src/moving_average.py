class MovingAverageFilter:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.classes = []

    def update(self, new_class):
        self.classes.append(new_class)
        if len(self.classes) >  self.window_size:
            self.classes.pop(0)

    def get_average(self):
        if not self.classes:
            return None
        return round(sum(self.classes) / len(self.classes))

def class_inference_and_smoothing(preprocessed_frame, model, ma_filter, conf_threshold=0.3, eraser_class_id=3):
    # Model inference
    results = model.predict(preprocessed_frame, conf=conf_threshold)

    boxes = results[0].boxes.xywh.cpu()
    clss = results[0].boxes.cls.cpu().tolist()
    confs = results[0].boxes.conf.float().cpu().tolist()

    if len(boxes) == 0:
        ma_filter.update(-1)
        return {'Original Class': -1, 'Adjusted Class': ma_filter.get_average(), 'Confidence Score': 0, 'Bounding Box': []}
    else:
        max_conf_idx = confs.index(max(confs))
        original_class = int(clss[max_conf_idx])

        # Bypass the moving average filter for the eraser class
        if original_class == eraser_class_id:
            return {'Original Class': original_class, 'Adjusted Class': original_class, 'Confidence Score': confs[max_conf_idx], 'Bounding Box': boxes[max_conf_idx].tolist()}

        ma_filter.update(original_class)
        adjusted_class = ma_filter.get_average()

        return {'Original Class': original_class, 'Adjusted Class': adjusted_class, 'Confidence Score': confs[max_conf_idx], 'Bounding Box': boxes[max_conf_idx].tolist()}
