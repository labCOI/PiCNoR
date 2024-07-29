import yaml

class Config:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Operation settings
        operation = config.get('operation', {})
        self.type = operation.get('type', 'Pair')
        self.source_image = operation.get('source_image')
        self.target_image = operation.get('target_image')
        self.output_folder = operation.get('output_folder')

        # Image processing settings
        image_processing = config.get('image_processing', {})
        self.color_scale = image_processing.get('color_scale', 'RGB')
        self.detector_type = image_processing.get('detector_type', 'AKAZE')
        self.detector_threshold = image_processing.get('detector_threshold', 0.0001)
        self.max_keypoints = image_processing.get('max_keypoints', 2048)

        # Registration settings
        registration = config.get('registration', {})
        self.global_registration = registration.get('global', True)
        self.num_try = registration.get('retry_numbers', 10)
        self.fine_registration = registration.get('fine', True)
        self.num_clusters = registration.get('num_clusters', 'auto')
        self.valid_weight = registration.get('valid_range_weight', 0.5)
        self.max_clusters = registration.get('max_clusters', 20)

        # Matching settings
        matching = config.get('matching', {})
        self.matcher_type = matching.get('matcher_type', 'L2')
        
        # Output settings
        output_settings = config.get('output_settings', {})
        self.save_results = output_settings.get('save_results', True)
        self.show_results = output_settings.get('show_results', True)
        self.apply_fix = output_settings.get('apply_fix', True)
