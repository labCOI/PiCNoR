from datetime import datetime
import os

from utils import *
from config import Config
from customlogger import CustomFullLogger, CustomLogger
from registration import regPair

def main():
    config = Config('default_config.yaml')
    #Output Folder
    directory = config.output_folder
    if not os.path.exists(directory):
        os.makedirs(directory)
    full_logger  = CustomFullLogger(log_file=os.path.join(directory, "full_log.log"))
    short_logger = CustomLogger(log_file=os.path.join(directory, "full_log.log"))
    if config.type == "Pair":
        full_logger.message("Starting Pairwise Registration")
        regPair(source=config.source_image,
                target=config.target_image,
                outFolder=config.output_folder,
                colorScale=config.color_scale,
                detector=config.detector_type,
                threshold=config.detector_threshold,
                maxkps=config.max_keypoints,
                matcher=config.matcher_type,
                nclusters=config.num_clusters,
                save=config.save_results,
                fix=config.apply_fix,
                show=config.show_results,
                logger=(full_logger,short_logger))
    elif config.type == "Dir":
        nclusters = None
        full_logger.message(f"Starting Directory Registration")
        stime = datetime.now()
        filenames = sorted(f for f in os.listdir(config.source_image))
        os.makedirs(os.path.join(config.output_folder,"final"), exist_ok=True)
        for i in range(len(filenames) - 1):
            if i == 0:
                full_logger.message("Copying first slice")
                shutil.copy2(os.path.join(config.source_image, filenames[i]), os.path.join(config.output_folder, "final", "0.jpg"))

                file1 = filenames[i]
                file2 = filenames[i + 1]
                
                file1_path = os.path.join(config.source_image, file1)
                file2_path = os.path.join(config.source_image, file2)
                
            else: 
                file2 = filenames[i + 1]
                file1_path = os.path.join(config.output_folder, str(i),"SourceTransformed_fineT.jpg")
                file2_path = os.path.join(config.source_image, file2)
                
            full_logger.message(f"Processing pair: {file1_path} and {file2_path}")
            dir = os.path.join(config.output_folder, str(i+1))
            if not os.path.exists(dir):
                os.makedirs(dir)
            if nclusters == None:
                nclusters = regPair(source=file2_path,
                                target=file1_path,
                                outFolder=dir,
                                colorScale=config.color_scale,
                                detector=config.detector_type,
                                threshold=config.detector_threshold,
                                matcher=config.matcher_type,
                                nclusters=config.num_clusters,
                                maxkps=config.max_keypoints,
                                save=config.save_results,
                                fix=config.apply_fix,
                                show=config.show_results,
                                logger=(full_logger,short_logger))
            else:
                _ = regPair(source=file2_path,
                                target=file1_path,
                                outFolder=dir,
                                colorScale=config.color_scale,
                                detector=config.detector_type,
                                threshold=config.detector_threshold,
                                matcher=config.matcher_type,
                                nclusters=config.num_clusters,
                                maxkps=config.max_keypoints,
                                save=config.save_results,
                                fix=config.apply_fix,
                                show=config.show_results,
                                logger=(full_logger,short_logger))

        etime = datetime.now()
        full_logger.message(f"Whole Time: {(etime-stime).total_seconds()}")
        full_logger.message("Creating final Folder")
        createFinalOutputs(config.output_folder, os.path.join(config.output_folder,"final"), len(filenames))
if __name__ == "__main__":
    main()