from datetime import datetime
import os

from utils import *
from config import Config
from customlogger import CustomFullLogger, CustomLogger
from registration import regPair
import argparse
from PlotViewer import PlotViewer
def main():
    parser = argparse.ArgumentParser(description='Run the image registration pipeline with a specified config file.')
    parser.add_argument('-c', '--config', type=str, default='default_config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)

    #Output Folder
    directory = config.output_folder
    if not os.path.exists(directory):
        os.makedirs(directory)
    full_logger  = CustomFullLogger(name="",log_file=os.path.join(directory, "full_log.log"))
    short_logger = CustomLogger(name="",log_file=os.path.join(directory, "short_log.log"))
    if config.type == "Pair":
        full_logger.message("Starting Pairwise Registration")
        _, viewer, outImage = regPair(source=config.source_image,
                target=config.target_image,
                outFolder=config.output_folder,
                colorScale=config.color_scale,
                detector=config.detector_type,
                threshold=config.detector_threshold,
                maxkps=config.max_keypoints,
                globalreg=config.global_registration,
                numtry=config.num_try,
                finereg=config.fine_registration,
                valid_weight=config.valid_weight,
                matcher=config.matcher_type,
                nclusters=config.num_clusters,
                max_clusers=config.max_clusters,
                save=config.save_results,
                fix=config.apply_fix,
                show=config.show_results,
                logger=(full_logger,short_logger))
        viewer.save_configuration(add=config.output_folder,name="allplots")
        viewer.mainloop()
    elif config.type == "Dir":
        nclusters = None
        full_logger.message(f"Starting Directory Registration")
        stime = datetime.now()
        filenames = sorted(f for f in os.listdir(config.source_image))
        os.makedirs(os.path.join(config.output_folder,"final"), exist_ok=True)
        viewer = PlotViewer("PairWise")
        final_viewer = PlotViewer("Final")
        nclusters = config.num_clusters
        for i in range(len(filenames) - 1):
            if i == 0:
                full_logger.message("Copying first slice")
                shutil.copy2(os.path.join(config.source_image, filenames[i]), os.path.join(config.output_folder, "final", "0.jpg"))

                file1 = filenames[i]
                file2 = filenames[i + 1]
                
                file1_path = os.path.join(config.source_image, file1)
                file2_path = os.path.join(config.source_image, file2)

                fig, ax = plot_image(img=loadImage(file1_path),
                                 save=False,
                                 saveAddress="",
                                 fileName=f"{i}")
                final_viewer.add_figure(fig)

            else: 
                file2 = filenames[i + 1]
                file1_path = os.path.join(config.output_folder, str(i),"FinalImage.png")
                file2_path = os.path.join(config.source_image, file2)
            full_logger.log_section_divider()
            short_logger.log_section_divider()
            full_logger.message(f"Processing pair: {file1_path} and {file2_path}")
            short_logger.message(f"Processing pair: {file1_path} and {file2_path}")
            dir = os.path.join(config.output_folder, str(i+1))
            if not os.path.exists(dir):
                os.makedirs(dir)
            nclusters, _, outImage = regPair(source=file2_path,
                                                target=file1_path,
                                                outFolder=dir,
                                                colorScale=config.color_scale,
                                                detector=config.detector_type,
                                                threshold=config.detector_threshold,
                                                maxkps=config.max_keypoints,
                                                globalreg=config.global_registration,
                                                numtry=config.num_try,
                                                finereg=config.fine_registration,
                                                valid_weight=config.valid_weight,
                                                matcher=config.matcher_type,
                                                nclusters=nclusters,
                                                max_clusers=config.max_clusters,
                                                save=config.save_results,
                                                fix=config.apply_fix,
                                                show=config.show_results,
                                                logger=(full_logger,short_logger))
                
            fig, ax = plot_3image(images=(loadImage(file2_path),outImage,loadImage(file1_path)),
                                    save=False,
                                    saveAddress="",
                                    fileName=f"{i} and {i+1}")
            viewer.add_figure(fig)
            fig, ax = plot_image(img=outImage,
                                 save=False,
                                 saveAddress="",
                                 fileName=f"{i+1}")
            final_viewer.add_figure(fig)
        etime = datetime.now()
        full_logger.message(f"Whole Time: {(etime-stime).total_seconds()}")
        full_logger.message("Creating final Folder")
        createFinalOutputs(config.output_folder, os.path.join(config.output_folder,"final"), len(filenames))
        viewer.mainloop()
        viewer.save_configuration(add=config.output_folder, name="Pairwise")
        final_viewer.mainloop()
        final_viewer.save_configuration(add=config.output_folder, name="Final")
if __name__ == "__main__":
    main()