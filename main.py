import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from utils import *
from datetime import datetime
import os

def main():
    parser = argparse.ArgumentParser(description='Keypoint Extraction, Matching, and transformation')
    parser.add_argument('-type', '--type', type=str, required=False, default="Pair", help='Pair/Dir')
    parser.add_argument('-s', '--source', type=str, required=True, help='Path to the source image')
    parser.add_argument('-t', '--target', type=str, required=True, help='Path to the target image')
    parser.add_argument('-o', '--out', type=str, required=True, help='Path to the Output Folder')
    parser.add_argument('-c', '--color', type=str, required=False,default="RGB", help='RGB/Gray')
    parser.add_argument('-d', '--detector', type=str, required=False, default="AKAZE", help='Detector: SIFT, AKAZE, SUPER')
    parser.add_argument('-th', '--threshold', type=float, required=False, default=0.0001, help='Detector Threshold')
    parser.add_argument('-maxkps', '--maxkps', type=int, required=False, default=2048, help='Maximum Number of Keypoints (Only for superpoint)')
    parser.add_argument('-m', '--matcher', type=str, required=False,default="L2", help='Matcher: L2, Hamming, Light')
    parser.add_argument('-n', '--nclusters', type=int, required=True, help="Number of Clusters")
    
    parser.add_argument('--save', action='store_false', help="Save Outputs")
    parser.add_argument('--show', action='store_false', help="show Outputs")
    parser.add_argument('--fix', action='store_false', help='fix Outputs')

    args = parser.parse_args()
    
    #Output Folder
    directory = args.out
    if not os.path.exists(directory):
        os.makedirs(directory)
    log_file_path = os.path.join(directory, "script_log.log")

    logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path, mode='w'),
                        logging.StreamHandler()
                    ])
    if args.type == "Pair":
        logging.info(f"Starting Pairwise Registration")
        regPair(source=args.source,
                target=args.target,
                outFolder= args.out,
                colorScale=args.color,
                detector=args.detector,
                threshold=args.threshold,
                maxkps=args.maxkps,
                matcher=args.matcher,
                nclusters=args.nclusters,
                save=args.save,
                fix= args.fix,
                show=args.show,
                logging = logging)
    elif args.type == "Dir":
        logging.info(f"Starting Directory Registration")
        stime = datetime.now()
        filenames = sorted(f for f in os.listdir(args.source))
        os.makedirs(os.path.join(args.out,"final"), exist_ok=True)
        for i in range(len(filenames) - 1):
            if i == 0:
                logging.info("Copying first slice")
                shutil.copy2(os.path.join(args.source, filenames[i]), os.path.join(args.out,"final","0.jpg"))

                file1 = filenames[i]
                file2 = filenames[i + 1]
                
                file1_path = os.path.join(args.source, file1)
                file2_path = os.path.join(args.source, file2)
                
            else: 
                file2 = filenames[i + 1]
                file1_path = os.path.join(args.out, str(i),"SourceTransformed_fineT.jpg")
                file2_path = os.path.join(args.source, file2)
                
            logging.info(f"Processing pair: {file1_path} and {file2_path}")
            dir = os.path.join(args.out, str(i+1))
            if not os.path.exists(dir):
                os.makedirs(dir)
            regPair(source=file2_path,
                target=file1_path,
                outFolder= dir,
                colorScale=args.color,
                detector=args.detector,
                threshold=args.threshold,
                matcher=args.matcher,
                nclusters=args.nclusters,
                maxkps=args.maxkps,
                save=args.save,
                fix= args.fix,
                show=args.show,
                logging = logging
                )

        etime = datetime.now()
        logging.info(f"Whole Time: {(etime-stime).total_seconds()}")
        logging.info("Creating final Folder")
        createFinalOutputs(args.out, os.path.join(args.out,"final"), len(filenames))
if __name__ == "__main__":
    main()