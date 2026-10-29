import os
from metrics import *
from model import *
from PIL import Image
import time
import numpy as np

if __name__ == "__main__":
    #ONLY CHANGE 5 LINES
    data_path = './sample_test_rsud20k' #path to dataset
    nc = 20 #number of class
    path_to_model = "./best_float32.tflite" #path to model
    model = Model(model_path=path_to_model)
    dataset_name = "rsud20k" #must be one of these names: "idd_fgvd", "rsud20k", "grazpedwri-dx", "seadronessee object detection v2", "svrdd", "btxrd", "rip current segmentation", "trashcan", "lars", "rescuenet", "loveda"
    #

    input_size = 320
    model.prepare()

    image_paths = get_image_paths(data_path, dataset_name)
    total_time = 0.0
    total_file = len(image_paths)
    results = []
    targets = get_target_from_data(data_path, dataset_name, input_size)

    for fi in image_paths:
        print(fi)
        img = Image.open(fi).resize((input_size, input_size))
        labels = targets[os.path.basename(fi).rsplit(".", 1)[0]]

        start_time = time.time()
        preds = model.predict(img)
        stop_time = time.time()
        run_time = stop_time - start_time
        total_time += run_time
        results.append((preds, labels))

    FPS = total_file/total_time
    print("Average FPS: {:.3f}".format(FPS))
    normFPS = FPS/10
    mp, mr, map50, map, f1 = eval_detection_results(results, nc, input_size)
    score = 2*normFPS*f1/(normFPS + f1)
    print("Score: {:.3f}".format(score))
    










