from allegroai import DatasetVersion, DataView, IterationOrder, Dataset, Task, SingleFrame, InputModel
import csv
from PIL import Image
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import torch
import torchvision.transforms as transforms
#from yolov5 import utils
from ultralytics import YOLO
import torchvision
import logging
import time
import base64
import os
import io
import datetime
from io import BytesIO
from urllib.parse import urlparse
import datetime
from concurrent.futures import ThreadPoolExecutor
import os
import argparse
import requests
from pathlib import Path
from urllib.parse import unquote
import re


class FrameImport:

    blob_service_client : BlobServiceClient
    source_container_client : ContainerClient
    dest_container_client : ContainerClient

    download_urls: list[str] = []
    upload_urls: list[str] = []
    
    OD_MODELS: list[str] = []
    CLASSIFIER_MODELS: list[str] = []
    OD_CONFIDENCE : float
    CLASSIFIER_CONFIDENCE : float

    frames: list[SingleFrame] = []
    IMAGE_RESIZE = False
    IMAGE_UPLOAD = False
    MAX_IMAGES_TO_PROCESS : int = 0
    IMAGE_SIZE = (832, 832)
    LOCAL_DOWNLOAD_PATH : str
    BLOB_DEST_FOLDER_NAME : str

    clearml_dataset : Dataset

    PROCESS_CSV_SOURCE : bool
    PROCESS_BLOB_SOURCE : bool
    CSV_SOURCE_FILENAME : str

    clearml_task : Task

    BLOB_SOURCE_FOLDER_NAME : str
    SKIP_LABELS: list[str] = []
    SAVE_CROPS : bool
    IGNORE_EMPTY_IMAGES : bool
    ENABLE_CLASSIFIER : bool
    RUN_CLASSIFIER_ON_LABELS: list[str] = []

    OVERRIDE_LABEL : str
    OVERRIDE_LABEL_WITH_PATH : bool = False

    def configure_models(self, YOLOV5_OD_MODEL_URLS, YOLOV8_CLASSIFIER_MODEL_URLS):

        for model_url in YOLOV5_OD_MODEL_URLS:
            print(f'Loading YOLOv5 model from {model_url}')
            self.OD_MODELS.append(torch.hub.load('ultralytics/yolov5', 'custom', path=model_url, force_reload=True, trust_repo=True))

        for model_url in YOLOV8_CLASSIFIER_MODEL_URLS:
            print(f'Loading YOLOv8 model from {model_url}')
            self.CLASSIFIER_MODELS.append(YOLO(model_url))

    def configure_task(self, CLEARML_TASK_NAME : str, CLEARML_PROJECT_NAME : str):
        self.clearml_task = Task.init(project_name=CLEARML_PROJECT_NAME,
            task_name=CLEARML_TASK_NAME,
            task_type=Task.TaskTypes.data_processing)

    def configure_dataset(self, CLEARML_DELETE_EXISTING_DATASET : bool, CLEARML_PROJECT_NAME : str, CLEARML_DATASET_NAME : str):

        if CLEARML_DELETE_EXISTING_DATASET:
            try:
                Dataset.delete(dataset_name=CLEARML_DATASET_NAME, dataset_project=CLEARML_PROJECT_NAME, delete_all_versions=True, force=False)
            except:
                print('Dataset does not exist, skipping delete')
        
        try:
            self.clearml_dataset = Dataset.create(dataset_name=CLEARML_DATASET_NAME, dataset_project=CLEARML_PROJECT_NAME)
        except:
            print('Dataset already exists, getting current version')

        self.clearml_dataset = DatasetVersion.get_current(dataset_name=CLEARML_DATASET_NAME, dataset_project=CLEARML_PROJECT_NAME)

    def configure_blob_clients(self, BLOB_CONNECTION_STRING : str, BLOB_SOURCE_CONTAINER : str, BLOB_DESTINATION_CONTAINER : str):

        self.blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        self.source_container_client = self.blob_service_client.get_container_client(BLOB_SOURCE_CONTAINER)

        if BLOB_DESTINATION_CONTAINER:
            self.dest_container_client = self.blob_service_client.get_container_client(BLOB_DESTINATION_CONTAINER)

    def resize_and_pad(self, image, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
        try:
            width, height = image.size
            aspect_ratio = width / height

            if aspect_ratio > target_size[0] / target_size[1]:  # Image is wider than target
                new_width = target_size[0]
                new_height = int(target_size[0] / aspect_ratio)
            else:  # Image is taller than target or aspect ratios are equal
                new_height = target_size[1]
                new_width = int(target_size[1] * aspect_ratio)

            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            result_image = Image.new('RGB', target_size, (0, 0, 0))  # Black background
            result_image.paste(resized_image, ((target_size[0] - new_width) // 2,
                                                (target_size[1] - new_height) // 2))

            return result_image

        except Exception as e:
            print(f"resize_and_pad exception {e}")

            return image

    def crop_image(self, img, left, upper, right, lower):
        # Crop the portion of the image
        try:
            box = (left, upper, right, lower)
            return img.crop(box)
        except Exception as e:
            print(f'Crop exception {e}')
            return img

    def download_image(self, url):
        filename = os.path.basename(url)
        filepath = os.path.join(self.LOCAL_DOWNLOAD_PATH, filename)

        if os.path.exists(filepath):
            print(f'Local image already exists, skipping download {filepath}')
            return

        print(f'Downloading {filepath} from {url}')

        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad responses (e.g., 404)

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)

        print(f'Download {filepath} completed')

    def find_partial_match(self, my_list, partial_string):
        matches = [item for item in my_list if partial_string.lower() in item.lower()]
        return matches[0] if matches else None

    def upload_image(self, local_file_name):
        image_stream = BytesIO()

        img = Image.open(local_file_name)
        img.save(image_stream, format="JPEG")
        image_stream.seek(0)

        new_blob_name = local_file_name.split('/')[-1]
        if len(self.BLOB_DEST_FOLDER_NAME) > 0:
            new_blob_name = self.BLOB_DEST_FOLDER_NAME + '/' + new_blob_name

        print(f'Uploading BLOB {new_blob_name}')

        dest_blob_client = self.dest_container_client.get_blob_client(new_blob_name)

        try:
            dest_blob_client.upload_blob(image_stream.read(), blob_type="BlockBlob", overwrite=True)

            if self.find_partial_match(self.upload_urls, dest_blob_client.url) == None:
                self.upload_urls.append(dest_blob_client.url)

        except Exception as e:
            print(f'upload blob failed {e}')

    def get_blob_list(self):
        """Gets a list of blobs from specified blob source folder"""
        print(f'Getting a max of {self.MAX_IMAGES_TO_PROCESS} download URLS from blob source')

        try:
            if self.BLOB_SOURCE_FOLDER_NAME:
                blob_list = self.source_container_client.list_blobs(name_starts_with=self.BLOB_SOURCE_FOLDER_NAME) #  + '/'
            else:
                blob_list = self.source_container_client.list_blobs() #  + '/'
        except Exception as e:
            print(f'Blob list exception {e}')
            return

        count = 0

        for blob in blob_list:
            blob_url = self.source_container_client.get_blob_client(blob=blob.name).url
            count += 1

            if blob_url in self.download_urls:
                continue

            if self.MAX_IMAGES_TO_PROCESS > 0:
                if count >= self.MAX_IMAGES_TO_PROCESS:
                    break

            self.download_urls.append(blob_url)

            print(f'added download url {blob_url}')

    def generate_filename_with_ms(self):
        """Generates a filename in the format: YYYY-MM-DD_HH-MM-SS-ms.txt"""
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d_%H-%M-%S-%f") + ".jpg"

    def process_csv(self):
        """Gets a list of image URLs from column 0 of a CSV"""
        print(f'Getting a max of {self.MAX_IMAGES_TO_PROCESS} download URLS from CSV {self.CSV_SOURCE_FILENAME}')

        urls = []

        #utf-8-sig is for windows compat
        with open(self.CSV_SOURCE_FILENAME, 'r', encoding="utf-8-sig") as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            urls = [row[0] for row in rows]

        print(f'{len(urls)} urls...')

        count = 0

        for url in urls:
            #if not os.path.exists(LOCAL_DOWNLOAD_PATH + url.split('/')[-1]):

            if url in self.download_urls:
                continue

            count += 1

            if self.MAX_IMAGES_TO_PROCESS > 0:
                if count > self.MAX_IMAGES_TO_PROCESS:
                    break

            self.download_urls.append(url)

    def download_images_from_sources(self):

        if self.PROCESS_BLOB_SOURCE:
            self.get_blob_list()

        if self.PROCESS_CSV_SOURCE:
            self.process_csv()

        print(f'Downloading {len(self.download_urls)} images')

        with ThreadPoolExecutor(max_workers=10) as executor:
            for i in range(0, len(self.download_urls), 200):
                chunk = self.download_urls[i:i+200]
                executor.map(self.download_image, chunk)

    def upload_images(self):
        files_to_upload = []

        for filename in os.listdir(self.LOCAL_DOWNLOAD_PATH):
            filename = self.LOCAL_DOWNLOAD_PATH + filename
            files_to_upload.append(filename)

        with ThreadPoolExecutor(max_workers=10) as executor:
            for i in range(0, len(files_to_upload), 200):
                chunk = files_to_upload[i:i+200]
                executor.map(self.upload_image, chunk)

    def annotate_yolov5(self, img, frame, label_override=None):

        for model in self.OD_MODELS:
            results = model(img, augment=True)

            if len(results.xyxy[0]) > 0:
                for det in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = map(float, det)
                    label = results.names[int(cls)]
                    print(f'{label} {conf} => {int(x1)} {y1} {x2} {y2}')

                    if label in self.SKIP_LABELS:
                        print(f'Label is in skip list, skipping {label}')
                        continue

                    if conf < self.OD_CONFIDENCE:
                        print(f'Label is below confidence threshold, skipping {label} {conf} < {self.OD_CONFIDENCE}')
                        continue

                    if self.OVERRIDE_LABEL_WITH_PATH:
                        if label == self.OVERRIDE_LABEL:
                            print(f"Overriding label {label} with {label_override}")
                            frame.add_annotation(box2d_xywh=(x1, y1, x2 - x1, y2 - y1), labels=[label_override], confidence=round(conf, 2))
                        else:
                            frame.add_annotation(box2d_xywh=(x1, y1, x2 - x1, y2 - y1), labels=[label], confidence=round(conf, 2))
                    else:
                        frame.add_annotation(box2d_xywh=(x1, y1, x2 - x1, y2 - y1), labels=[label], confidence=round(conf, 2))
                        
                    img_crop = self.crop_image(img, int(x1), int(y1), int(x2), int(y2))
                    img_crop = self.resize_and_pad(img_crop, target_size=(224, 224))

                    if self.SAVE_CROPS:
                        try:
                            img_crop.Save("crops/" + frame.id + "-" + label + ".jpg")
                        except Exception as e:
                            print(f"Crop save failed {e}")

                    if self.ENABLE_CLASSIFIER:
                        if label in self.RUN_CLASSIFIER_ON_LABELS:
                            frame = self.run_classifiers(img_crop, frame, label, int(x1), int(y1), int(x2), int(y2))

        return frame

    def run_classifiers(self, img, frame, label, x1, y1, x2, y2):
        print(f'Running classifier on {label} crop {img.width}x{img.height}')
        
        transform = transforms.Compose([
            # transforms.Resize((img.height, img.width)),  # Resize to calculated dimensions
            # transforms.Pad((0, 0, max(0, 640 - img.width), max(0, 640 - img.height)),
            #                 padding_mode='constant'),  # Add padding if needed
            # transforms.CenterCrop((640, 640)),  # If padding, center the image
            transforms.ToTensor()
        ])

        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)

        for model in self.CLASSIFIER_MODELS:
            try:

                results = model(img_tensor)

                for result in results:
                    probs = result.probs  # Probs object for classification outputs

                conf = float(probs.top1conf)

                # if conf < self.CLASSIFIER_CONFIDENCE:
                #     continue

                label = result.names[probs.top1]

                frame.add_annotation(box2d_xywh=(x1, y1, x2 - x1, y2 - y1), labels=['Namibia-' + label], confidence=round(conf, 2))

                if self.SAVE_CROPS:
                    os.makedirs(self.LOCAL_DOWNLOAD_PATH + 'crops/' + label + '/', exist_ok=True)
                    img.save(self.LOCAL_DOWNLOAD_PATH + 'crops/' + label + '/' + frame.preview_uri.split('/')[-1])

                # for i in range(5):
                #   conf = float(probs.top5conf[i])
                #   label = result.names[probs.top5[i]]
                #   print(f'{label}, {round(conf, 2)}')
                #   frame.add_annotation(box2d_xywh=(x1, y1, x2 - x1, y2 - y1), labels=[label], confidence=round(conf, 2))

            except Exception as e:
                print(f'YOLOv8 exception => {e}')

        return frame

    def resize_images(self):
        print('Resizing images')
        for filename in os.listdir(self.LOCAL_DOWNLOAD_PATH):
            try:
                filename = self.LOCAL_DOWNLOAD_PATH + filename
                new_img = Image.open(filename)

                if new_img.width == self.IMAGE_SIZE and new_img.height == self.IMAGE_SIZE:
                    continue
                
                new_img = self.resize_and_pad(new_img)
                new_img.save(filename)
                print(f'Resized {filename}')
            except Exception as e:
                print(f'Exception resizing {filename}')

    def process_images(self, upload_urls, download_urls, frames):
        """Main loop for processing all images that were downloaded"""
        # We now have all of our sources images saved locally and resized if specified

        count = 0

        print(f'Processing {len(os.listdir(self.LOCAL_DOWNLOAD_PATH))} images')
        
        for filename in os.listdir(self.LOCAL_DOWNLOAD_PATH):

            if self.MAX_IMAGES_TO_PROCESS > 0:
                if count > self.MAX_IMAGES_TO_PROCESS:
                    break

            count += 1

            try:
                print('###################################################################')
                print(f'{count} - local filename {filename}')

                img_source_upload_url = self.find_partial_match(upload_urls, filename)
                img_source_download_url = self.find_partial_match(download_urls, filename)

                print(f'IMAGE_UPLOAD => {self.IMAGE_UPLOAD}')
                print(f'img_source_upload_url => {img_source_upload_url}')
                print(f'img_source_download_url => {img_source_download_url}')

                if self.IMAGE_UPLOAD:
                    img_source_url = img_source_upload_url
                else:
                    img_source_url = img_source_download_url

                # if len(img_source_url) == 0:
                #   print(f'No matching source URL for {filename}')
                #   continue

                filename = self.LOCAL_DOWNLOAD_PATH + filename

                print(f'img_source_url => {img_source_url}')
                print(f'filename {filename}')

                img = Image.open(filename)

                #metadata = find_partial_match(download_urls, filename)

                if img_source_download_url:
                    try:
                        metadata = unquote(img_source_download_url.split("/")[-2])
                    except:
                        metadata = img_source_download_url.split("/")[-2]

                    print(f'metadata {metadata}')

                id = img_source_url.split(".net/")[-1]
                id = id.lower().replace(".jpg","").replace(".jpeg","").replace(".JPG","")
                clean_id = re.sub(r'[^a-zA-Z0-9-_]', '_', id)
                print(f'ID {clean_id}')

                frame = SingleFrame(
                        id=clean_id,
                        source=img_source_url,
                        width=img.width,
                        height=img.height,
                        preview_uri=img_source_url,
                        metadata={'Label':metadata},
                    )

                if self.OVERRIDE_LABEL_WITH_PATH:
                    print(f'Overriding label {self.OVERRIDE_LABEL} with path {metadata}')
                    frame = self.annotate_yolov5(img, frame, metadata)
                else:
                    frame = self.annotate_yolov5(img, frame)

                if self.IGNORE_EMPTY_IMAGES: # IF there are not any annotations for an image, dont add to ClearML dataset
                    if len(frame.annotations) > 0:
                        frames.append(frame)
                else:
                  frames.append(frame)

                if len(frames) > 50:
                    self.clearml_dataset.add_frames(frames)
                    frames.clear()

            except Exception as e:
                print(f'exception => {e}')


    def main(self):

        self.download_images_from_sources()

        if self.IMAGE_RESIZE:
            self.resize_images()

        if self.IMAGE_UPLOAD:
            self.upload_images()

        self.process_images(self.upload_urls, self.download_urls, self.frames)

        self.clearml_dataset.add_frames(self.frames)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="This creates a ClearML hyperdataset sourcing images from a blob container and/or a CSV file.")

    # parser.add_argument("--yolov5_od_models", type=lambda s: s.split(","), default="https://mlopsprod.blob.core.windows.net/models/md_v5a.0.0.pt", help="Comma seperated list of YOLOv5 object detection pt URLs")
    parser.add_argument("--yolov5_od_models", type=list[str], default=['https://mlopsprod.blob.core.windows.net/models/md_v5a.0.0.pt'], help="List of YOLOv5 object detection pt URL strings")
    parser.add_argument("--yolov5_od_confidence", type=float, default=0.25, help="Minimum confidence of a detected label that will be sent to the classifiers")

    parser.add_argument("--enable_classifiers", action='store_true', help="Enable classifiers on specified object detection labels")
    parser.add_argument("--yolov8_classifier_models", type=list[str], default=["https://labelstudioqa.blob.core.windows.net/models/namib_desert_v1.pt"], help="List of YOLOv8 classifier pt URL strings")
    parser.add_argument("--yolov8_classifier_confidence", type=float, default=0.25, help="Minimum confidence of a detected label that will be added to a frame")
    
    parser.add_argument("--clearml_delete_existing_dataset", action='store_true', help="Delete existing dataset before importing, otherwise create a new version")
    parser.add_argument("--clearml_project", type=str, help="ClearML project name", required=True)
    parser.add_argument("--clearml_dataset", type=str, help="ClearML hyperdataset name", required=True)

    parser.add_argument("--clearml_task", type=str, help="ClearML task name", required=True)
   
    parser.add_argument("--max_images", type=int, default=0, help="Maximum number of images to process, if set to 0, will process all images in the blob source and/or CSV file")
    parser.add_argument("--image_size", type=int, default=832, help="Resize input images to this size 832x832")
    parser.add_argument("--resize_images", action='store_true', help="Resize images")
    parser.add_argument("--save_crops", action='store_true', help="Save crops")
    parser.add_argument("--ignore_empty_images", action='store_true', help="Ignore images that are empty")
    parser.add_argument("--upload_images", action='store_true', help="Uploads resized images to destination blob container/path")

    parser.add_argument("--skip_detector_labels", type=list[str], default=["person", "vehicle"], help="List of object detection label strings to skip importing")
    parser.add_argument("--run_classifier_labels", type=list[str], default=["animal"], help="List of label strings to run the classifiers on")

    parser.add_argument("--blob_connection_string", type=str, help="Blob connection string", required=True)
    parser.add_argument("--blob_source_container", type=str, help="Blob source container", required=True)
    parser.add_argument("--blob_source_folder", type=str, help="Blob source folder", required=False)
    parser.add_argument("--blob_destination_container", type=str, help="Blob destination container")
    parser.add_argument("--blob_destination_folder", type=str, help="Blob destination folder")

    parser.add_argument("--process_blob_source", action='store_true', help="Process images from blob source")
    parser.add_argument("--process_csv_source", action='store_true', help="Process images from CSV source")

    parser.add_argument("--csv_filename", type=str, default="import_urls.csv", help="Local path to CSV file")

    parser.add_argument("--local_cache_path", type=str, default="image_cache/", help="Local image cache path")

    parser.add_argument("--override_label", type=str, default="animal", help="If an an animal is detected, override the label with the folder name")
    parser.add_argument("--override_label_with_path", action='store_true', help="Local image cache path")

    args = parser.parse_args()

    Importer = FrameImport()

    Importer.configure_task(CLEARML_TASK_NAME=args.clearml_task, CLEARML_PROJECT_NAME=args.clearml_project)

    Importer.MAX_IMAGES_TO_PROCESS = args.max_images
    Importer.PROCESS_BLOB_SOURCE = args.process_blob_source
    Importer.PROCESS_CSV_SOURCE = args.process_csv_source
    Importer.LOCAL_DOWNLOAD_PATH = args.local_cache_path
    Importer.CSV_SOURCE_FILENAME = args.csv_filename
    Importer.CLASSIFIER_CONFIDENCE = args.yolov8_classifier_confidence
    Importer.OD_CONFIDENCE = args.yolov5_od_confidence
    Importer.BLOB_DEST_FOLDER_NAME = args.blob_destination_folder
    Importer.BLOB_SOURCE_FOLDER_NAME = args.blob_source_folder
    Importer.IMAGE_UPLOAD = args.upload_images
    Importer.SKIP_LABELS = args.skip_detector_labels
    Importer.SAVE_CROPS = args.save_crops
    Importer.IGNORE_EMPTY_IMAGES = args.ignore_empty_images
    Importer.RUN_CLASSIFIER_ON_LABELS = args.run_classifier_labels
    Importer.ENABLE_CLASSIFIER = args.enable_classifiers

    Importer.OVERRIDE_LABEL_WITH_PATH = args.override_label_with_path

    if args.override_label_with_path:
        Importer.OVERRIDE_LABEL = args.override_label

    Importer.configure_dataset(CLEARML_DELETE_EXISTING_DATASET=args.clearml_delete_existing_dataset, CLEARML_PROJECT_NAME=args.clearml_project, CLEARML_DATASET_NAME=args.clearml_dataset)
    Importer.configure_models(YOLOV5_OD_MODEL_URLS=args.yolov5_od_models, YOLOV8_CLASSIFIER_MODEL_URLS=args.yolov8_classifier_models)

    Importer.configure_blob_clients(BLOB_CONNECTION_STRING=args.blob_connection_string, BLOB_SOURCE_CONTAINER=args.blob_source_container, BLOB_DESTINATION_CONTAINER=args.blob_destination_container)

    directory = Path(args.local_cache_path)
    directory.mkdir(exist_ok=True)
    print("Directory created successfully")

    Importer.main()
