#!/usr/bin/env python3
# To run this model, you will need to add a PIR sensor to your Google AIY.
# You can also attach a lipo battery to run the Google AIY
import argparse
import collections
import io
import logging
import math
import os
import queue
import signal
import threading
import time
import sys

from aiy._drivers._hat import get_aiy_device_name
from aiy.toneplayer import TonePlayer
from aiy.vision.leds import Leds
from aiy.vision.inference import ImageInference
from aiy.vision.models import image_classification
from aiy.vision.leds import PrivacyLed
from aiy.vision.pins import (PIN_A)

from contextlib import contextmanager
from gpiozero import Button
from picamera import PiCamera
from gpiozero import SmoothedInputDevice
from gpiozero import InputDevice
from gpiozero.mixins import GPIOQueue
from gpiozero import EventsMixin

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@contextmanager
def stopwatch(message):
    try:
        logger.info('%s...', message)
        begin = time.time()
        yield
    finally:
        end = time.time()
        logger.info('%s done. (%fs)', message, end - begin)

class Actor(object):
    def __init__(self):
        print('initializing')
        self._requests = queue.Queue()
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def _run(self):
        while True:
            print('this infinite loop')
            request = self._requests.get()
            print('request', request)
            if request is None:
                break
            self.process(request)
            self._requests.task_done()

    def join(self):
        self._thread.join()

    def stop(self):
        print('stop')
        self._requests.put(None)

    def process(self, request):
        pass

    def submit(self, request):
        print('submit')
        self._requests.put(request)

class ImageClassification(Actor):
    def __init__(self, file):
        super().__init__()

    def process(self, file):
        model_type = image_classification.MOBILENET
        with ImageInference(image_classification.model(model_type)) as inference:
            image = file
            classes = image_classification.get_classes(
                inference.run(image), max_num_objects=5, object_prob_threshold=0.1)
            for i, (label, score) in enumerate(classes):
                print ('Result %d: %s (prob=%f)' % (i, label, score))

    def run_inference(self, file):
        print('submit inference')
        self.submit(file)


class ImageClassification2(Actor):
    def __init__(self, image):
        super().__init__()

    def process(self, image):
        print('in imageclassification2')
        model_type = image_classification.MOBILENET
        with ImageInference(image_classification.model(model_type)) as inference:

            classes = image_classification.get_classes(
                inference.run(image), max_num_objects=5, object_prob_threshold=0.1)
            for i, (label, score) in enumerate(classes):
                print ('Result %d: %s (prob=%f)' % (i, label, score))

    def run_inference2(self, image):
        print('submit inference')
        self.submit(image)


class Photographer(Actor):
    def __init__(self, format, folder):
        super().__init__()
        assert format in ('jpeg', 'bmp', 'png')

        self._format = format
        self._folder = folder

    def _make_filename(self, timestamp, annotated):
        path='%s/%s_annotated.%s' if annotated else '%s/%s.%s'
        return os.path.expanduser(path % (self._folder, timestamp, self._format))

    def process(self, camera):
        timestamp = time.strftime('%y-%m-%d_%H.%M.%S')

        stream = io.BytesIO()
        print('befere capture')
        #with stopwatch('Taking Photo'):
        camera.capture(stream, format=self._format)
        print('camera capture')

        filename = self._make_filename(timestamp, annotated=False)
        #with stopwatch('Saving original %s' % filename):
        stream.seek(0)
        with open(filename, 'wb') as file:
            file.write(stream.read())
            print('this is file', file)
            #calling image classification here
            image = Image.open(stream)
            print('this is image', image)
            #image.save(filename)
            #print('this is image', image)
            imageclassification = ImageClassification2(image)
            imageclassification.run_inference2(image)
            #imageclassification = ImageClassification(file)
            #imageclassification.run_inference(file)

            print('saved file')

    def shoot(self, camera):
        print('submit camera', camera)
        self.submit(camera)


class MotionSensor(SmoothedInputDevice):
    def __init__(self, pin=None, queue_len=1, sample_rate=1, threshold=0.5, partial=False, pin_factory=None):
        super(MotionSensor, self).__init__(pin, pull_up=True, threshold=threshold, queue_len=queue_len, sample_wait=1 / sample_rate, partial=partial, pin_factory=pin_factory)
        print('motion initializing')
        try:
            self._queue.start()
            print ('motion queue')
        except:
            self.close()
            raise

MotionSensor.motion_detected = MotionSensor.is_active
MotionSensor.when_motion = MotionSensor.when_activated
MotionSensor.when_no_motion = MotionSensor.when_deactivated
MotionSensor.wait_for_motion = MotionSensor.wait_for_active


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--image_format', type=str, dest='image_format', default='jpeg',
            choices=('jpeg', 'bmp', 'png'), help='Format of captured images.')
    parse.add_argument('--image_folder', type=str, dest='image_folder', default='~/Pictures',
            help='Folder to save captured images')
    args = parse.parse_args()
    print('good')

    logger.info('starting')
    leds = Leds()
    photographer = Photographer(args.image_format, args.image_folder)
    button = Button(23)
    pir = MotionSensor(PIN_A)

    try:
        with PiCamera(sensor_mode=4, resolution=(1641, 1232)) as camera, PrivacyLed(leds):
            while True:
                def take_photo():
                    logger.info('button pressed')
                    print('this is camera', camera)
                    photographer.shoot(camera)
                #button = Button(23)

                pir.when_motion = take_photo
                #pir.when_motion = take_photo
                button.when_pressed = take_photo
    finally:
        photographer.stop()
        photographer.join()



if __name__ == '__main__':
    main()
