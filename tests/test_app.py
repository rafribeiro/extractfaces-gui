"""Check coordinate mapping and extraction without downloading face models."""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import tempfile
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtCore import QMimeData, QPointF, QRectF, Qt, QUrl
from PySide6.QtGui import QDropEvent, QImage
from PySide6.QtWidgets import QApplication
from forensicface.app import ForensicFace
from extract_faces.app import ConsoleStream, ExtractionWorker, MainWindow, VideoView, main


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_selection_with_letterboxing(self):
        view = VideoView()
        view.resize(800, 600)
        view.image = QImage(1920, 1080, QImage.Format.Format_RGB888)
        self.assertEqual(view.image_point(QPointF(400, 300)), QPointF(960, 540))
        self.assertEqual(view.image_point(QPointF(0, 0)), QPointF(0, 0))
        self.assertEqual(view.image_point(QPointF(800, 600)), QPointF(1920, 1080))
        view.region = QRectF(10.2, 20.3, 30.4, 40.5)
        self.assertEqual(view.region_pixels(), (10, 20, 41, 61))

    def test_window_constructs(self):
        window = MainWindow()
        self.assertFalse(window.extract_button.isEnabled())
        window.close()

    def test_gui_launch_without_console_streams(self):
        from tqdm import tqdm

        def run_event_loop():
            self.assertIsNotNone(sys.stdout)
            self.assertIsNotNone(sys.stderr)
            with tqdm(total=1, disable=False) as progress:
                progress.update(1)
            print("Library output also works without a console")
            return 0

        with patch.object(sys, "stdout", None), patch.object(sys, "stderr", None), \
             patch("extract_faces.app.QApplication") as application, \
             patch("extract_faces.app.MainWindow"), patch.object(sys, "exit"):
            application.return_value.exec.side_effect = run_event_loop
            main()
            self.assertIsNone(sys.stdout)
            self.assertIsNone(sys.stderr)

    def test_console_receives_worker_thread_output_while_collapsed(self):
        window = MainWindow()
        stream = ConsoleStream(None)
        stream.text_written.connect(window.append_console)
        thread = threading.Thread(target=lambda: stream.write("Detector ready\n\rFrame 1\n"))
        thread.start()
        thread.join()
        self.app.processEvents()
        self.assertIn("Detector ready", window.console.toPlainText())
        self.assertIn("Frame 1", window.console.toPlainText())
        self.assertTrue(window.console.isHidden())
        window.console_toggle.setChecked(True)
        self.assertFalse(window.console.isHidden())
        window.close()

    def test_drop_opens_local_file_and_is_rejected_during_extraction(self):
        window = MainWindow()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "video with spaces.mp4"
            path.touch()
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(str(path))])
            event = QDropEvent(QPointF(20, 20), Qt.DropAction.CopyAction, mime,
                               Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
            with patch.object(window, "load_video") as load:
                window.dropEvent(event)
                load.assert_called_once()
                self.assertEqual(Path(load.call_args.args[0]), path)
                self.assertTrue(event.isAccepted())
                window.worker = object()
                self.assertIsNone(window.dropped_video(mime))
                window.worker = None
            mime.setUrls([QUrl("https://example.com/video.mp4")])
            self.assertIsNone(window.dropped_video(mime))
        window.close()

    def test_roi_crops_original_video_and_honors_frame_skip(self):
        # Keep the real forensicface extraction method; replace only model inference.
        class FakeDetector:
            extract_faces = ForensicFace.extract_faces

            def process_image(self, frame, **kwargs):
                self_shape.append(frame.shape[:2])
                return [{"bbox": np.array([2, 3, 10, 11]),
                         "keypoints": np.zeros((5, 2))}]

        self_shape = []
        with tempfile.TemporaryDirectory() as temporary:
            video = str(Path(temporary) / "video.avi")
            writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            for _ in range(6):
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                frame[10:30, 20:40] = 180
                writer.write(frame)
            writer.release()
            destination = str(Path(temporary) / "faces")
            worker = ExtractionWorker(video, destination, (20, 10, 40, 30), 2, 1, 0.2, False,
                                      det_size=640, det_thresh=0.35)
            messages = []
            worker.message.connect(messages.append)
            with patch("forensicface.app.ForensicFace", return_value=FakeDetector()) as detector:
                worker.run()
            self.assertEqual(detector.call_args.kwargs["det_size"], 640)
            self.assertEqual(detector.call_args.kwargs["det_thresh"], 0.35)
            crops = list(Path(destination).glob("*.png"))
            self.assertEqual(len(crops), 2, messages)
            self.assertEqual(self_shape, [(20, 20), (20, 20)])
            self.assertEqual(cv2.imread(str(crops[0])).shape[:2], (8, 8))
            self.assertGreater(cv2.imread(str(crops[0])).mean(), 150)
            self.assertIn("Finished: 2", messages[-1])

    def test_cancel_before_first_frame(self):
        class FakeDetector:
            def extract_faces(self, *args, **kwargs):
                self.process_image(np.zeros((48, 64, 3)), single_face=False)

            def process_image(self, *args, **kwargs):
                raise AssertionError("Inference should not run after cancellation")

        worker = ExtractionWorker("unused", "unused", None, 1, 1, 0, False)
        messages = []
        worker.message.connect(messages.append)
        with patch("forensicface.app.ForensicFace", return_value=FakeDetector()), \
             patch.object(worker, "isInterruptionRequested", return_value=True):
            worker.run()
        self.assertIn("cancelled", messages[-1])


if __name__ == "__main__":
    unittest.main()
