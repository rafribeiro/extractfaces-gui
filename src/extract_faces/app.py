"""A small Qt window for previewing videos and extracting face crops."""

import math
import sys
import traceback
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QShortcut, QTextCursor
from PySide6.QtMultimedia import QMediaMetaData, QMediaPlayer, QVideoSink
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPlainTextEdit, QPushButton, QSlider, QSpinBox, QStyle, QStyleOptionSlider, QVBoxLayout, QWidget,
)


class ConsoleStream(QObject):
    """Send Python output to the GUI safely, including from the worker thread."""

    text_written = Signal(str)
    encoding = "utf-8"

    def __init__(self, original):
        super().__init__()
        self.original = original

    def write(self, text):
        if self.original is not None:
            self.original.write(text)
        self.text_written.emit(text)
        return len(text)

    def flush(self):
        if self.original is not None:
            self.original.flush()

    def isatty(self):
        return False


class VideoView(QWidget):
    """Paint the video and store the selection in original image pixels."""

    selection_changed = Signal()
    selection_started = Signal()

    def __init__(self):
        super().__init__()
        self.image = QImage()
        self.region = None
        self.anchor = None
        self.setMinimumSize(480, 270)

    def set_frame(self, frame):
        image = frame.toImage()
        if not image.isNull():
            self.image = image
            self.update()

    def image_rect(self):
        if self.image.isNull():
            return QRectF()
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def image_point(self, point):
        rect = self.image_rect()
        return QPointF(
            max(0, min(self.image.width(), (point.x() - rect.x()) * self.image.width() / rect.width())),
            max(0, min(self.image.height(), (point.y() - rect.y()) * self.image.height() / rect.height())),
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#202020"))
        if self.image.isNull():
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a video to begin")
            return
        rect = self.image_rect()
        painter.drawImage(rect, self.image)
        if self.region is not None:
            scale = rect.width() / self.image.width()
            selection = QRectF(rect.x() + self.region.x() * scale,
                               rect.y() + self.region.y() * scale,
                               self.region.width() * scale, self.region.height() * scale)
            painter.setPen(QPen(QColor("#00e5ff"), 2))
            painter.drawRect(selection)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.image_rect().contains(event.position()):
            self.selection_started.emit()
            self.anchor = self.image_point(event.position())
            self.region = QRectF(self.anchor, self.anchor)

    def mouseMoveEvent(self, event):
        if self.anchor is not None:
            self.region = QRectF(self.anchor, self.image_point(event.position())).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.anchor is not None:
            self.mouseMoveEvent(event)
            self.anchor = None
            if self.region.width() < 2 or self.region.height() < 2:
                self.region = None
            self.selection_changed.emit()
            self.update()

    def clear_region(self):
        self.region = None
        self.selection_changed.emit()
        self.update()

    def region_pixels(self):
        if self.region is None:
            return None
        return (math.floor(self.region.left()), math.floor(self.region.top()),
                math.ceil(self.region.right()), math.ceil(self.region.bottom()))


class SeekSlider(QSlider):
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            option = QStyleOptionSlider()
            self.initStyleOption(option)
            groove = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderGroove, self)
            handle = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderHandle, self)
            value = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(),
                round(event.position().x() - groove.x() - handle.width() / 2),
                max(1, groove.width() - handle.width()), option.upsideDown)
            self.setValue(value)
            self.setSliderDown(True)
            self.sliderMoved.emit(value)
            event.accept()
        else:
            super().mousePressEvent(event)


class ExtractionCancelled(Exception):
    pass


class ExtractionWorker(QThread):
    """Run forensicface away from the GUI thread; report sampled frames."""

    progress = Signal(int)
    message = Signal(str)

    def __init__(self, video, destination, region, every_n_frames, margin, start_from,
                 use_gpu, det_size=320, det_thresh=0.5, stop_at=None):
        super().__init__()
        self.video = video
        self.destination = destination
        self.region = region
        self.every_n_frames = every_n_frames
        self.margin = margin
        self.start_from = start_from
        self.stop_at = stop_at
        self.use_gpu = use_gpu
        self.det_size = det_size
        self.det_thresh = det_thresh

    def run(self):
        try:
            # Import and load the detector here so the window remains responsive.
            from forensicface.app import ForensicFace

            detector = ForensicFace(detection="scrfd", embedding=None, pose=None,
                                    gender=None, age=None, quality=None, use_gpu=self.use_gpu,
                                    det_size=self.det_size, det_thresh=self.det_thresh)
            original_process_image = detector.process_image
            processed = 0

            def process_region(frame, **kwargs):
                nonlocal processed
                if self.isInterruptionRequested():
                    raise ExtractionCancelled()
                if self.region is None:
                    results = original_process_image(frame, **kwargs)
                else:
                    x1, y1, x2, y2 = self.region
                    if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
                        raise ValueError("Video preview and extraction dimensions differ. Clear the region and retry.")
                    results = original_process_image(frame[y1:y2, x1:x2], **kwargs)
                    # extract_faces crops the full frame, so restore full-frame coordinates.
                    for result in results:
                        result["bbox"] = result["bbox"] + [x1, y1, x1, y1]
                        result["keypoints"] = result["keypoints"] + [x1, y1]
                processed += 1
                self.progress.emit(processed)
                return results

            # extract_faces has no ROI argument; adapt this instance's image processing.
            detector.process_image = process_region
            count = self.extract_video(detector)
            self.message.emit(f"Finished: {count} face crops saved to {self.destination}")
        except ExtractionCancelled:
            self.message.emit("Extraction cancelled. Crops already saved remain in the output folder.")
        except Exception as error:
            traceback.print_exc()
            self.message.emit(f"Extraction failed: {error}")


    def extract_video(self, detector):
        from forensicface.geometry import extend_bbox
        from forensicface.image_io import write_image

        capture = cv2.VideoCapture(self.video)
        count = 0
        try:
            if self.isInterruptionRequested():
                raise ExtractionCancelled()
            if not capture.isOpened():
                raise ValueError("The video could not be read.")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("The video has no valid frame rate.")
            frame_index = int(fps * self.start_from)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            Path(self.destination).mkdir(parents=True, exist_ok=True)
            while self.stop_at is None or frame_index / fps < self.stop_at:
                if self.isInterruptionRequested():
                    raise ExtractionCancelled()
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % self.every_n_frames:
                    continue
                for face_index, result in enumerate(detector.process_image(frame, single_face=False)):
                    box = extend_bbox(result["bbox"], frame.shape, margin_factor=self.margin)
                    path = Path(self.destination) / f"frame_{frame_index:07}_face_{face_index:02}.png"
                    write_image(str(path), frame[box[1]:box[3], box[0]:box[2]])
                    count += 1
        finally:
            capture.release()
        return count


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Extract Faces")
        self.setAcceptDrops(True)
        self.resize(1000, 800)
        self.video_path = None
        self.worker = None
        self.duration_seconds = 0
        self.fps = 0

        self.player = QMediaPlayer(self)
        self.seek_shortcuts = []
        for key, offset in ((Qt.Key.Key_Left, -10000), (Qt.Key.Key_Right, 10000)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda offset=offset: self.seek_relative(offset))
            self.seek_shortcuts.append(shortcut)
        self.sink = QVideoSink(self)
        self.player.setVideoSink(self.sink)
        self.view = VideoView()
        self.sink.videoFrameChanged.connect(self.view.set_frame)
        self.view.selection_started.connect(self.player.pause)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.open_button = QPushButton("Open video…")
        self.open_button.clicked.connect(self.open_video)
        layout.addWidget(self.open_button)
        layout.addWidget(self.view, 1)

        playback = QHBoxLayout()
        self.play_button = QPushButton("Play / Pause")
        self.play_button.clicked.connect(self.toggle_playback)
        playback.addWidget(self.play_button)
        self.seek = SeekSlider(Qt.Orientation.Horizontal)
        self.seek.setSingleStep(10000)
        self.seek.setToolTip("Click or drag to seek; Left / Right move by 10 seconds.")
        self.seek.sliderMoved.connect(self.player.setPosition)
        self.seek.actionTriggered.connect(lambda action: self.player.setPosition(self.seek.sliderPosition()))
        self.player.positionChanged.connect(self.update_position)
        self.player.durationChanged.connect(lambda duration: self.seek.setRange(0, duration))
        playback.addWidget(self.seek, 1)
        self.time_label = QLabel("0.00 s")
        playback.addWidget(self.time_label)
        layout.addLayout(playback)

        self.metadata = QLabel("No video selected")
        self.metadata.setWordWrap(True)
        layout.addWidget(self.metadata)
        self.player.metaDataChanged.connect(self.update_codecs)
        self.player.errorOccurred.connect(lambda error, text: self.status.setText(f"Playback error: {text}"))

        self.settings = QWidget()
        form = QFormLayout(self.settings)
        region_row = QHBoxLayout()
        self.region_label = QLabel("Full frame — drag on the video to select a region")
        region_row.addWidget(self.region_label, 1)
        clear_button = QPushButton("Clear region")
        clear_button.clicked.connect(self.view.clear_region)
        region_row.addWidget(clear_button)
        form.addRow("Search area", region_row)
        self.view.selection_changed.connect(self.update_region)

        folder_row = QHBoxLayout()
        self.destination = QLineEdit()
        self.destination.setAcceptDrops(False)
        folder_row.addWidget(self.destination)
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self.choose_folder)
        folder_row.addWidget(browse_button)
        form.addRow("Output folder", folder_row)

        self.margin = QDoubleSpinBox()
        self.margin.setRange(1, 10)
        self.margin.setSingleStep(0.1)
        self.margin.setValue(2)
        self.margin.setToolTip("1 keeps the detected box; 2 doubles its width and height.")
        form.addRow("Crop size multiplier", self.margin)
        self.skip = QSpinBox()
        self.skip.setRange(1, 100000)
        self.skip.setToolTip("1 processes every frame; 5 looks for faces once every 5 frames.")
        form.addRow("Process every Nth frame", self.skip)
        self.start = QDoubleSpinBox()
        self.start.setRange(0, 864000)
        self.start.setSuffix(" s")
        start_row = QHBoxLayout()
        start_row.addWidget(self.start)
        current_button = QPushButton("Use current position")
        current_button.clicked.connect(lambda: self.start.setValue(self.player.position() / 1000))
        start_row.addWidget(current_button)
        form.addRow("Start time", start_row)
        self.stop_enabled = QCheckBox("Stop at")
        self.stop = QDoubleSpinBox()
        self.stop.setRange(0, 864000)
        self.stop.setSuffix(" s")
        self.stop.setEnabled(False)
        self.stop_enabled.toggled.connect(self.stop.setEnabled)
        stop_row = QHBoxLayout()
        stop_row.addWidget(self.stop)
        self.stop_current = QPushButton("Use current position")
        self.stop_current.setEnabled(False)
        self.stop_enabled.toggled.connect(self.stop_current.setEnabled)
        self.stop_current.clicked.connect(lambda: self.stop.setValue(self.player.position() / 1000))
        stop_row.addWidget(self.stop_current)
        form.addRow(self.stop_enabled, stop_row)
        self.det_size = QSpinBox()
        self.det_size.setRange(32, 4096)
        self.det_size.setSingleStep(32)
        self.det_size.setValue(320)
        self.det_size.setToolTip("Square detector input size in pixels. Larger sizes can detect smaller faces but take longer.")
        form.addRow("Detector size (det_size)", self.det_size)
        self.det_thresh = QDoubleSpinBox()
        self.det_thresh.setRange(0, 1)
        self.det_thresh.setSingleStep(0.05)
        self.det_thresh.setValue(0.5)
        self.det_thresh.setToolTip("Minimum detection confidence. Lower values accept more detections, including possible false positives.")
        form.addRow("Detection threshold (det_thresh)", self.det_thresh)
        self.gpu = QCheckBox("Use GPU (requires a compatible runtime)")
        form.addRow(self.gpu)
        layout.addWidget(self.settings)

        actions = QHBoxLayout()
        self.extract_button = QPushButton("Extract faces")
        self.extract_button.setEnabled(False)
        self.extract_button.clicked.connect(self.extract)
        actions.addWidget(self.extract_button)
        self.cancel_button = QPushButton("Cancel extraction")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_extraction)
        actions.addWidget(self.cancel_button)
        layout.addLayout(actions)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.status = QLabel("Ready. Playback is silent; extraction uses the original video.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)

        self.console_toggle = QPushButton("Show console output")
        self.console_toggle.setCheckable(True)
        layout.addWidget(self.console_toggle)
        self.console = QPlainTextEdit()
        self.console.setAcceptDrops(False)
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(2000)
        self.console.setMinimumHeight(120)
        self.console.setMaximumHeight(240)
        self.console.hide()
        self.console_toggle.toggled.connect(self.toggle_console)
        layout.addWidget(self.console)

    def toggle_console(self, expanded):
        self.console.setVisible(expanded)
        self.console_toggle.setText("Hide console output" if expanded else "Show console output")

    @Slot(str)
    def append_console(self, text):
        # tqdm uses carriage returns; show each update as a readable log line.
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text.replace("\r", "\n"))
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def dropped_video(self, mime_data):
        if self.worker is not None:
            return None
        for url in mime_data.urls():
            if url.isLocalFile() and Path(url.toLocalFile()).is_file():
                return url.toLocalFile()
        return None

    def dragEnterEvent(self, event):
        if self.dropped_video(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = self.dropped_video(event.mimeData())
        if path:
            event.acceptProposedAction()
            self.load_video(path)
        else:
            event.ignore()

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", "Video files (*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.mpeg *.mts);;All files (*)")
        if not path:
            return
        self.load_video(path)

    def load_video(self, path):
        if self.worker is not None:
            return
        capture = cv2.VideoCapture(path)
        try:
            if not capture.isOpened():
                QMessageBox.warning(self, "Cannot open video", "The video could not be read.")
                return
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            codec = int(capture.get(cv2.CAP_PROP_FOURCC))
            fourcc = "".join(chr((codec >> (8 * i)) & 255) for i in range(4)).strip("\x00")
        finally:
            capture.release()
        self.player.stop()
        self.video_path = path
        self.fps = fps if math.isfinite(fps) and fps > 0 else 0
        self.duration_seconds = frames / self.fps if self.fps and math.isfinite(frames) else 0
        self.base_metadata = f"{Path(path).name} | {width} × {height} | {self.fps:.3f} fps | {self.duration_seconds:.2f} s | Video: {fourcc or 'unknown'}"
        self.metadata.setText(self.base_metadata)
        self.view.image = QImage()
        self.view.clear_region()
        self.destination.setText(str(Path(path).with_name(Path(path).stem + "_faces")))
        self.start.setValue(0)
        self.start.setMaximum(max(0, self.duration_seconds - 0.01))
        self.stop_enabled.setChecked(False)
        self.stop.setMaximum(self.duration_seconds)
        self.stop.setValue(self.duration_seconds)
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.play()
        self.extract_button.setEnabled(True)
        self.status.setText("Pause or drag on the video to select the search area.")

    def update_codecs(self):
        if not self.video_path:
            return
        metadata = self.player.metaData()
        video = metadata.stringValue(QMediaMetaData.Key.VideoCodec)
        audio = metadata.stringValue(QMediaMetaData.Key.AudioCodec)
        self.metadata.setText(self.base_metadata + f" | Media codecs: {video or 'unknown video'}, {audio or 'no audio / unknown'}")

    def update_position(self, position):
        if not self.seek.isSliderDown():
            self.seek.setValue(position)
        self.time_label.setText(f"{position / 1000:.2f} s")

    def toggle_playback(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def update_region(self):
        region = self.view.region_pixels()
        self.region_label.setText(f"Pixels (x1, y1, x2, y2): {region}" if region else "Full frame — drag on the video to select a region")

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            self.seek_relative(-10000 if event.key() == Qt.Key.Key_Left else 10000)
            event.accept()
        else:
            super().keyPressEvent(event)

    def seek_relative(self, offset):
        if self.video_path:
            self.player.setPosition(max(0, min(self.player.duration(), self.player.position() + offset)))

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Output folder", self.destination.text())
        if folder:
            self.destination.setText(folder)

    def extract(self):
        stop_at = self.stop.value() if self.stop_enabled.isChecked() else None
        if stop_at is not None and stop_at <= self.start.value():
            QMessageBox.warning(self, "Invalid stop time", "Stop time must be later than start time.")
            return
        destination = self.destination.text().strip()
        if not destination:
            QMessageBox.warning(self, "Output folder required", "Choose a folder for the PNG crops.")
            return
        try:
            folder = Path(destination).expanduser().resolve()
            folder.mkdir(parents=True, exist_ok=True)
            if any(folder.iterdir()):
                QMessageBox.warning(self, "Choose an empty folder", "Choose an empty output folder to avoid overwriting previous crops.")
                return
        except OSError as error:
            QMessageBox.warning(self, "Cannot use output folder", str(error))
            return
        self.player.pause()
        self.settings.setEnabled(False)
        self.view.setEnabled(False)
        self.open_button.setEnabled(False)
        self.extract_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        every_n = self.skip.value()
        end = stop_at if stop_at is not None else self.duration_seconds
        estimated_frames = max(1, math.ceil((end - self.start.value()) * self.fps / every_n))
        self.progress.setRange(0, estimated_frames if self.fps else 0)
        self.progress.setValue(0)
        self.status.setText("Loading face detector…")
        self.worker = ExtractionWorker(self.video_path, str(folder), self.view.region_pixels(),
                                       every_n, self.margin.value(), self.start.value(), self.gpu.isChecked(),
                                       det_size=self.det_size.value(), det_thresh=self.det_thresh.value(), stop_at=stop_at)
        self.worker.progress.connect(self.update_progress)
        self.worker.message.connect(self.status.setText)
        self.worker.finished.connect(self.extraction_finished)
        self.worker.start()

    def update_progress(self, processed):
        if self.progress.maximum() > 0:
            self.progress.setMaximum(max(self.progress.maximum(), processed))
            self.progress.setValue(processed)
        self.status.setText(f"Processed {processed} sampled frames…")

    def cancel_extraction(self):
        if self.worker:
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.status.setText("Cancelling after the current detector operation…")

    def extraction_finished(self):
        self.settings.setEnabled(True)
        self.view.setEnabled(True)
        self.open_button.setEnabled(True)
        self.extract_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker.deleteLater()
        self.worker = None

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.cancel_extraction()
            event.ignore()
        else:
            self.player.stop()
            event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    original_stdout, original_stderr = sys.stdout, sys.stderr
    stdout = ConsoleStream(original_stdout)
    stderr = ConsoleStream(original_stderr)
    stdout.text_written.connect(window.append_console)
    stderr.text_written.connect(window.append_console)
    sys.stdout, sys.stderr = stdout, stderr
    try:
        window.show()
        exit_code = app.exec()
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
    sys.exit(exit_code)
