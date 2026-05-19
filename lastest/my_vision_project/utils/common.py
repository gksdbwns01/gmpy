import sys, os, json, shutil, math, time, platform, multiprocessing, signal, re, gc, traceback, sqlite3, csv, html, subprocess
import queue as qlib
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timedelta
import zipfile, cv2, numpy as np, pandas as pd, psutil, torch, yaml
import threading

from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import QSize, Qt, QThread, pyqtSignal, QTimer, QRectF, QSettings, QDate
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QLineEdit, QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QTextEdit,
    QProgressBar, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QFormLayout, 
    QScrollArea, QGridLayout, QSplitter, QDialog, QInputDialog, QGraphicsView, QGraphicsScene, 
    QGraphicsRectItem, QGraphicsPixmapItem, QListWidget, QListWidgetItem, QGraphicsLineItem,
    QDateEdit)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QColor, QPainter, QPen

if hasattr(Qt, 'AA_EnableHighDpiScaling'): QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
if hasattr(Qt, 'AA_UseHighDpiPixmaps'): QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

APP_ORG_NAME = "MyVisionProject"
APP_NAME = "YoloTrainerApp"
APP_WINDOW_TITLE = "YOLO Training Pipeline (PyQt5) - AutoML + Graceful Stop + Embed Webhooks + VRAM Isolation"
SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
MODEL_MAX_SIZE_BYTES = 5 * 1024**3
DEFAULT_NOTI_FLAGS = {
    "error": True,
    "start": True,
    "early_stop": True,
    "fold": True,
    "tune": True,
    "epoch": True,
    "epoch_interval": 100,
    "task": True,
}


def get_app_base_dir():
    return Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[2]


def build_default_noti_flags():
    return dict(DEFAULT_NOTI_FLAGS)


def is_supported_image(path):
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def iter_supported_images(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    return [path for path in sorted(directory.iterdir()) if is_supported_image(path)]


def find_image_for_label(image_dir, label_file):
    image_dir = Path(image_dir)
    label_file = Path(label_file)
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = image_dir / label_file.with_suffix(suffix).name
        if candidate.exists():
            return candidate
    return None

