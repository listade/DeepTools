"""Utils module"""

import json
import sys
import contextlib
import datetime
import glob
import os
import os.path

import cv2
import numpy as np
import requests
import scipy
import yaml

import PIL
import pyproj


def xy_to_lonlat(x, y, width, heigth, clon, clat, m_px, yaw=.0):
    """Returns (lon, lat) of (x, y)"""

    epsg = pyproj.Proj(init="epsg:4326")
    merc = pyproj.Proj(init="epsg:3857")

    x -= width // 2
    y -= heigth // 2

    x, y = rotate(x, y, np.deg2rad(yaw))

    m_ox, m_oy = pyproj.transform(epsg, merc, clon, clat)

    mx = x * m_px + m_ox
    my = y * m_px + m_oy

    lon, lat =  pyproj.transform(merc, epsg, mx, my)

    return lon, lat


def lonlat_to_xy(lon, lat, clon, clat, m_px):
    """Returns (x, y) of (lon, lat)"""

    epsg = pyproj.Proj(init="epsg:4326")
    merc = pyproj.Proj(init="epsg:3857")

    m_ox, m_oy = pyproj.transform(epsg, merc, clon, clat)
    mx, my = pyproj.transform(epsg, merc, lon, lat)

    return int((mx - m_ox) / m_px), int((my - m_oy) / m_px)


def px_to_meters(w, height, sensor_width, focal_len):
    """Returns m/px"""

    alpha = np.arctan((sensor_width / 2) / focal_len)
    distance = 2 * np.tan(alpha) * height

    return distance / w


def bbox_to_latlon(x, y, w, h, pw, ph, lat, lon, alt, yaw, focal_len, elevation, sensor_width):
    """Returns (lat, lon), area (m) of bbox"""

    res = px_to_meters(pw, alt - elevation, sensor_width, focal_len)

    x = x + w // 2
    y = y + h // 2

    lat, lon = xy_to_lonlat(x, y, pw, ph, lat, lon, 180 - yaw, res)

    return lat, lon, w * h * res ** 2


def rotate(x, y, rad=.0):
    """Returns (x, y) rotated by rad"""

    cos, sin = np.cos(rad), np.sin(rad)
    j = np.matrix([[cos, sin], [-sin, cos]])
    mat = np.dot(j, [x, y])

    return int(mat.T[0]), int(mat.T[1])


def xyxy_to_xywh(x0, y0, x1, y1):
    """Returns (cx, cy) (w, h) by (x0, y0) (x1, y1)"""

    w = abs(x1 - x0)
    h = abs(y1 - y0)
    x = x0 + w // 2
    y = y0 + h // 2

    return x, y, w, h


def merge_two_images(im_dst, im_src, im_out, x=0, y=0, rad=.0):
    """Returns numpy image of img_dst with rotated img_src on (x, y)"""

    np_dst = cv2.imread(im_dst)
    np_src = cv2.imread(im_src)

    np_src = scipy.ndimage.rotate(np_src, rad)

    h, w, _ = np_src.shape

    x0 = max(x - w // 2, 0)
    y0 = max(y - h // 2, 0)

    x1 = x0 + w
    y1 = y0 + h

    np_dst[y0:y1,x0:x1] += np_src

    cv2.imwrite(im_out, np_dst)


def draw_bbox(im_np, x, y, w, h, rad=0, color=(0, 255, 0), line_thickness=5):
    """Draw rotated bbox"""

    x0 = x - w // 2
    y0 = y - h // 2

    x1 = x0 + w; y1 = y0
    x2 = x1; y2 = y0 + h

    x3 = x0; y3 = y2
    x0 -= x; y0 -= y
    x1 -= x; y1 -= y
    x2 -= x; y2 -= y
    x3 -= x; y3 -= y

    rad = np.deg2rad(rad)

    x0, y0 = rotate(x0, y0, rad)
    x1, y1 = rotate(x1, y1, rad)
    x2, y2 = rotate(x2, y2, rad)
    x3, y3 = rotate(x3, y3, rad)

    x0 += x; y0 += y
    x1 += x; y1 += y
    x2 += x; y2 += y
    x3 += x; y3 += y

    p0 = (x0, y0); p1 = (x1, y1)
    p2 = (x2, y2); p3 = (x3, y3)

    cv2.line(im_np, p0, p1, color, thickness=line_thickness)
    cv2.line(im_np, p1, p2, color, thickness=line_thickness)
    cv2.line(im_np, p2, p3, color, thickness=line_thickness)
    cv2.line(im_np, p3, p0, color, thickness=line_thickness)


def countours_and_bboxes(src_path):
    """Returns im_src countours with bboxes generator"""

    im_np = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert im_np

    conts, _ = cv2.findContours(im_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in conts:
        x, y, w, h = map(int, cv2.boundingRect(contour))

        yield contour.squeeze(), (x, y, w, h)

def contour_center(cnt):
    """Returns (x, y) of contour center"""

    M = cv2.moments(cnt)

    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    return cx, cy


def create_mask(src_img, dst_img, polys):
    """Generate image grayscale mask"""

    im_src = cv2.imread(src_img)
    im_mask = np.zeros_like(im_src[:,:,0])

    for p in polys:
        im_mask = cv2.fillPoly(im_mask, np.array([p]), color=(255, 255, 255))

    im_dst = cv2.bitwise_and(im_src, im_src, mask=im_mask)

    cv2.imwrite(dst_img, im_dst)


@contextlib.contextmanager
def cwd(path):
    """Enter dir during executing and return"""

    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


@contextlib.contextmanager
def timing(action, file=sys.stdout):
    """Print execution duration"""

    start = datetime.datetime.now()
    yield
    elapsed = datetime.datetime.now() - start
    file.write(f"{action} [{elapsed}]\n")


def twins(x_dir, y_dir):
    """Returns file pairs generator with same name"""

    for x_file in glob.glob(x_dir + "\\*.*"):
        for y_file in glob.glob(y_dir + "\\*.*"):
            x_base = os.path.basename(x_file).split(".")[0]
            y_base = os.path.basename(y_file).split(".")[0]
            if x_base == y_base:
                yield x_file, y_file


def make_dirs(*dirs, exist_ok=True):
    """Make dir vararg"""

    for folder in dirs:
        os.makedirs(folder, exist_ok=exist_ok)


def remove_files(*files):
    """Remove file vararg """

    for file in files:
        os.remove(file)


def json_dump(data, path):
    """Save dict to json file"""

    with open(path, "w", encoding="utf8") as file:
        json.dump(data, file, indent=2)


def json_load(path):
    """Load dict from json file"""

    with open(path, "r", encoding="utf8") as file:
        return json.load(file)


def yaml_dump(data, path):
    """Save dict to yaml file"""

    with open(path, "w", encoding="utf8") as file:
        yaml.dump(data, file)


def move_files(src_dir, dst_dir, pattern="*.*"):
    """Move files using pattern"""

    for file in glob.glob(os.path.join(src_dir, pattern)):
        os.replace(file, os.path.join(dst_dir, os.path.basename(file)))


def clean_folders(*folders, pattern="*.*"):
    """Remove files using pattern vararg"""

    for folder in folders:
        for file in glob.glob(os.path.join(folder, pattern)):
            os.remove(file)


def download_file(url, path):
    """Download and save file"""

    with open(path, "wb") as f:
        resp = requests.get(url, timeout=(5))
        resp.raise_for_status()

        f.write(resp.content)


def get_focal_len(path):
    """Returns focal length from image exif"""

    with PIL.Image.open(path) as f:
        return f._getexif()[0x920A]


def get_lat_lon_alt(path):
    """Returns image lat, lon and alt from exif"""

    with PIL.Image.open(path) as f:
        exif = f._getexif()
        geo = exif[0x8825]
        return geo[2], geo[4], geo[6]


def folder_size(path):
    """Returns folder size"""

    size_bytes = 0
    for folder, _, files in os.walk(path):
        for file in files:
            filename = os.path.join(folder, file)
            if not os.path.islink(filename):
                size_bytes += os.path.getsize(filename)

    return size_bytes


def get_elevation(lat, lon):
    """Getting elevation from api.opentopodata.org"""

    url = f"https://api.opentopodata.org/v1/mapzen?locations={lat},{lon}"
    resp = requests.get(url)
    resp.raise_for_status()

    data = resp.json()
    elev = data["results"][0]["elevation"]

    return float(elev)


def get_elevation_2(lon, lat):
    """Getting elevation from api.open-elevation.com"""

    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    resp = requests.get(url)
    resp.raise_for_status()

    data = resp.json()
    elev = data["results"][0]["elevation"]

    return float(elev)
