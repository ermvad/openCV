import cv2
import numpy as np
import traceback
from logger import logger, log_filter_error

# Обертка для логирования ошибок фильтра
def safe_filter(func):
    """Декоратор для безопасного применения фильтров с логированием ошибок"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            stack_trace = traceback.format_exc()
            log_filter_error(func.__name__, e, stack_trace)
            # Возвращаем исходное изображение в случае ошибки
            return args[0]
    return wrapper


# Фильтры размытия и сглаживания
def apply_gaussian_blur(img, ksize, sigma):
    """Применяет размытие по Гауссу."""
    kx = ksize[0] if ksize[0] % 2 == 1 else ksize[0] + 1
    ky = ksize[1] if ksize[1] % 2 == 1 else ksize[1] + 1
    return cv2.GaussianBlur(img, (kx, ky), sigma)


def apply_median_blur(img, ksize):
    """Применяет медианный фильтр."""
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur(img, k)


def apply_bilateral_filter(img, d, sigma_color, sigma_space):
    """Применяет двустороннее размытие."""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def apply_blur(img, ksize):
    """Применяет размытие среднего."""
    return cv2.blur(img, ksize)


def apply_box_filter(img, ksize):
    """Применяет размытие по блоку."""
    return cv2.boxFilter(img, -1, ksize)


def apply_sobel(img, dx, dy, ksize):
    """Применяет фильтр Собеля."""
    sobel = cv2.Sobel(img, cv2.CV_64F, dx, dy, ksize=ksize)
    return cv2.convertScaleAbs(sobel)


def apply_laplacian(img, ksize):
    """Применяет фильтр Лапласа."""
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(lap)


def apply_stack_blur(img, ksize):
    """Применяет размытие по стэку."""
    for _ in range(3):
        img = cv2.blur(img, ksize)
    return img


def apply_lowpass_filter(img, cutoff):
    """Применяет фильтр нижних частот."""
    sigma = cutoff / 10.0
    return cv2.GaussianBlur(img, (5, 5), sigma)


# Фильтры обнаружения краев
def apply_canny_edge(img, threshold1, threshold2):
    """Применяет детектор краев Canny."""
    return cv2.Canny(img, threshold1, threshold2)


def apply_sobel_x(img, ksize):
    """Применяет фильтр Собеля по оси X."""
    sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    return cv2.convertScaleAbs(sobel)


def apply_sobel_y(img, ksize):
    """Применяет фильтр Собеля по оси Y."""
    sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    return cv2.convertScaleAbs(sobel)


def apply_sobel_xy(img, ksize):
    """Применяет комбинированный фильтр Собеля по осям X и Y."""
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel = cv2.magnitude(sobelx, sobely)
    return cv2.convertScaleAbs(sobel)


def apply_scharr_x(img):
    """Применяет фильтр Шарра по оси X."""
    scharr = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    return cv2.convertScaleAbs(scharr)


def apply_scharr_y(img):
    """Применяет фильтр Шарра по оси Y."""
    scharr = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    return cv2.convertScaleAbs(scharr)


def apply_scharr_xy(img):
    """Применяет комбинированный фильтр Шарра по осям X и Y."""
    scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharr = cv2.magnitude(scharrx, scharry)
    return cv2.convertScaleAbs(scharr)


def apply_harris_corner(img, block_size, ksize, k):
    """Применяет детектор углов Харриса."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    return img


def apply_shi_tomasi(img, max_corners, quality_level, min_distance):
    """Применяет детектор углов Ши-Томаси."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=quality_level,
                                      minDistance=min_distance)
    if corners is not None:
        for corner in np.int0(corners):
            x, y = corner.ravel()
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
    return img


def apply_find_contours(img, mode, method, threshold1, threshold2):
    """Применяет поиск и отображение контуров."""
    edges = cv2.Canny(img, threshold1, threshold2)
    contours, hierarchy = cv2.findContours(edges, mode, method)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    return img


# Цветовые преобразования
def apply_grayscale(img):
    """Преобразует изображение в оттенки серого."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_sepia(img, intensity):
    """Применяет эффект сепии."""
    kernel = np.array([[0.272, 0.534, 0.131],
                        [0.349, 0.686, 0.168],
                        [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, kernel)
    sepia = np.clip(sepia, 0, 255)
    return sepia.astype(np.uint8)


def apply_hsv(img):
    """Преобразует изображение в цветовое пространство HSV."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def apply_lab(img):
    """Преобразует изображение в цветовое пространство LAB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


def apply_ycrcb(img):
    """Преобразует изображение в цветовое пространство YCrCb."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)


def apply_negative(img):
    """Инвертирует цвета изображения."""
    return cv2.bitwise_not(img)


def apply_saturation(img, scale):
    """Изменяет насыщенность изображения."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= scale
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_hue_shift(img, shift):
    """Изменяет оттенок изображения."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint8)
    hsv[..., 0] = (hsv[..., 0].astype(int) + shift) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_clahe(img, clip_limit, tile_grid_size):
    """Применяет адаптивное выравнивание гистограммы CLAHE."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_colormap_jet(img):
    """Применяет цветовую карту 'jet'."""
    return cv2.applyColorMap(img, cv2.COLORMAP_JET)


def apply_colormap_hot(img):
    """Применяет цветовую карту 'hot'."""
    return cv2.applyColorMap(img, cv2.COLORMAP_HOT)


def apply_colormap_cool(img):
    """Применяет цветовую карту 'cool'."""
    return cv2.applyColorMap(img, cv2.COLORMAP_COOL)


def apply_rgb_balance(img, r_scale, g_scale, b_scale):
    """Изменяет баланс RGB."""
    b, g, r = cv2.split(img)
    r = cv2.multiply(r, r_scale)
    g = cv2.multiply(g, g_scale)
    b = cv2.multiply(b, b_scale)
    return cv2.merge((b, g, r))


# Морфологические операции
def apply_erosion(img, ksize, iterations):
    """Применяет эрозию."""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(img, kernel, iterations=iterations)


def apply_dilation(img, ksize, iterations):
    """Применяет дилатацию."""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iterations=iterations)


def apply_opening(img, ksize, iterations):
    """Применяет морфологическое открытие."""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)


def apply_closing(img, ksize, iterations):
    """Применяет морфологическое закрытие."""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def apply_morph_gradient(img, ksize):
    """Применяет морфологический градиент."""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)


def apply_tophat(img, ksize):
    """Применяет морфологию 'top hat'."""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)


def apply_blackhat(img, ksize):
    """Применяет морфологию 'black hat'."""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)


def apply_skeleton(img, iterations):
    """Применяет скелетизацию изображения."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    skel = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(binary, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary, temp)
        skel = cv2.bitwise_or(skel, temp)
        binary = eroded.copy()
        if cv2.countNonZero(binary) == 0:
            break
    return cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)


def apply_thinning(img, iterations):
    """Применяет утончение изображения."""
    # Для примера использует скелетизацию
    return apply_skeleton(img, iterations)


# Геометрические преобразования
def apply_resize(img, scale):
    """Изменяет размер изображения."""
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    return cv2.resize(img, (width, height))


def apply_rotate(img, angle):
    """Поворачивает изображение."""
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def apply_flip(img, mode):
    """Отражает изображение."""
    if mode == "Вертикально":
        return cv2.flip(img, 0)
    elif mode == "Горизонтально":
        return cv2.flip(img, 1)
    elif mode == "Оба":
        return cv2.flip(img, -1)
    return img


def apply_translate(img, x_shift, y_shift):
    """Выполняет сдвиг изображения."""
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def apply_perspective_transform(img):
    """Выполняет перспективное преобразование."""
    # Заглушка – возвращаем исходное изображение
    return img


def apply_affine_transform(img):
    """Выполняет аффинное преобразование."""
    # Заглушка – возвращаем исходное изображение
    return img


def apply_perspective_correction(img):
    """Выполняет коррекцию перспективы."""
    # Заглушка – возвращаем исходное изображение
    return img


# Улучшение изображения
def apply_gamma_correction(img, gamma):
    """Выполняет гамма-коррекцию."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def apply_brightness_contrast(img, brightness, contrast):
    """Регулирует яркость и контраст."""
    return cv2.convertScaleAbs(img, alpha=contrast / 50.0 + 1, beta=brightness)


def apply_auto_white_balance(img):
    """Применяет автоматический баланс белого."""
    # Заглушка – возвращаем исходное изображение
    return img


def apply_auto_contrast(img):
    """Применяет автоконтраст."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)


def apply_denoise(img):
    """Удаляет шум из изображения."""
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def apply_sharpen(img):
    """Повышает резкость изображения."""
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def apply_detail(img):
    """Улучшает детализацию изображения."""
    return cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)


def apply_hdr(img):
    """Применяет HDR эффект."""
    return cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)


def apply_histogram_equalization(img):
    """Выполняет выравнивание гистограммы."""
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


# Сегментация и пороговая обработка
def apply_threshold(img, thresh):
    """Применяет бинаризацию по порогу."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def apply_adaptive_threshold(img, block_size, C):
    """Применяет адаптивный порог."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, block_size, C)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def apply_otsu_threshold(img):
    """Применяет порог Отсу."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def apply_watershed(img):
    """Применяет алгоритм водораздела."""
    # Заглушка – возвращаем исходное изображение
    return img


def apply_mean_shift(img):
    """Применяет алгоритм mean shift."""
    return cv2.pyrMeanShiftFiltering(img, 21, 51)


def apply_kmeans(img, K):
    """Применяет алгоритм k-means."""
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((img.shape))


def apply_grabcut(img):
    """Применяет алгоритм GrabCut."""
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, img.shape[1] - 20, img.shape[0] - 20)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    return img


def apply_motion_detector(img):
    """Применяет детектор движения."""
    # Заглушка – возвращаем исходное изображение
    return img


# Выделение признаков
def apply_sift(img):
    """Выделяет SIFT дескрипторы."""
    orb = cv2.ORB_create()  # Используем ORB вместо SIFT, если SIFT не доступен
    kp = orb.detect(img, None)
    return cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)


def apply_surf(img):
    """Выделяет SURF дескрипторы."""
    # Используем ORB, если SURF не доступен
    return apply_sift(img)


def apply_orb(img):
    """Выделяет ORB дескрипторы."""
    orb = cv2.ORB_create()
    kp = orb.detect(img, None)
    return cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)


def apply_hog(img):
    """Выделяет HOG дескрипторы."""
    # Заглушка – возвращаем исходное изображение
    return img


def apply_face_detection(img):
    """Выполняет детектирование лиц."""
    # Заглушка – возвращаем исходное изображение
    return img


def apply_eye_detection(img):
    """Выполняет детектирование глаз."""
    # Заглушка – возвращаем исходное изображение
    return img


def apply_hough_lines(img):
    """Выделяет линии методом Хафа."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img


def apply_hough_circles(img):
    """Выделяет окружности методом Хафа."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    return img


# Пользовательские фильтры
def apply_depth_map(img):
    """Создает карту глубины изображения."""
    # Заглушка – возвращаем исходное изображение
    return img


def apply_channel_multiplex(img):
    """Мультиплексирует каналы изображения."""
    b, g, r = cv2.split(img)
    return cv2.merge((r, g, b))


def apply_cartoon(img):
    """Применяет эффект мультфильма."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


def apply_pixelation(img, block_size):
    """Применяет эффект пикселизации."""
    height, width = img.shape[:2]
    temp = cv2.resize(img, (width // block_size, height // block_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)


def apply_vignette(img):
    """Применяет эффект виньетки."""
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, 200)
    kernel_y = cv2.getGaussianKernel(rows, 200)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    vignette = np.copy(img)
    for i in range(3):
        vignette[:, :, i] = vignette[:, :, i] * mask
    return vignette


def apply_motion_blur(img, kernel_size):
    """Применяет эффект размытия в движении."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(img, -1, kernel)


def apply_watercolor(img):
    """Применяет акварельный эффект."""
    return cv2.stylization(img, sigma_s=60, sigma_r=0.6)


def apply_oil_painting(img):
    """Применяет эффект масляной живописи."""
    # Заглушка – возвращаем исходное изображение
    return img


def apply_pencil_sketch(img):
    """Применяет эффект карандашного рисунка."""
    gray, sketch = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return sketch