# Ramiro Isa-Jara, ramiro.isaj@gmail.com
# ANSIS -> Functions for Analysis of cells Images using Segmentation methods for automatic control
# Using GEMA version 1.3.3

import cv2
import glob
import os
import time
import gc
import serial
import copy
import numpy as np
import pandas as pd
from statistics import mean
from skimage import morphology
from skimage.filters import rank
from skimage.morphology import disk
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use("TkAgg")


def load_image(orig, type_, id_sys):
    symbol = '\\' if id_sys == 0 else '/'
    filenames = [img for img in glob.glob(orig+type_)]
    filenames.sort()
    total_ = len(filenames)
    name = filenames[total_-1]
    parts = name.split(symbol)
    exp, name_i = parts[len(parts) - 2], parts[len(parts) - 1]
    # read image
    image_ = cv2.imread(name)

    return image_, total_, exp, name_i


def load_image_i(orig, i, type_, filenames, exp, id_sys):
    symbol = '\\' if id_sys == 0 else '/'
    if len(filenames) == 0:
        filenames = [img for img in glob.glob(orig+type_)]
        filenames.sort()
    if i < len(filenames):
        name = filenames[i]
        parts = name.split(symbol)
        exp, name_i = parts[len(parts)-2], parts[len(parts)-1]
        # read image
        image_ = cv2.imread(name)
    else:
        image_, name_i = [], []

    return filenames, image_, exp, name_i


def update_dir(path):
    path_s = path.split('/')
    cad, path_f = len(path_s), path_s[0]
    for p in range(1, cad):
        path_f += '\\' + path_s[p]
    return path_f


def bytes_(img):
    ima = cv2.resize(img, (350, 250))
    return cv2.imencode('.png', ima)[1].tobytes()


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)

    return figure_canvas_agg


def preprocessing(img):
    image_gray_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clh = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    clh_img = clh.apply(image_gray_)

    # filter image
    final_img_ = cv2.GaussianBlur(clh_img, (5, 5), 0)

    return image_gray_, final_img_


def build_filters():
    filters_, k_size, sigma = [], 21, 3.0
    for theta in np.arange(0, np.pi, np.pi / 4):
        kern = cv2.getGaborKernel((k_size, k_size), sigma, theta, 10.0, 0.50, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters_.append(kern)
    return filters_


def apply_gabor(img, filters):
    gabor_img_ = np.zeros_like(img)
    for kern in filters:
        np.maximum(gabor_img_, cv2.filter2D(img, cv2.CV_8UC3, kern), gabor_img_)
    return gabor_img_


def calculate_contour(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    big_contour, area_contour, cx_contour, cy_contour, id_contour = [], [], [], [], []
    i = 0
    for c in contours:
        area = cv2.contourArea(c)
        M = cv2.moments(c)
        if area > 100:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            big_contour.append(c)
            area_contour.append(area)
            cx_contour.append(cx)
            cy_contour.append(cy)
            id_contour.append(i)
        i += 1
    return big_contour, area_contour, cx_contour, cy_contour, id_contour


def generate_contour(img, mark):
    ima_sel_ = img.copy()
    ima_sel_[mark == 0] = 0

    contours, area, cx, cy, ide = calculate_contour(mark)
    color1 = (0, 255, 0)
    # draw contours
    cv2.drawContours(ima_sel_, contours, -1, color1, 3)

    cant_back_ = np.sum(mark == 0)
    cant_cell_ = np.sum(mark == 1)
    tot_pix_ = cant_back_ + cant_cell_
    percent_ = np.round((cant_cell_ * 100) / tot_pix_, 2)

    return ima_sel_, cant_back_, cant_cell_, percent_


def fc_show_img(total, img, img_grad, img_lab_, ima_sel_, des):
    if total % 25 == 0:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16), sharex=True, sharey=True)
        ax = axes.ravel()
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title("Original")
        ax[1].imshow(img_grad, cmap='nipy_spectral')
        ax[1].set_title("Gabor image")
        ax[2].imshow(img_lab_, cmap='binary')
        ax[2].set_title("Segmented image")
        ax[3].imshow(ima_sel_, cmap='gray')
        ax[3].set_title("Final image")
        fig.tight_layout()
        plt.savefig(des)
        fig.clf()
        plt.close()
    # control RAM memory
    del img, img_grad, img_lab_, ima_sel_
    gc.collect()


def graph_data(des, exp, ide):
    _root_data = os.path.join(des, 'Results_'+exp+'_'+ide+'.csv')
    data_ = pd.read_csv(_root_data)
    y = np.array(data_['Percentage'])
    x = np.arange(1, len(y) + 1, 1)
    fig = plt.figure()
    plt.plot(x, y, 'o')
    plt.grid()
    plt.xlabel('N. of image')
    plt.ylabel('Percentage')
    _root_fig = os.path.join(des, 'Percentage_'+exp+'_'+ide+'.jpg')
    fig.tight_layout()
    plt.savefig(_root_fig)


def save_csv_file(data, des, exp, ide, header):
    # Save data in csv file
    _root_result = os.path.join(des, header+'_'+exp+'_'+ide+'.csv')
    data.to_csv(_root_result, index=False)
    print('----------------------------------------------')
    print('..... Save data in CSV file successfully .....')
    print('----------------------------------------------')


def serial_test(port_n, bauds):
    try:
        c, port = 1, serial.Serial(port=port_n,
                                   baudrate=bauds,
                                   bytesize=serial.EIGHTBITS,
                                   parity=serial.PARITY_NONE,
                                   stopbits=serial.STOPBITS_ONE)
        print('--------------------------------------')
        print('           Port: ' + port.name)
        print('----------- Test successfully --------')

    except serial.SerialException:
        print('------- Port is not available ---------')
        c = 0
    return c


def pump_control(port_n, bauds, mean_, time_, act, area_sup, area_inf, flu_sup, flu_inf):
    v_fluid = 0
    try:
        port = serial.Serial(port=port_n,
                             baudrate=bauds,
                             bytesize=serial.EIGHTBITS,
                             parity=serial.PARITY_NONE,
                             stopbits=serial.STOPBITS_ONE)
        if 0 < area_sup < mean_:
            action = True
            v_fluid = flu_sup
        elif area_inf > 0 and mean_ < area_inf:
            action = True
            v_fluid = flu_inf
        else:
            action = False

        if action:
            # Start pump
            port.write(b'<<J000R>\n')
            time.sleep(1)
            # Vary flow pump
            cad_port = '<<J000F0'+str(v_fluid)+'.0000>\n'
            port.write(bytes(cad_port.encode()))
            time.sleep(time_)
            # Stop pump
            port.write(b'<<J000S>\n')
            port.close()
            act = 'ON'

    except serial.SerialException:
        print('------- Not connect to PUMP ---------')
    return act, v_fluid


def buffer_mean(k_, buffer_, buffer_size_, area_):
    val_mean = 0
    if k_ < buffer_size_-1:
        buffer_.append(area_)
        k_ += 1
    else:
        c = np.array(buffer_).reshape(-1, 1)
        model = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(c)
        distances, indices = model.kneighbors(c)
        outlier_index = np.where(distances.mean(axis=1) > 1.0)

        if len(outlier_index[0]) > 0:
            aux_ = [x for i1, x in enumerate(c) if i1 != outlier_index[0].all]
            val_mean = np.average(aux_)
        else:
            val_mean = np.average(np.array(c))
        k_ = 0
        buffer_ = []

    return k_, buffer_, np.round(val_mean, 2)


def variation_cf(gabor):
    hist = cv2.calcHist([gabor], [0], None, [256], [0, 256])
    hist = hist.ravel()
    mean_, sigma_ = np.mean(hist), np.std(hist)
    return sigma_ / mean_


def adaptive_ini(gabor, error):
    p1, p2, p3, p4 = 1.40, 1.50, 1.65, 1.80
    cv_ima = variation_cf(gabor)
    print(cv_ima)
    if cv_ima < p1 + error:
        adapt_ini = 9
    elif cv_ima < p2 + error:
        adapt_ini = 11
    elif cv_ima < p3 + error:
        adapt_ini = 5
    elif cv_ima < p4 + error:
        adapt_ini = 6
    else:
        adapt_ini = 5
    return cv_ima, adapt_ini


def max_f(val_, max_):
    ctr_ = False
    if val_ > max_:
        ctr_ = True
        max_ = val_
    return ctr_, max_


def min_f(val_, min_):
    ctr_ = False
    if val_ < min_:
        ctr_ = True
        min_ = val_
    return ctr_, min_


def max_min(max_, min_, y_reg_f):
    var_ = min_ / max_
    increase, max1 = max_f(y_reg_f, max_)
    decrease, min1 = min_f(y_reg_f, min_)
    ctr_ = 0
    # window parameter <<<<< -------------------------------------------------
    if 0.94 > var_ > 0.93:
        if increase:
            ctr_ = 1
        elif decrease:
            ctr_ = 2
        max_ = y_reg_f
        min_ = y_reg_f
    else:
        if increase:
            max_ = max1
        elif decrease:
            min_ = min1

    return max_, min_, ctr_


def verify_seq(y_base_, y_reg_, m1_, m_):
    error = np.abs(y_base_[0] - y_reg_[0])
    ctr_ = False
    if (m1_ > 0 and m_ < 0) or (m1_ < 0 and m_ > 0):
        ctr_ = True
    elif (m1_ > 0 and m_ > 0) or (m1_ < 0 and m_ < 0):
        # error parameter <<<<< -------------------------------------------------
        if error > 0.07:
            ctr_ = True

    return ctr_, m_


def regression(x_, y_):
    x_ = np.array(x_, dtype=np.float64)
    y_ = np.array(y_, dtype=np.float64)
    m_ = ((mean(x_)*mean(y_)) - mean(x_*y_)) / (mean(x_)**2 - mean(x_**2))
    b_ = mean(y_) - m_*mean(x_)
    return b_, m_


def adap_min(k_, x_, y_, data_, cont_, max_v_, min_v_, adap1_, adap_, y_base_, m1_):
    x_.append(k_+1)
    y_.append(data_[k_])
    cont_ += 1

    adaptive_val = adap_
    if cont_ == 3:
        b, m = regression(x_, y_)
        y_base_ = [(m * x1) + b for x1 in x_]
        m1_ = copy.deepcopy(m)
    elif cont_ > 3:
        b, m = regression(x_, y_)
        y_reg = [(m * x1) + b for x1 in x_]

        ctr, m1_ = verify_seq(y_base_, y_reg, m1_, m)

        if ctr:
            x_ = [k_-1]
            y_ = [data_[k_-1]]
            cont_ = 0
        else:
            y_final = copy.deepcopy(y_reg)
            max_v_, min_v_, control = max_min(max_v_, min_v_, y_final[-1])
            if adap1_ < 11:
                if control == 1:
                    adaptive_val -= 1
                elif control == 2:
                    adaptive_val += 1
            else:
                if control == 1:
                    adaptive_val += 1
                elif control == 2:
                    adaptive_val -= 1

    adaptive_val = adaptive_val if adaptive_val >= 5 else 5
    adaptive_val = adaptive_val if adaptive_val <= 11 else 11

    return x_, y_, max_v_, min_v_, y_base_, m1_, cont_, adaptive_val


def gema(total, data, ima, ima_name, filters, buffer_size, buffer, k, lambda_g, cf_vars, des,
         x, y, cont, max_v, min_v, adap1, adap, y_base, m1):
    print('')
    print('----------------------------------------------')
    print('Processing image with GEMA: ... ' + str(total))
    tic = time.process_time()
    image_gray, final_img = preprocessing(ima)
    gabor_img = apply_gabor(final_img, filters)

    # computed adaptive values with histogram
    if total == 0:
        cf_var, adap = adaptive_ini(gabor_img, 0.10)
        cf_vars.append(cf_var)
        x, y = [k+1], [cf_vars[k]]
        adap1 = copy.deepcopy(adap)
        max_v, min_v = cf_vars[k], cf_vars[k]
    else:
        cf_vars.append(variation_cf(gabor_img))
        x, y, max_v, min_v, y_base, m1, cont, adap = adap_min(total, x, y, cf_vars, cont, max_v, min_v, adap1,
                                                              adap, y_base, m1)

    # threshold
    thresh = cv2.adaptiveThreshold(gabor_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 75, adap)

    # apply morphology close with a circular shaped kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    binary = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    # filter small regions around interest area
    arr = binary > 0
    principal = morphology.remove_small_objects(arr, min_size=100, connectivity=1).astype(np.uint8)
    contours, area, cx, cy, ide = calculate_contour(principal)
    a_max, a_min, n_area = np.max(area), np.min(area), len(area)
    # threshold area value computed by image
    a_thr = np.round((a_max + a_min) / (0.25 * n_area))
    # remove region outside interest area
    arr = principal > 0
    principal = morphology.remove_small_objects(arr, min_size=a_thr, connectivity=1).astype(np.uint8)

    # apply morphology dilate with a circular shaped kernel
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate = cv2.morphologyEx(principal, cv2.MORPH_DILATE, kernel2, iterations=3)

    # remove small holes inside interest area
    contours, area, cx, cy, ide = calculate_contour(dilate)
    a_max, a_min, n_area = np.max(area), np.min(area), len(area)
    # threshold area value computed by image
    a_thr = np.round((a_max + a_min) / (lambda_g * n_area))

    markers = morphology.remove_small_holes(dilate.astype(np.bool), area_threshold=a_thr, connectivity=2)
    markers = markers.astype(np.uint8)

    # obtain regions with cells inside image
    ima_out, cant_back, cant_cell, percent = generate_contour(ima, markers)

    print('')
    table = [['Background value      : ', str(cant_back)],
             ['Cells area value      : ', str(cant_cell)],
             ['Percentage area value : ', str(percent)]]
    for line in table:
        print('{:>10} {:>10}'.format(*line))
    print('')
    time_p = np.round(time.process_time() - tic, 2)
    print('Time used by GEMA     : ' + str(time_p) + ' sec.')
    print('Adaptive value used   : ' + str(adap))

    print('-------------------------------------------------------------------------')
    print('Loading buffer of Percentage AREA...:  ' + str(k+1) + ' of ' + str(buffer_size))
    k, buffer, mean_area = buffer_mean(k, buffer, buffer_size, percent)

    # output image binary
    if np.prod(markers[:]) == 0:
        markers = 1 - markers

    # Output image
    nom_img_sp = ima_name.split('.')[0] + '_gema.jpg'
    _root_des = os.path.join(des, nom_img_sp)
    # Save obtained data
    data = data.append({'Image': ima_name, 'Percentage': percent, 'Back_area': cant_back,
                        'Cell_area': cant_cell, 'Time (sec.)': time_p}, ignore_index=True)
    # Save final images
    fc_show_img(total, ima, gabor_img, markers, ima_out, _root_des)

    del ima, gabor_img, markers, image_gray, final_img
    gc.collect()

    return ima_out, data, percent, k, buffer, mean_area, cf_vars, x, y, max_v, min_v, y_base, m1, cont, adap


def original(total, data, ima, ima_name, buffer_size, buffer, k, umbra, des):
    print('')
    print('----------------------------------------------')
    print('Processing image with ORIGINAL: ... ' + str(total))
    tic = time.process_time()

    image_gray, final_img = preprocessing(ima)
    gradient = rank.gradient(final_img, disk(20))
    markers = rank.gradient(final_img, disk(20)) > umbra
    markers = np.array(1*np.array(markers)).astype(np.uint8)

    # obtain regions with cells inside image
    ima_out, cant_back, cant_cell, percent = generate_contour(ima, markers)

    print('')
    table = [['Background value      : ', str(cant_back)],
             ['Cells area value      : ', str(cant_cell)],
             ['Percentage area value : ', str(percent)]]
    for line in table:
        print('{:>10} {:>10}'.format(*line))
    print('')
    time_p = np.round(time.process_time() - tic, 2)
    print('Time used by ORIGINAL     : ' + str(time_p) + ' sec.')

    print('-------------------------------------------------------------------------')
    print('Loading buffer of Percentage AREA...:  ' + str(k) + ' of ' + str(buffer_size-1))
    k, buffer, mean_area = buffer_mean(k, buffer, buffer_size, percent)

    # output image binary
    if np.prod(markers[:]) == 0:
        markers = 1 - markers

    # Output image
    nom_img_sp = ima_name.split('.')[0] + '_original.jpg'
    _root_des = os.path.join(des, nom_img_sp)
    # Save obtained data
    data = data.append({'Image': ima_name, 'Percentage': percent, 'Back_area': cant_back,
                        'Cell_area': cant_cell, 'Time (sec.)': time_p}, ignore_index=True)
    # Save final images
    fc_show_img(total, ima, gradient, markers, ima_out, _root_des)

    del ima, gradient, markers, image_gray, final_img
    gc.collect()

    return ima_out, data, percent, k, buffer, mean_area
