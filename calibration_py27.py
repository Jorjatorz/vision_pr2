import time
import cv2
import misc
import glob
import copy
import numpy as np
from operator import itemgetter
from scipy.misc import imread
import matplotlib.pyplot as ppl
from matplotlib.pyplot import imshow, plot


def play_ar(intrinsic, extrinsic, imgs, model):
    fig = ppl.gcf()

    v = model.vertices
    e = model.edges

    for T, img in zip(extrinsic, imgs):
        fig.clf()

        # Do not show invalid detections.
        if T is None:
            continue

        # TODO: Project the model with proj.
        # Hint: T is the extrinsic matrix for the current image.
        m = proj(intrinsic, T, v)

        # TODO: Draw the model with plothom or plotedges.
        plothom(m)


        # Plot the image.
        imshow(img)
        ppl.show()
        time.sleep(0.1)


def calibrate(image_corners, chessboard_points, image_size):
    """Calibrate a camera.
    This function determines the intrinsic matrix and the extrinsic
    matrices of a camera.
    Parameters
    ----------
    image_corners : list
        List of the M outputs of cv2.findChessboardCorners, where
        M is the number of images.
    chessboard_points : ndarray
        Nx3 matrix with the (X,Y,Z) world coordinates of the
        N corners of the calibration chessboard pattern.
    image_size : tuple
        Size (height,width) of the images captured by the camera.
    Output
    ------
    intrinsic : ndarray
        3x3 intrinsic matrix
    dist_coefs: Non-linear distortion coefficients for the camera
    """
    valid_corners = [carr for validP, carr in image_corners if validP]
    num_images = len(image_corners)
    num_valid_images = len(valid_corners)
    num_corners = len(valid_corners[0][1])

    # Input data.
    object_points = np.array([chessboard_points] * num_valid_images, dtype=np.float32)
    image_points = np.array([carr[:, 0, :] for carr in valid_corners], dtype=np.float32)
    # Output matrices.
    intrinsics = np.identity(3)
    dist_coeffs = np.zeros(4)

    # Calibrate for square pixels
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size,
                                                                     intrinsics, dist_coeffs,
                                                                     flags=cv2.CALIB_FIX_ASPECT_RATIO)

    # def vecs2matrices(rvec,tvec):
    #    R,jacov = cv2.Rodrigues(rvec)
    #    return np.vstack((np.hstack((R,tvec)),np.array([0,0,0,1])))
    def vecs2matrices(rtvec):
        R, jacov = cv2.Rodrigues(rtvec[0])
        return np.vstack((np.hstack((R, rtvec[1])), np.array([0, 0, 0, 1])))

    extrinsicsAux = map(vecs2matrices, zip(rvecs, tvecs))

    extrinsicsiter=iter(extrinsicsAux)
    extrinsics = []
    for i, corner in enumerate(image_corners):
        if corner[0]:
            extrinsics.append(extrinsicsiter.next())
            #extrinsics.append(next(extrinsicsAux))
        else:
            extrinsics.append(None)

    return intrinsics, extrinsics, dist_coeffs


# Codigo de la practica
def practica2():
    # Ejercicio 1
    print("Obteniendo nombres de las imagenes en la carpeta left")
    #fileNames = glob.glob("left\left*.jpg")
    fileNames = glob.glob("./left/left*.jpg")
    fileNames = misc.sort_nicely(fileNames)

    print("Cargando imagenes de la carpeta left por orden alfabetico")
    images = load_images(fileNames)

    # Ejercicio 2
    print("Detectando esquinas del tablero en las imagenes")
    boardCorners = [cv2.findChessboardCorners(image, (8, 6)) for image in
                    images]  # Buscamos los bordes en todas las imagenes y los almacenamos en una lista

    # Ejercicio 3
    print("Mostrando esquinas detectadas")
    imagesToShow = copy.deepcopy(images)
    for i in range(0, len(images)):
        if boardCorners[i][0]:  # Solo mostramos las imagenes con bordes detectados
            cv2.drawChessboardCorners(imagesToShow[i], (8, 6), boardCorners[i][1], boardCorners[i][0])
            #cv2.imshow("img", imagesToShow[i])
            #cv2.waitKey(500)
            if i==1:
                imshow(imagesToShow[i])
                ppl.show()


    # Ejercicio 4
    print("Obeteniendo coordenadas en el mundo de las esquinas")
    cornersRealPositions = get_chessboard_points((8, 6), 30, 30)

    # Ejercicio 5
    print("Calibrando camara left...")
    intrinsics, extrinsics, dist_coeffs = calibrate(boardCorners, cornersRealPositions, (320, 240))
    print("Guardando resultados de la calibracion en el archivo calib_left")
    # np.savez("calib_left", intrinsic=intrinsics, extrinsic=extrinsics)

    # Ejercicio 6
    FOV_x = 2 * np.arctan(320 / (2 * intrinsics[0][0]))
    FOV_y = 2 * np.arctan(240 / (2 * intrinsics[1][1]))
    FOV_d = 2 * np.arctan2(np.sqrt(320 * 320 + 240 * 240) / 2, intrinsics[0][0])  # Se usa fx dado que es igual que fy
    print("FOV diagonal de la camara left (grados) = {}".format(FOV_d * 180 / np.pi))

    # Ejercicio 7,8,9
    from models import teapot
    play_ar(intrinsics, extrinsics, images, teapot)

    # Ejercicio 10
    T = np.zeros((4, 4))
    T[3][3] = 1
    T[:-1, :-1] = misc.ang2rotmatrix(0, 0, np.pi / 2.)
    T[0, 3] = (cornersRealPositions[-1,0] - cornersRealPositions[0,0])/2.
    T[1, 3] = (cornersRealPositions[-1,1] - cornersRealPositions[0,1])/2.
    T[2, 3] = (cornersRealPositions[-1, 2] - cornersRealPositions[0, 2]) / 2.
    '''
    las tres ultimas lineas se pueden escribir como la siguiente, dejamos lo que te parezca mas legible
    T[:3,3] = (cornersRealPositions[-1,:] - cornersRealPositions[0,:]) / 2.
    '''
    # extrinsics empieza a indexar en la posicion 1, tiene None en la posicion 0
    extrinsics2 = [np.matmul(e, T) if e is not None else None for e in extrinsics]
    play_ar(intrinsics, extrinsics2, images, teapot)

# Carga las imagenes y devuelve una lista con arrays numpy
# parametro: lista de nombres de las imagenes
def load_images(fileNames):
    images = [imread(file) for file in fileNames]
    return images


# Devuelve las coordenadas x,y,z en el mundo de un conjunto de puntos N
def get_chessboard_points(chessboard_shape, dx, dy):
    N = chessboard_shape[0] * chessboard_shape[1]
    coordinates = []

    ladoLargo = chessboard_shape[0] if chessboard_shape[0] >= chessboard_shape[1] else chessboard_shape[1]
    ladoCorto = chessboard_shape[0] if chessboard_shape[0] < chessboard_shape[1] else chessboard_shape[1]
    for x in range(0, ladoCorto):
        for y in range(0, ladoLargo):
            coordinates.append((x * dx, y * dy, 0))

    coordinates = np.array(coordinates).reshape((N, 3))

    return coordinates


def proj(K, T, verts):
    # Multiplicamos matrices. Se le anade una columna a K de ceros para hacerla 3x4
    P = np.matmul(np.hstack((K, np.array([0, 0, 0]).reshape(1, 3).transpose())), T)
    return np.matmul(P, verts)


def plothom(points):
    # Transformamos de coordenadas homogeneas a castesianas y mostramos los puntos
    p = points / points[2,:]
    return ppl.plot(p[0,:], p[1,:], 'bo')


def play_ar_modified(intrinsic, extrinsic, imgs, model):
    fig = ppl.gcf()

    v = model.vertices
    e = model.edges

    for T, img in zip(extrinsic, imgs):
        fig.clf()

        # Do not show invalid detections.
        if T is None:
            continue

        # TODO: Project the model with proj.
        # Hint: T is the extrinsic matrix for the current image.
        m = proj(intrinsic, T, v)

        # TODO: Draw the model with plothom or plotedges.
        plothom(m)


        # Plot the image.
        imshow(img)
        ppl.show()
        time.sleep(0.1)

if __name__ == "__main__":
    practica2()
