import cv2 as cv
import numpy as np
import glob
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import threading
import os
import open3d as o3d

LARGE_FONT = ("Verdana", 12)
LARGE_BOLD_FONT = ("Verdana", 12, "bold")
cyml = ""


class TCCapp(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.iconbitmap(self, default="favicon.ico")
        tk.Tk.wm_title(self, "Structure from Motion")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, CamCalib, Reconstruct):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

    def combine_funcs(*funcs):
        def combined_func(*args, **kwargs):
            for f in funcs:
                f(*args, **kwargs)

        return combined_func


class StartPage(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        l = tk.Label(self, text="                                                ")
        label = tk.Label(self, text="Reconstrução 3D", font=LARGE_BOLD_FONT)
        l1 = tk.Label(self, text="                                               ")
        label1 = tk.Label(self,
                          text="Você tem acesso à câmera utilizada para a captura de imagens do objeto a ser reconstruído? ",
                          font=LARGE_FONT)
        b1 = ttk.Button(self, text="Sim", command=lambda: controller.show_frame(CamCalib))
        b2 = ttk.Button(self, text="Não", command=None)

        l.grid(row=0, column=0)
        l1.grid(row=1, column=0)
        label.grid(row=0, column=1, sticky="nsew")
        label1.grid(row=1, column=1)
        b1.grid(row=1, column=2, sticky="nsew")
        b2.grid(row=1, column=3, sticky="nsew")


class CamCalib(tk.Frame):

    # Auto-Calibração da Câmera
    def calibrate(self, dirpath, prefix, image_format, square_size, width=9, height=6):
        """ Apply camera calibration operation for images in the given directory path. """
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (8,6,0)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((height * width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
        objp = objp * square_size
        # if square_size is 1.5 centimeters, it would be better to write it as 0.015 meters. Meter is a better metric because most of the time we are working on meter level projects.
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        images = glob.glob(dirpath.get() + '/' + prefix + '*.' + image_format)

        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (width, height), None)
            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                # Draw and display the corners
                img = cv.drawChessboardCorners(img, (width, height), corners2, ret)
            #     cv.imshow('img', cv.resize(img, (800, 1400)))
            #     cv.waitKey()
            # cv.destroyAllWindows()
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(objpoints, imgpoints,
                                                                                   gray.shape[::-1], None, None)

        # return [ret, mtx, dist, rvecs, tvecs]

    def save_coefficients(self, mtx, dist, path):
        """ Save the camera matrix and the distortion coefficients to given path/file. """
        cv_file = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
        cv_file.write("K", mtx)
        cv_file.write("D", dist)
        # note you *release* you don't close() a FileStorage object
        cv_file.release()

    def browse_button(self):
        """ Store selected folder directory path """
        self.filepath.set(filedialog.askdirectory())
        global cyml
        cyml = self.filepath.get()

    def start_calib(self):
        self.b2["state"] = "disabled"
        self.b3["state"] = "disabled"
        self.label7.config(text="Calibração Iniciada, aguarde...")
        self.calibrate(dirpath=self.filepath, prefix=self.stringvar1.get(), image_format=self.rb_var.get(),
                       square_size=float(self.stringvar2.get()) / 100)
        self.label7.config(text='Calibração Concluída! Salvando no arquivo "camera.yml"...')
        self.save_coefficients(self.mtx, self.dist, self.filepath.get() + "/" + 'camera.yml')
        self.b3["state"] = "enabled"
        self.b4["state"] = "enabled"
        self.label7.config(text='Calibração Concluída! Arquivo "camera.yml" salvo em ' + self.filepath.get())

    def iniciar_trace(self, *args):
        if self.rb_var.get() and self.stringvar1.get() and self.stringvar2.get() and self.filepath.get() and not glob.glob(
                self.filepath.get() + "/" + 'camera.yml'):
            self.b2.config(state='normal')
        else:
            self.b2.config(state='disabled')

    def continue_trace(self, *args):
        if glob.glob(self.filepath.get() + "/" + 'camera.yml'):
            self.b4.config(state="enabled")
            self.label7.config(text="Arquivo 'camera.yml' encontrado em " + self.filepath.get() + ".")

    def __init__(self, parent, controller):

        # Calibration Variables
        self.ret = 0.0
        self.mtx = []
        self.dist = []
        self.rvecs = []
        self.tvecs = []

        # Tkinter page config
        tk.Frame.__init__(self, parent)
        self.filepath = tk.StringVar(self)
        self.rb_var = tk.StringVar(self)
        self.stringvar1 = tk.StringVar(self)
        self.stringvar2 = tk.StringVar(self)
        ph = tk.Label(self, text="                                   ")
        label1 = tk.Label(self, text="Calibração de Câmera", font=LARGE_BOLD_FONT)
        label3 = tk.Label(self, textvariable=self.filepath, font=LARGE_FONT)
        self.entry1 = ttk.Entry(self, width=30, textvariable=self.stringvar1)
        label2 = tk.Label(self, text="Selecione o diretório das imagens para realizar a Calibração de Câmera: ",
                          font=LARGE_FONT)
        b1 = ttk.Button(self, text="Browse", command=TCCapp.combine_funcs(self.browse_button,
                                                                          lambda: label3.config(text=self.filepath)))
        ph1 = tk.Label(self, text="                                   ")
        label4 = tk.Label(self,
                          text="Prefixo em comum das imagens no diretório (Ex: imagem1.jpg, imagem2.jpg -> Prefixo = imagem): ",
                          font=LARGE_FONT)
        self.b3 = ttk.Button(self, text="Voltar", command=TCCapp.combine_funcs(lambda: controller.show_frame(StartPage),
                                                                               lambda: self.entry1.delete(0, tk.END),
                                                                               lambda: self.entry2.delete(0, tk.END)))
        label5 = tk.Label(self, text="Formato da imagem: ", font=LARGE_FONT)
        radio1 = ttk.Radiobutton(self, text="jpg/jpeg", variable=self.rb_var, value="jpg")
        radio2 = ttk.Radiobutton(self, text="png", variable=self.rb_var, value="png")
        label6 = tk.Label(self,
                          text="Tamanho de um quadrado na folha quadriculada (Ex: Insira 2.5 para 2,5cm, por exemplo): ",
                          font=LARGE_FONT)
        self.entry2 = ttk.Entry(self, width=30, textvariable=self.stringvar2)
        self.label7 = tk.Label(self, text="", font=LARGE_BOLD_FONT)
        self.b2 = ttk.Button(self, text="Iniciar", state='disabled',
                             command=lambda: threading.Thread(target=self.start_calib).start())
        self.b4 = ttk.Button(self, text="Continuar",
                             command=TCCapp.combine_funcs(lambda: controller.show_frame(Reconstruct)),
                             state="disabled")

        # Trace Organizer
        self.rb_var.trace("w", self.iniciar_trace)
        self.stringvar1.trace("w", self.iniciar_trace)
        self.stringvar2.trace("w", self.iniciar_trace)
        self.filepath.trace("w", self.continue_trace)

        # Grid Organizer
        ph.grid(row=0, column=0)
        ph1.grid(row=1, column=4)
        b1.grid(row=1, column=3)
        self.b2.grid(row=6, column=3, sticky="nsw")
        self.b3.grid(row=6, column=3, sticky="nse")
        self.b4.grid(row=7, column=3)
        self.entry1.grid(row=3, column=3)
        self.entry2.grid(row=5, column=3)
        label1.grid(row=0, column=2)
        label2.grid(row=1, column=2, sticky="nse")
        label3.grid(row=2, column=2, sticky="nsew")
        label4.grid(row=3, column=2)
        label5.grid(row=4, column=2, sticky="nse")
        label6.grid(row=5, column=2, sticky="nse")
        self.label7.grid(row=7, column=2)
        radio1.grid(row=4, column=3, sticky="nsw")
        radio2.grid(row=4, column=3, sticky="nse")


class Reconstruct(tk.Frame):

    def browse_button(self):
        """ Store selected folder directory path """
        self.filepath = filedialog.askdirectory()

    def load_coefficients(self, path):
        """ Loads camera matrix and distortion coefficients. """
        # FILE_STORAGE_READ
        cv_file = cv.FileStorage(path + '/' + 'camera.yml', cv.FILE_STORAGE_READ)

        # note we also have to specify the type to retrieve other wise we only get a
        # FileNode object back instead of a matrix
        self.mtx = cv_file.getNode("K").mat()
        self.mtx_inv = np.linalg.inv(self.mtx)
        self.dist = cv_file.getNode("D").mat()

        cv_file.release()

    def new_folder(self):
        self.newpath = self.filepath + '/' + 'Undistorted Pictures' + '/' + os.path.splitext(os.path.basename(str(self.video[0])))[0]
        if not os.path.exists(self.newpath):
            os.makedirs(self.newpath)

    def trace_rb(self, *args):
        if self.rb_var.get() == ".mp4":
            self.video = glob.glob(self.filepath+"/"+self.prefix.get() + "*" +self.rb_var.get())
            self.r4["state"] = "enabled"
            self.r5["state"] = "enabled"
            self.r6["state"] = "enabled"
        elif self.rb_var.get() == ".jpg":
            self.images = glob.glob(self.filepath+"/"+self.prefix.get() + "*" +self.rb_var.get())
            self.r4["state"] = "disabled"
            self.r5["state"] = "disabled"
            self.r6["state"] = "disabled"
        elif self.rb_var.get() == ".png":
            self.images = glob.glob(self.filepath+"/"+self.prefix.get() + "*" +self.rb_var.get())
            self.r4["state"] = "disabled"
            self.r5["state"] = "disabled"
            self.r6["state"] = "disabled"

    def _in_front_of_both_cameras(self, first_points, second_points, rot, trans):
        rot_inv = rot
        for first, second in zip(first_points, second_points):
            first_z = np.dot(rot[0, :] - second[0] * rot[2, :], trans) / np.dot(rot[0, :] - second[0] * rot[2, :], second)
            first_3d_point = np.array([first[0] * first_z, second[0] * first_z, first_z])
            second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)

        if first_3d_point[2] < 0 or second_3d_point[2] < 0:
            return False
        return True

    def video_routine(self):
        self.load_coefficients(cyml)
        self.b2.config(state="disabled")
        self.b3.config(state="disabled")
        if self.rb_var.get() == ".mp4":
            self.new_folder()
            vidcap = cv.VideoCapture(self.video[0])
            length = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
            print("video length: " + str(length))
            success, size = vidcap.read()
            framelist = list(range(0, length, self.frameskip.get()))
            count = 0
            sift = cv.SIFT_create()
            matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
            # knn_matches = []
            keypoints1 = []
            keypoints2 = []
            descriptors1 = []
            descriptors2 = []
            h, w = size.shape[:2]
            self.newcameramtx, self.roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
            while success:
                if count in framelist:
                    i = framelist.index(count)
                    print(i)
                    print("img 1: " + str(count))
                    print("img 2: " + str(count + self.frameskip.get()))
                    vidcap.set(1, count)
                    success, image1 = vidcap.read()
                    vidcap.set(1, count+self.frameskip.get())
                    # try:
                    success, image2 = vidcap.read()
                    cv.imshow("Imagem 1", image1)
                    cv.imshow("Imagem 2", image2)
                    k = cv.waitKey(0)
                    if success:
                        dst1 = cv.undistort(image1, self.mtx, self.dist, None, self.newcameramtx)
                        dst2 = cv.undistort(image2, self.mtx, self.dist, None, self.newcameramtx)
                        x, y, w, h = self.roi
                        dst1 = dst1[y:y + h, x:x + w]  # undistorted img
                        dst2 = dst2[y:y + h, x:x + w]  # undistorted img
                        gray1 = cv.cvtColor(dst1, cv.COLOR_BGR2GRAY)
                        gray2 = cv.cvtColor(dst2, cv.COLOR_BGR2GRAY)
                        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
                        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
                        # good_matches = matcher.match(descriptors1, descriptors2)
                        knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)

                        ratio_thresh = 0.7
                        good_matches = []
                        for m, n in knn_matches:
                            if m.distance < ratio_thresh * n.distance:
                                good_matches.append(m)

                        first_match_points = np.zeros((len(good_matches), 2), dtype=np.float32)
                        second_match_points = np.zeros_like(first_match_points)
                        for i in range(len(good_matches)):
                            first_match_points[i] = keypoints1[good_matches[i].queryIdx].pt
                            second_match_points[i] = keypoints2[good_matches[i].trainIdx].pt

                        self.match_pts1 = first_match_points
                        self.match_pts2 = second_match_points

                        self.F, self.Fmask = cv.findFundamentalMat(self.match_pts1, self.match_pts2, cv.FM_RANSAC, 0.1, 0.99)
                        self.E = self.mtx.T.dot(self.F).dot(self.mtx)
                        U, S, Vt = np.linalg.svd(self.E)
                        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)

                        first_inliers = []
                        second_inliers = []
                        for i in range(len(self.Fmask)):
                            if self.Fmask[i]:
                                first_inliers.append(self.mtx_inv.dot([self.match_pts1[i][0], self.match_pts1[i][1], 1.0]))
                                second_inliers.append(self.mtx_inv.dot([self.match_pts2[i][0], self.match_pts2[i][1], 1.0]))

                        R = U.dot(W).dot(Vt)
                        T = U[:, 2]

                        if not self._in_front_of_both_cameras(first_inliers, second_inliers, R, T):
                            # Second choice: R = U * W * Vt, T = -u_3
                            T = - U[:, 2]

                        if not self._in_front_of_both_cameras(first_inliers, second_inliers, R, T):
                            # Third choice: R = U * Wt * Vt, T = u_3
                            R = U.dot(W.T).dot(Vt)
                            T = U[:, 2]

                        if not self._in_front_of_both_cameras(first_inliers, second_inliers, R, T):
                            # Fourth choice: R = U * Wt * Vt, T = -u_3
                            T = - U[:, 2]

                        self.match_inliers1 = first_inliers
                        self.match_inliers2 = second_inliers
                        self.Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
                        self.Rt2 = np.hstack((R, T.reshape(3, 1)))

                        first_inliers = np.array(self.match_inliers1).reshape(-1, 3)[:, :2]
                        second_inliers = np.array(self.match_inliers2).reshape(-1, 3)[:, :2]

                        pts4D = cv.triangulatePoints(self.Rt1, self.Rt2, first_inliers.T, second_inliers.T).T
                        pts3D = pts4D[:, :3] / np.repeat(pts4D[:, 3], 3).reshape(-1, 3)

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(pts3D)
                        o3d.io.write_point_cloud("PointCloud" + str(i) + ".ply", pcd, write_ascii=True)

                        o3d.visualization.draw_geometries([pcd])
                        cv.waitKey()

                    # except Exception as e:
                    #     print(e)
                    #     break


                count += 1







        self.b2.config(state="enabled")
        self.b3.config(state="enabled")

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.filepath = tk.StringVar(self)
        self.newpath = ""
        self.rb_var = tk.StringVar(self)
        self.prefix = tk.StringVar(self)
        self.frameskip = tk.IntVar(self)
        self.video = []
        self.images = []
        self.mtx = []
        self.mtx_inv = []
        self.dist = []

        ph = tk.Label(self, text="                                   ")
        title = tk.Label(self, text="Reconstrução 3D", font=LARGE_BOLD_FONT)
        l1 = tk.Label(self, text="Diretório com o vídeo ou as fotos do objeto a ser reconstruído: ", font=LARGE_FONT)
        l2 = tk.Label(self, textvariable=self.filepath.get(), font=LARGE_FONT)
        l3 = tk.Label(self, text="Selecione o tipo de arquivo do vídeo ou das fotos do objeto a ser reconstruído: ",
                      font=LARGE_FONT)
        l4 = tk.Label(self, text="Digite o nome do arquívo de vídeo ou o prefixo em comum das imagens: ",
                      font=LARGE_FONT)
        b1 = ttk.Button(self, text="Browse",
                        command=TCCapp.combine_funcs(self.browse_button, lambda: l2.config(text=self.filepath)))
        self.b2 = ttk.Button(self, text="Voltar", command=lambda: controller.show_frame(CamCalib))
        self.b3 = ttk.Button(self, text="Iniciar", command=TCCapp.combine_funcs(lambda: threading.Thread(target=self.video_routine).start()))
        r1 = ttk.Radiobutton(self, text="Vídeo (.mp4)", variable=self.rb_var, value=".mp4")
        r2 = ttk.Radiobutton(self, text="Fotos (.jpg/.jpeg)", variable=self.rb_var, value=".jpg")
        r3 = ttk.Radiobutton(self, text="Fotos (.png)", variable=self.rb_var, value=".png")
        l5 = tk.Label(self, text="Quanto do vídeo deve ser usado na reconstrução?", font=LARGE_FONT)
        self.r4 = ttk.Radiobutton(self, text="1/4", variable=self.frameskip, value=4, state="disabled")
        self.r5 = ttk.Radiobutton(self, text="1/2", variable=self.frameskip, value=2, state="disabled")
        self.r6 = ttk.Radiobutton(self, text="Completo", variable=self.frameskip, value=1, state="disabled")
        e1 = ttk.Entry(self, width=30, textvariable=self.prefix)

        # Trace Organizer
        self.rb_var.trace("w", self.trace_rb)

        # Grid Organizer
        ph.grid(row=0, column=0)
        title.grid(row=0, column=1)
        l1.grid(row=1, column=1, sticky="nse")
        l2.grid(row=2, column=1)
        l3.grid(row=4, column=1, sticky="nse")
        l4.grid(row=3, column=1, sticky="nse")
        l5.grid(row=6, column=1, sticky="nse")
        b1.grid(row=1, column=2)
        self.b2.grid(row=8, column=2, sticky="nse")
        self.b3.grid(row=8, column=2, sticky="nsw")
        r1.grid(row=4, column=2, sticky="nsw")
        r2.grid(row=5, column=2)
        r3.grid(row=4, column=2, sticky="nse")
        self.r4.grid(row=6, column=2, sticky="nsw")
        self.r5.grid(row=6, column=2, sticky="nse")
        self.r6.grid(row=7, column=2)
        e1.grid(row=3, column=2)


app = TCCapp()
app.mainloop()