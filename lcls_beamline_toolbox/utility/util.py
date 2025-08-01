"""
util module for xraybeamline2d package
"""
import numpy as np
import numpy as xp
import scipy as sp
#try:
#    import cupy as xp
#    import cupyx.scipy as sp
#except ImportError:
#    import numpy as xp
#    import scipy as sp
import scipy.special
import scipy.optimize as optimize
import scipy.spatial.transform as transform
from scipy.integrate import cumulative_trapezoid


class Util:
    """
    Class for defining helper static methods.
    No attributes.
    """
    @staticmethod
    def interp_flip(x, xp, fp):
        """
        Helper function to deal with flipped input
        :param x: (N,) numpy array
            points to interpolate onto
        :param xp: (M,) numpy array
            points at which the function is known
        :param fp: (M,) numpy array
            function values, same length as xp
        :return y: (N,) numpy array
            interpolated values
        """
        # check if array is backwards
        if xp[0] > xp[1]:
            y = np.interp(x, np.flipud(xp), np.flipud(fp), left=0, right=0)
        else:
            y = np.interp(x, xp, fp, left=0, right=0)

        return y

    @staticmethod
    def pyfft(a, fft_object):
        fft_object.input_array[:] = fft.ifftshift(a)
        return fft.fftshift(fft_object())

    @staticmethod
    def ipyfft(a, ifft_object):
        ifft_object.input_array[:] = fft.ifftshift(a)
        return fft.fftshift(ifft_object())
    
    @staticmethod
    def laplace(array,p,q):
        # xp = cp.get_array_module(array)
        # sp = cupyx.scipy.get_array_module(array)

        N,M = xp.shape(array)
        out = -sp.fft.idctn(sp.fft.dctn(array) * (p ** 2 + q ** 2)) * 4 * np.pi ** 2 / (N * M)
        return out

    @staticmethod
    def inverse_laplace(array,p,q):
        # xp = cp.get_array_module(array)
        # sp = cupyx.scipy.get_array_module(array)

        N, M = xp.shape(array)
        out = -sp.fft.idctn(sp.fft.dctn(array) / (p ** 2 + q ** 2 + np.finfo(float).eps)) * M * N / (4 * np.pi ** 2)
        return out

    @staticmethod
    def solvePoisson(array, pix=1):
        # xp = cp.get_array_module(array)
        # sp = cupyx.scipy.get_array_module(array)

        N, M = xp.shape(array)
        rho_hat = sp.fft.dctn(array)
        i = xp.linspace(0, M - 1, M)
        j = xp.linspace(0, N - 1, N)
        i, j = xp.meshgrid(i, j)
        divisor = 2 * (xp.cos(np.pi * i / M) + xp.cos(np.pi * j / N) - 2)
        divisor[0, 0] = 1
        phi_hat = rho_hat / divisor
        phi_hat[0, 0] = 0
        phi = sp.fft.idctn(phi_hat) * pix**2

        return phi

    @staticmethod
    def applyQ(p, WW, pix=1):
        # xp = cp.get_array_module(p)

        N, M = xp.shape(p)
        dx = xp.hstack((xp.diff(p, axis=1), xp.zeros((N, 1)))) / pix
        dy = xp.vstack((xp.diff(p, axis=0), xp.zeros((1, M)))) / pix

        WWdx = WW * dx
        WWdy = WW * dy

        WWdx2 = xp.hstack((xp.zeros((N, 1)), WWdx))
        WWdy2 = xp.vstack((xp.zeros((1, M)), WWdy))
        Qp = xp.diff(WWdx2, axis=1) / pix + xp.diff(WWdy2, axis=0) / pix

        return Qp

    @staticmethod
    def wrapToPi(array):
        # xp = cp.get_array_module(array)

        out = xp.mod(array, 2 * np.pi)
        out[out > np.pi] -= 2 * np.pi
        return out

    @staticmethod
    def laplacian_from_gradient(grad_x, grad_y, pix=1, weight=None):
        N, M = xp.shape(grad_x)

        if weight is None:
            weight = xp.ones_like(grad_x)

        WWdx2 = xp.hstack((xp.zeros((N,1)), grad_x * weight))
        WWdy2 = xp.vstack((xp.zeros((1,M)), grad_y * weight))

        laplacian = xp.diff(WWdx2, axis=1) / pix + xp.diff(WWdy2, axis=0) / pix

        return laplacian

    @staticmethod
    def integrate_gradient_gpu(grad_x, grad_y, pix=1, weight=None, eps=1e-8):
        # xp = cp.get_array_module(psi)

        N, M = xp.shape(grad_x)
        # dx = xp.hstack((Util.wrapToPi(xp.diff(psi, axis=1)), xp.zeros((N, 1))))
        # dy = xp.vstack((Util.wrapToPi(xp.diff(psi, axis=0)), xp.zeros((1, M))))

        if weight is None:
            weight = xp.ones_like(grad_x)

        WW = weight * weight
        # WWdx = WW * dx
        # WWdy = WW * dy
        #
        # WWdx2 = xp.hstack((xp.zeros((N, 1)), WWdx))
        # WWdy2 = xp.vstack((xp.zeros((1, M)), WWdy))
        # rk = xp.diff(WWdx2, axis=1) + xp.diff(WWdy2, axis=0)
        rk = Util.laplacian_from_gradient(grad_x, grad_y, pix=pix, weight=WW)
        normR0 = xp.linalg.norm(rk.flatten())

        k = 0
        phi = xp.zeros_like(grad_x)

        rk_old = xp.copy(rk)
        zk_old = xp.zeros_like(rk)
        zk = xp.zeros_like(rk)
        pk = xp.zeros_like(rk)
        beta = 0
        alpha = 0

        # norm1 = cp.zeros(50)

        while xp.sum(xp.abs(rk)) > 0:
            zk = Util.solvePoisson(rk, pix=pix)

            k += 1

            if k == 1:
                pk = zk
            else:
                beta = xp.dot(rk.flatten(), zk.flatten()) / xp.dot(rk_old.flatten(), zk_old.flatten())
                pk = zk + beta * pk
            # print(k)
            # print(beta)

            rk_old = xp.copy(rk)
            zk_old = xp.copy(zk)

            Qpk = Util.applyQ(pk, WW, pix=pix)

            alpha = xp.dot(rk.flatten(), zk.flatten()) / xp.dot(pk.flatten(), Qpk.flatten())
            # print(alpha)
            phi = phi + alpha * pk
            rk = rk - alpha * Qpk

            # norm1[k - 1] = cp.linalg.norm(rk.flatten())
            # print(cp.linalg.norm(rk.flatten()))
            if xp.linalg.norm(rk.flatten()) < eps * normR0 or k>20:
                print('phase unwrap stopping after {} iterations'.format(k))

                break

        return phi

    @staticmethod
    def unwrap_phase_gpu(psi, weight, eps=1e-8):
        # xp = cp.get_array_module(psi)

        N, M = xp.shape(psi)
        dx = xp.hstack((Util.wrapToPi(xp.diff(psi, axis=1)), xp.zeros((N, 1))))
        dy = xp.vstack((Util.wrapToPi(xp.diff(psi, axis=0)), xp.zeros((1, M))))

        WW = weight * weight
        WWdx = WW * dx
        WWdy = WW * dy

        WWdx2 = xp.hstack((xp.zeros((N, 1)), WWdx))
        WWdy2 = xp.vstack((xp.zeros((1, M)), WWdy))
        rk = xp.diff(WWdx2, axis=1) + xp.diff(WWdy2, axis=0)
        normR0 = xp.linalg.norm(rk.flatten())

        k = 0
        phi = xp.zeros_like(psi)

        rk_old = xp.copy(rk)
        zk_old = xp.zeros_like(rk)
        zk = xp.zeros_like(rk)
        pk = xp.zeros_like(rk)
        beta = 0
        alpha = 0

        # norm1 = cp.zeros(50)

        while xp.sum(xp.abs(rk)) > 0:
            zk = Util.solvePoisson(rk)

            k += 1

            if k == 1:
                pk = zk
            else:
                beta = xp.dot(rk.flatten(), zk.flatten()) / xp.dot(rk_old.flatten(), zk_old.flatten())
                pk = zk + beta * pk
            # print(k)
            # print(beta)

            rk_old = xp.copy(rk)
            zk_old = xp.copy(zk)

            Qpk = Util.applyQ(pk, WW)

            alpha = xp.dot(rk.flatten(), zk.flatten()) / xp.dot(pk.flatten(), Qpk.flatten())
            # print(alpha)
            phi = phi + alpha * pk
            rk = rk - alpha * Qpk

            # norm1[k - 1] = cp.linalg.norm(rk.flatten())
            # print(cp.linalg.norm(rk.flatten()))
            if xp.linalg.norm(rk.flatten()) < eps * normR0 or k>20:
                print('phase unwrap stopping after {} iterations'.format(k))

                break

        return phi

    # @staticmethod
    # def unwrap_phase_gpu(array):
    #     p,q = Util.get_spatial_frequencies(array,1)
    #     p = sfft.fftshift(p)
    #     q = sfft.fftshift(q)
    #
    #     L_wrapped = Util.laplace(array, p, q)
    #     L_unwrapped = (cp.cos(array) * Util.laplace(cp.sin(array), p, q) -
    #                    cp.sin(array) * Util.laplace(cp.cos(array), p, q))
    #     k = cp.round(Util.inverse_laplace(L_unwrapped - L_wrapped, p, q) / 2 / np.pi)
    #
    #     unwrapped = array + 2 * np.pi * k
    #
    #     return unwrapped, k

    @staticmethod
    def nfft(a):
        """
        Class method for 2D FFT with zero frequency at center
        :param a: (N,M) ndarray
            array to be Fourier transformed
        :return: (N,M) ndarray
            Fourier transformed array of same shape as a
        """
        # xp = cp.get_array_module(a)

        return xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(a)))

    @staticmethod
    def nfft1(a):
        """
        Class method for 2D FFT with zero frequency at center
        :param a: (N,M) ndarray
            array to be Fourier transformed
        :return: (N,M) ndarray
            Fourier transformed array of same shape as a
        """
        # xp = cp.get_array_module(a)

        return xp.fft.fftshift(xp.fft.fft(xp.fft.ifftshift(a)))

    @staticmethod
    def infft(a):
        """
        Class method for 2D IFFT with zero frequency at center
        :param a: (N,M) ndarray
            array to be inverse Fourier transformed
        :return: (N,M) ndarray
            Array after inverse Fourier transform, same shape as a
        """
        # xp = cp.get_array_module(a)

        return xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(a)))

    @staticmethod
    def infft1(a):
        """
        Class method for 2D IFFT with zero frequency at center
        :param a: (N,M) ndarray
            array to be inverse Fourier transformed
        :return: (N,M) ndarray
            Array after inverse Fourier transform, same shape as a
        """
        # xp = cp.get_array_module(a)

        return xp.fft.fftshift(xp.fft.ifft(xp.fft.ifftshift(a)))

    @staticmethod
    def fit_sinc_squared(x, x0, w):
        """
        Method for fitting to a sinc squared function. This method is a parameter to Scipy's optimize.curve_fit routine.
        Parameters
        ----------
        x: array_like
            Copied from Scipy docs: "The independent variable where the data is measured. Should usually be an
            M-length sequence or an (k,M)-shaped array for functions with k predictors, but can actually be any
            object." Units are meters.
        x0: float
            Initial guess for beam center (m).
        w: float
            Initial guess for sinc width (m).

        Returns
        -------
        array_like with same shape as x
            Function evaluated at all points in x.
        """
        if w == 0:
            return np.zeros_like(x)
        else:
            return np.sinc((x-x0)/w)**2

    @staticmethod
    def fit_gaussian(x, x0, w):
        """
        Method for fitting to a Gaussian function. This method is a parameter to Scipy's optimize.curve_fit routine.
        :param x: array_like
            Copied from Scipy docs: "The independent variable where the data is measured. Should usually be an
            M-length sequence or an (k,M)-shaped array for functions with k predictors, but can actually be any
            object." Units are meters.
        :param x0: float
            Initial guess for beam center (m).
        :param w: float
            Initial guess for gaussian sigma (m).
        :return: array_like with same shape as x
            Function evaluated at all points in x.
        """
        # just return an array evaluating the Gaussian function based on input parameters.
        if w == 0:
            return np.zeros_like(x)
        else:
            return np.exp(-((x - x0) ** 2 / (2 * w ** 2)))

    @staticmethod
    def fit_voigt(x, x0, wg, wl, eta):
        """
        Method for fitting to a pseudo-Voigt function. This method is a parameter to Scipy's optimize.curve_fit routine.
        :param x: array_like
            Copied from Scipy docs: "The independent variable where the data is measured. Should usually be an
            M-length sequence or an (k,M)-shaped array for functions with k predictors, but can actually be any
            object." Units are meters.
        :param x0: float
            Initial guess for beam center (m).
        :param w: float
            Initial guess for gaussian sigma (m).
        :param eta: float
            Initial guess for eta parameter
        :return: array_like with same shape as x
            Function evaluated at all points in x.
        """
        if wg == 0 or wl == 0:
            return np.zeros_like(x)
        else:
            g = np.exp(-((x - x0) ** 2 / (2 * wg ** 2)))
            l = (wl/2)**2 / ((x - x0)**2 + (wl/2)**2)

            return (eta * g + (1-eta)*l)

    @staticmethod
    def fit_lorentzian(x, x0, w):
        return (w/2)**2/((x-x0)**2+(w/2)**2)

    @staticmethod
    def fit_log_voigt(x, x0, wg, wl, eta):
        """
        Method for fitting to a pseudo-Voigt function. This method is a parameter to Scipy's optimize.curve_fit routine.
        :param x: array_like
            Copied from Scipy docs: "The independent variable where the data is measured. Should usually be an
            M-length sequence or an (k,M)-shaped array for functions with k predictors, but can actually be any
            object." Units are meters.
        :param x0: float
            Initial guess for beam center (m).
        :param w: float
            Initial guess for gaussian sigma (m).
        :param eta: float
            Initial guess for eta parameter
        :return: array_like with same shape as x
            Function evaluated at all points in x.
        """
        if wg == 0 or wl == 0:
            return np.zeros_like(x)
        else:
            g = np.exp(-((x - x0) ** 2 / (2 * wg ** 2)))
            l = (wl / 2)**2 / ((x - x0) ** 2 + (wl / 2) ** 2)

            return np.log((eta * g + (1 - eta) * l))

    @staticmethod
    def decentering(coeff, order, offset):
        """
        Method to add up phase contributions due to de-centering. Polynomial orders greater than param order
        contribute.
        :param coeff: (M+1,) array-like
            polynomial coefficients in np.polyfit ordering. Polynomial is order M.
        :param order: int
            which polynomial order to calculate for
        :param offset: float
            beam offset due to beam center and/or mirror offset, along mirror z-axis
        :return: float
            polynomial coefficient due to de-centering for param order.
        """

        # initialize output
        p_coeff = 0.0

        # polynomial order
        M = np.size(coeff) - 1

        # number of terms
        num_terms = M - order

        # loop through polynomial orders
        for i in range(num_terms):
            # current order
            n = M - i
            # difference between n and order we're calculating for
            k = n - order
            # binomial coefficient
            b_c = scipy.special.binom(n, k)
            # add contribution to p_coeff
            p_coeff += coeff[i] * b_c * offset**k

        return p_coeff

    @staticmethod
    def recenter_coeff(coeff, offset):
        """
        Method to recenter polynomial coefficients.
        :param coeff: (M+1,) array-like
            polynomial coefficients in np.polyfit ordering. Polynomial is order M.
        :param offset: float
            beam offset due to beam center and/or mirror offset, along mirror z-axis
        :return: (M+1,) array-like
            polynomial coefficients that are re-centered. Uses np.polyfit ordering. Polynomial is order M.
        """

        # initialize output
        coeff_out = np.zeros_like(coeff)

        # polynomial order
        M = np.size(coeff) - 1

        for num, coefficient in enumerate(coeff):
            # current order
            n = M - num
            # output: use coefficient from this order plus decentering contributions from higher orders
            coeff_out[num] = coeff[num] + Util.decentering(coeff, n, offset)

        return coeff_out

    @staticmethod
    def combine_coeff(coeff1, coeff2):
        """
        Method for combining polynomial coefficients that may have different polynomial order.
        :param coeff1: (M+1,) array-like
            polynomial coefficients in np.polyfit ordering. Polynomial is order M.
        :param coeff2: (N+1,) array-like
            polynomial coefficients in np.polyfit ordering. Polynomial is order N.
        :return:
        """

        # make sure we can use numpy functions
        coeff1 = np.array(coeff1)
        coeff2 = np.array(coeff2)

        # get larger of the orders
        order = np.max([np.size(coeff1), np.size(coeff2)]) - 1

        # pad arrays to ensure the size matches
        coeff1 = np.pad(coeff1, (order + 1 - np.size(coeff1), 0))
        coeff2 = np.pad(coeff2, (order + 1 - np.size(coeff2), 0))

        # combined output
        coeff_out = coeff1 + coeff2

        # print('coeff1: ' + str(coeff1))
        # print('coeff2: ' + str(coeff2))
        # print('coeff_out: ' + str(coeff_out))

        return coeff_out

    @staticmethod
    def polyval_high_order(p, x):
        """
        Method to calculate high order polynomial (ignore 2nd order and below)
        :param p: (M+1,) array-like
            polynomial coefficients in np.polyfit ordering. Polynomial is order M.
        :param x: (N,) array-like
            A number, an array of numbers, or an instance of poly1d, at which to evaluate p.
        :return values: (N,) array-like
            Evaluated polynomial at points in x.
        """
        # xp = cp.get_array_module(x)

        # remove low orders
        p2 = np.copy(p)
        p2[-3:] = 0

        # print('high order polycoeff: ' + str(p))

        # get polynomial order
        M = np.size(p2) - 1

        values = xp.zeros_like(x)

        for num, coeff in enumerate(p2):
            # order of current coefficient
            n = M - num

            # update output
            values += coeff * x**n

        return values

    @staticmethod
    def polyval_2nd(p, x):
        """
        Method to calculate high order polynomial (ignore 2nd order and below)
        :param p: (M+1,) array-like
            polynomial coefficients in np.polyfit ordering. Polynomial is order M.
        :param x: (N,) array-like
            A number, an array of numbers, or an instance of poly1d, at which to evaluate p.
        :return values: (N,) array-like
            Evaluated polynomial at points in x.
        """

        # remove low orders
        p2 = np.copy(p)
        p2[-2:] = 0

        # print('high order polycoeff: ' + str(p))

        # get polynomial order
        M = np.size(p2) - 1

        values = np.zeros_like(x)

        for num, coeff in enumerate(p2):
            # order of current coefficient
            n = M - num

            # update output
            values += coeff * x ** n

        return values

    @staticmethod
    def poly_change_coords(p, scale):
        """
        Method for scaling coefficients due to a change in coordinate system
        :param p: (M+1,) array-like
            polynomial coefficients in np.polyfit ordering. Polynomial is order M.
        :param scale: float
            Scaling between coordinate systems. Scale defined as x_new = scale * x
        :return p_new: (M+1,) array-like
            polynomial coefficients for scaled coordinates in np.polyfit ordering. Polynomial is order M.
        """

        p = np.array(p)

        # initialize output
        p_new = np.zeros_like(p)

        # get polynomial order
        M = np.size(p) - 1

        # loop through orders
        for num, coeff in enumerate(p):
            # order of current coefficient
            n = M - num
            p_new[num] = coeff / scale**n

        return p_new

    @staticmethod
    def get_borderval(img, radius=None):
        """
        Given an image and a radius, examine the average value of the image
        at most radius pixels from the edge
        """
        if radius is None:
            mindim = min(img.shape)
            radius = max(1, mindim // 20)
        mask = np.zeros_like(img, dtype=np.bool)
        mask[:, :radius] = True
        mask[:, -radius:] = True
        mask[:radius, :] = True
        mask[-radius:, :] = True

        mean = np.median(img[mask])
        return mean

    @staticmethod
    def threshold_array(array_in, frac):
        """Method for thresholding an array, useful for calculating center of mass
        :param array_in: array-like
            can be any shape array
        :param frac: float
            threshold fraction of image maximum
        :return array_out: array-like
            thresholded array, same shape as array_in
        """
        # xp = cp.get_array_module(array_in)

        # make sure the image is not complex
        array_out = xp.abs(array_in)

        # subtract minimum/background
        array_out -= xp.min(array_out)

        # get thresholding level
        thresh = xp.max(array_out) * frac
        # subtract threshold level
        array_out = array_out - thresh
        # set anything below threshold (now 0) to zero
        array_out[array_out < 0] = 0

        return array_out

    @staticmethod
    def coordinate_to_pixel(coord, dx, N):
        """
        Method to convert coordinate to pixel. Assumes zero is at the center of the array.
        Parameters
        ----------
        coord: float
            coordinate position with physical units
        dx: float
            pixel size in physical units
        N: int
            number of pixels in the array.

        Returns
        -------
        index: int
            index of pixel in the array corresponding to coord.
        """
        index = int(coord / dx) + N / 2
        return index

    @staticmethod
    def get_horizontal_lineout(array_in, x_center=None, y_center=None, half_length=None, half_width=None):
        """
        Method to get a horizontal lineout from a 2D array
        Parameters
        ----------
        array_in: (N, M) ndarray
            array to take lineout from
        x_center: int
            index of horizontal center position for the lineout
        y_center: int
            index of vertical center position for the lineout
        half_length: int
            distance from center (in pixels) to use along the lineout direction
        half_width: int
            distance from center (in pixels) to sum across for the lineout.

        Returns
        -------
        lineout: (2*half_length) ndarray
            Summed lineout from array_in (projected on horizontal axis)
        """
        # xp = cp.get_array_module(array_in)

        N, M = xp.shape(array_in)

        if x_center is None:
            x_center = int(M/2)
        if y_center is None:
            y_center = int(N/2)

        if half_length is None:
            x_start = 0
            x_end = M
        else:
            x_start = int(x_center - half_length)
            x_end = int(x_center + half_length)

        if half_width is None:
            y_start = 0
            y_end = N
        else:
            y_start = int(y_center - half_width)
            y_end = int(y_center + half_width)

        if half_width < 1:
            lineout = array_in[y_start, x_start:x_end]
        else:
            lineout = xp.sum(array_in[y_start:y_end, x_start:x_end], axis=0)

        return lineout

    @staticmethod
    def get_vertical_lineout(array_in, x_center=None, y_center=None, half_length=None, half_width=None):
        """
        Method to get a horizontal lineout from a 2D array
        Parameters
        ----------
        array_in: (N, M) ndarray
            array to take lineout from
        x_center: int
            index of horizontal center position for the lineout
        y_center: int
            index of vertical center position for the lineout
        half_length: int
            distance from center (in pixels) to use along the lineout direction
        half_width: int
            distance from center (in pixels) to sum across for the lineout.

        Returns
        -------
        lineout: (2*half_length) ndarray
            Summed lineout from array_in (projected on horizontal axis)
        """
        # xp = cp.get_array_module(array_in)

        N, M = xp.shape(array_in)

        if x_center is None:
            x_center = int(M/2)
        if y_center is None:
            y_center = int(N/2)

        if half_width is None:
            x_start = 0
            x_end = M
        else:
            x_start = int(x_center - half_width)
            x_end = int(x_center + half_width)

        if half_length is None:
            y_start = 0
            y_end = N
        else:
            y_start = int(y_center - half_length)
            y_end = int(y_center + half_length)

        if half_width < 1:
            lineout = array_in[y_start:y_end, x_start]
        else:
            lineout = xp.sum(array_in[y_start:y_end, x_start:x_end], axis=1)

        return lineout

    @staticmethod
    def get_coordinates(array_in, dx):
        """
        Method to get coordinates for a 1D or 2D array
        Parameters
        ----------
        array_in: ndarray
            array that we want coordinates for.
        dx: float
            pixel size

        Returns
        -------
        tuple of coordinate arrays with same shape as array_in
        """
        # xp = cp.get_array_module(array_in)

        array_shape = xp.shape(array_in)

        coord_list = []

        for dimension in array_shape:
            c = xp.linspace(-dimension / 2., dimension / 2. - 1, dimension, dtype=float) * dx
            coord_list.append(c)

        # make grid of spatial frequencies
        coord_tuple = xp.meshgrid(*coord_list)

        return coord_tuple

    @staticmethod
    def get_spatial_frequencies(array_in, dx):
        """
        Method to calculate spatial frequencies from array size and pixel size
        Parameters
        ----------
        array_in: ndarray
            array that we want spatial frequency coordinates for
        dx: float
            pixel size. Assume this is the same in all dimensions

        Returns
        -------
        tuple of spatial frequency arrays. Length of tuple depends is the same as length of shape.
        """
        # xp = cp.get_array_module(array_in)

        array_shape = xp.shape(array_in)

        # maximum spatial frequency
        fx_max = 1.0 / (dx * 2)

        # frequency list
        f_list = []

        for dimension in array_shape:
            df = 2 * fx_max / dimension
            f = xp.linspace(-dimension / 2., dimension / 2. - 1, dimension, dtype=float) * df
            f_list.append(f)

        # make grid of spatial frequencies
        fx_tuple = xp.meshgrid(*f_list)

        return fx_tuple

    @staticmethod
    def fourier_mask(frequencies, coordinates, radii, cosine_mask=False):
        """
        Method to create a mask in Fourier space, centered at coordinates
        Parameters
        ----------
        frequencies: tuple of arrays, or array
            tuple of spatial frequency arrays. All arrays must be the same shape.
        coordinates: tuple of floats, or float
            tuple of coordinates in spatial frequency. Same units as frequencies. Must be same length as frequencies.
        radii: tuple of floats, or float
            If this has length greater than 1, the mask is elliptical. If length is 1, then the mask is circular.
        cosine_mask: bool
            Whether or not to multiply mask by cosine filter

        Returns
        -------
        array of same shape as any of the arrays in frequencies.
        """
        # xp = cp.get_array_module(frequencies)

        # check if frequencies is a tuple
        if type(frequencies) is tuple:
            array_size = xp.shape(frequencies[0])
            num_arrays = len(frequencies)
        else:
            array_size = xp.shape(frequencies)
            num_arrays = 1

        # check size of locations tuple
        if type(coordinates) is tuple:
            num_coords = len(coordinates)
        else:
            num_coords = 1

        # enforce that there are at least as many locations
        if num_coords < num_arrays:
            raise ValueError('Number of locations does not match number of frequency arrays.')
        elif num_coords > num_arrays:
            # if there are more coordinates than arrays just take the same number as the arrays
            coordinates = coordinates[0:num_arrays]
            num_coords = num_arrays

        # check size of widths
        if type(radii) is tuple:
            first_width = radii[0]
            num_widths = len(radii)
        else:
            first_width = radii
            num_widths = 1

        # if number of widths doesn't match number of coordinates, just take the first width and copy it
        if num_widths != num_coords:
            radii = [first_width] * num_coords

        # initialize left hand side of inequality
        lhs = xp.zeros(array_size)

        # loop through coordinates
        for f, c, r in zip(frequencies, coordinates, radii):
            lhs += ((f - c)/r)**2

        # define mask
        mask = (lhs < 1).astype(float)

        # add cosine filter if desired
        if cosine_mask:
            for f, c, r in zip(frequencies, coordinates, radii):
                mask *= xp.cos(np.pi/2*(f-c)/r)

        return mask

    @staticmethod
    def fourier_downsampling(array_in, downsampling):
        """
        Method to perform fourier transform-based downsampling
        Parameters
        ----------
        array_in: (N,M) ndarray
            array to be downsampled
        downsampling: int
            amount to downsample

        Returns
        -------
        (N/downsampling,M/downsampling) ndarray
        """
        # xp = cp.get_array_module(array_in)

        # get array shape
        N, M = xp.shape(array_in)

        # fourier transform
        fourier_plane = Util.nfft(array_in)
        # crop fourier array

        cropped_array = Util.crop_center(fourier_plane, M/downsampling, N/downsampling)

        # ifft back to real space
        array_out = Util.infft(cropped_array)

        return array_out

    @staticmethod
    def crop_center(array_in, x_width, y_width):
        """
        Method to crop out the center of an array
        Parameters
        ----------
        array_in: (N,M) ndarray
            array to be cropped
        x_width: int
            resulting horizontal size of output
        y_width: int
            resulting vertical size of output

        Returns
        -------
        (y_width,x_width) ndarray
        """
        # xp = cp.get_array_module(array_in)

        N, M = xp.shape(array_in)

        cropped_array = array_in[int(N / 2 - y_width/2):int(N / 2 + y_width/2),
                                 int(M / 2 - x_width/2):int(M / 2 + x_width/2)]

        return cropped_array

    @staticmethod
    def normalize_trace(y_data):

        norm_data = (y_data - np.min(y_data))/(np.max(y_data) - np.min(y_data))

        return norm_data

    @staticmethod
    def gaussian_stats(x_data, y_data):

        # normalize input (and subtract any offset)
        y_norm = Util.normalize_trace(y_data)
        # threshold input
        y_data_thresh = Util.threshold_array(y_norm, 0.1)

        # calculate centroid
        cx = np.sum(y_data_thresh * x_data) / np.sum(y_data_thresh)

        # calculate second moment
        sx = np.sqrt(np.sum(y_data_thresh * (x_data - cx) ** 2) / np.sum(y_data_thresh))
        fwx_guess = sx * 2.355

        guess = [cx, sx]

        try:
            mask = y_data_thresh > 0
            px, pcovx = optimize.curve_fit(Util.fit_gaussian, x_data[mask], y_norm[mask],p0=guess)
            sx = px[1]
            cx = px[0]
        except:
            print('Fit failed. Using second moment for width.')

        return cx, sx

    @staticmethod
    def get_k(elevation, azimuth):
        x = np.array([1, 0, 0], dtype=float)
        y = np.array([0, 1, 0], dtype=float)
        z = np.array([0, 0, 1], dtype=float)

        r1 = transform.Rotation.from_rotvec(-x * elevation)
        Rx = r1.as_matrix()
        x = np.matmul(Rx, x)
        y = np.matmul(Rx, y)
        z = np.matmul(Rx, z)

        r2 = transform.Rotation.from_rotvec(y * azimuth)
        Ry = r2.as_matrix()
        x = np.matmul(Ry, x)
        y = np.matmul(Ry, y)
        z = np.matmul(Ry, z)

        # beam points in z direction
        k = z
        return k

    @staticmethod
    def rotate_3d(xhat, yhat, zhat, delta=0, dir='azimuth'):

        if dir=='elevation':
            # an "elevation" rotation corresponds to a rotation about the xhat unit vector
            r1 = transform.Rotation.from_rotvec(-xhat * delta)
            Rx = r1.as_matrix()
            xhat = np.matmul(Rx, xhat)
            yhat = np.matmul(Rx, yhat)
            zhat = np.matmul(Rx, zhat)
        elif dir=='azimuth':
            # an azimuth rotation corresponds to a rotation about the yhat unit vector
            r2 = transform.Rotation.from_rotvec(yhat * delta)
            Ry = r2.as_matrix()
            xhat = np.matmul(Ry, xhat)
            yhat = np.matmul(Ry, yhat)
            zhat = np.matmul(Ry, zhat)

        return xhat, yhat, zhat

    @staticmethod
    def rotate_3d_trace(xhat, yhat, zhat, delta=0, dir='azimuth'):

        if dir=='elevation':
            # an "elevation" rotation corresponds to a rotation about the xhat unit vector
            r1 = transform.Rotation.from_rotvec(xhat * delta)
            Rx = r1.as_matrix()
            xhat = np.matmul(Rx, xhat)
            yhat = np.matmul(Rx, yhat)
            zhat = np.matmul(Rx, zhat)
        elif dir=='azimuth':
            # an azimuth rotation corresponds to a rotation about the yhat unit vector
            r2 = transform.Rotation.from_rotvec(yhat * delta)
            Ry = r2.as_matrix()
            xhat = np.matmul(Ry, xhat)
            yhat = np.matmul(Ry, yhat)
            zhat = np.matmul(Ry, zhat)
        elif dir=='roll':
            # a roll rotation corresponds to a rotation about the zhat unit vector
            r2 = transform.Rotation.from_rotvec(zhat * delta)
            Ry = r2.as_matrix()
            xhat = np.matmul(Ry, xhat)
            yhat = np.matmul(Ry, yhat)
            zhat = np.matmul(Ry, zhat)

        return xhat, yhat, zhat

    @staticmethod
    def rotate_about_point(device, point, rot_vec):
        re = transform.Rotation.from_rotvec(rot_vec)
        Re = re.as_matrix()

        device_pos = Util.get_pos(device)
        new_pos = np.matmul(Re, device_pos - point) + point

        if hasattr(device,'normal'):
            device.normal = np.matmul(Re, device.normal)
            device.sagittal = np.matmul(Re, device.sagittal)
            device.tangential = np.matmul(Re, device.tangential)
        else:
            device.xhat = np.matmul(Re, device.xhat)
            device.yhat = np.matmul(Re, device.yhat)
            device.zhat = np.matmul(Re, device.zhat)

        device.global_x = new_pos[0]
        device.global_y = new_pos[1]
        device.z = new_pos[2]

        return new_pos

    @staticmethod
    def get_pos(device):
        pos_vec = np.zeros((3))
        pos_vec[0] = device.global_x
        pos_vec[1] = device.global_y
        pos_vec[2] = device.z

        return pos_vec

    @staticmethod
    def plan_checkerboard(wfs_obj=None, ppm_obj=None, photon_energy=None, f0=None,
                          zT=None, fraction=1):

        if wfs_obj is not None and ppm_obj is not None:
            # distance from focus to grating
            R1 = wfs_obj.f0
            # "Talbot" distance
            zT = ppm_obj.z - wfs_obj.z
        elif f0 is not None and zT is not None:
            R1 = f0
        else:
            print("Provide either WFS/PPM objects or distances (f0, zT)")
        # distance from focus to imager
        R2 = R1 + zT
        lambda0 = 1239.8/photon_energy*1e-9

        d = np.sqrt(lambda0/fraction*8*R1*(1-R1/R2))

        return d

    @staticmethod
    def integrate_gradient(g, h, x=None, y=None):
        """
        Method to integrate a 2D gradient.
        Parameters
        ----------
        g: (N,M) ndarray
            horizontal gradient term (df_dx)
        h: (N,M) ndarray
            vertical gradient term (df_dy)
        x: (M) ndarray
            horizontal coordinates (1D)
        y: (N) ndarray
            vertical coordinates (1D)

        Returns
        -------
        f: (N,M) ndarray
            the scalar field corresponding to the integrated gradient
        """
        if x is None:
            dx = 1
        else:
            dx = x[1]-x[0]
        if y is None:
            dy = 1
        else:
            dy = y[1]-y[0]

        # calculate second order mixed partial derivatives
        dg_dy = np.gradient(g, dy, axis=0)
        dh_dx = np.gradient(h, dx, axis=1)

        # integrate the gradient (average mixed partial derivatives since these are
        # supposed to be equal)
        f = (cumulative_trapezoid(g, axis=1, dx=dx, initial=0) +
             cumulative_trapezoid(h, axis=0, dx=dy, initial=0) -
            cumulative_trapezoid(cumulative_trapezoid(.5*(dg_dy+dh_dx), axis=0, dx=dy, initial=0),
                     axis=1, dx=dx, initial=0))

        return f

    @staticmethod
    def image_distance(s, p, q):
        if p + q == 0:
            i = -s
        else:
            f = p * q / (p + q)
            i = 1 / (1 / f - 1 / s)

        return i

    @staticmethod
    def lens_image_distance(s, f):
        if f == 0:
            i = -s
        else:
            i = 1 / (1 / f - 1 / s)

        return i

    @staticmethod
    def effective_focus(p, q):
        f = p * q / (p + q)
        return f


class LegendreUtil:

    def __init__(self, x, y, deg, recenter=True):
        self.N = x.size
        # if self.N > 0
        #     self.dx = np.abs(x[1] - x[0])
        # else:
        #     self.dx = 0
        self.x_center = np.mean(x)
        if recenter:
            self.x_centered = x - self.x_center
        else:
            self.x_centered = x

        if self.N > 0:
            self.scale = np.max(np.abs(self.x_centered))
        else:
            self.scale = 1

        self.x_norm = self.x_centered / self.scale
        self.x = x
        self.deg = deg

        if self.N > 0:
            self.c = np.polynomial.legendre.legfit(self.x_norm, y, deg)
        else:
            self.c = np.zeros(deg+1)

    def legint(self, m):
        if self.N > 0:
            self.c = np.polynomial.legendre.legint(self.c, m)*(self.scale)**m
            self.deg += m
        else:
            self.deg += m
            self.c = np.zeros(self.deg+1)

    def legder(self, m):
        if self.N > 0:
            self.c = np.polynomial.legendre.legder(self.c, m)/(self.scale)**m
            self.deg -= m
        else:
            self.deg -= m
            self.c = np.zeros(self.deg+1)

    def legval(self, deg=None):

        if deg is None:
            deg = self.deg

        return np.polynomial.legendre.legval(self.x_norm, self.c[:deg+1])

    def quad_coeff(self):

        return self.c[2] * 3 / 2 / (self.scale) ** 2

    def linear_coeff(self):

        return self.c[1] / (self.scale)
